import os
import math
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image, ImageOps

# 导入本地模型代码
# 确保 modeling_deepseekocr.py 在同一目录下
from models.modeling_deepseekocr import (
    DeepseekOCRForCausalLM,
    dynamic_preprocess,
    BasicImageTransform
)

# ==========================================
# 1. 配置参数
# ==========================================
MODEL_PATH = "deepseek-ai/DeepSeek-OCR" # 或者是本地路径
DATA_ROOT = "./data" # 图片存放根目录
MAX_LENGTH = 2048
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815 # DeepSeek-OCR 的 image token id
PAD_TOKEN_ID = 100015 # 或者是 tokenizer.pad_token_id
IGNORE_INDEX = -100

# ==========================================
# 2. 自定义 Dataset
# ==========================================
class DeepSeekOCRDataset(Dataset):
    def __init__(self, data_list, tokenizer, image_root, max_length=2048):
        """
        data_list: List[Dict], e.g., [{"image": "1.jpg", "prompt": "OCR this", "response": "Content..."}]
        """
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.max_length = max_length
        
        # 图像预处理参数 (参考 infer 函数)
        self.image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        self.base_size = 1024
        self.image_size = 640
        self.patch_size = 16
        self.downsample_ratio = 4

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = os.path.join(self.image_root, item['image'])
        prompt_text = item['prompt']
        response_text = item['response']

        # 1. 加载和处理图像
        try:
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个 dummy 数据或者抛出异常
            return self.__getitem__((idx + 1) % len(self))
        # /Users/zhangyf/PycharmProjects/cfel/deepseekocr/data/images/00009_hr-0.png

        # 动态分辨率预处理
        # 逻辑复用自 modeling_deepseekocr.py 的 infer 方法
        images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=self.image_size)
        width_crop_num, height_crop_num = crop_ratio
        
        # Global View
        global_view = ImageOps.pad(image, (self.base_size, self.base_size),
                                   color=tuple(int(x * 255) for x in self.image_transform.mean))
        global_view_tensor = self.image_transform(global_view).to(torch.bfloat16) # [3, H, W]
        
        # Local Patches
        patches_list = []
        for patch in images_crop_raw:
            patches_list.append(self.image_transform(patch).to(torch.bfloat16))
        
        if patches_list:
            patches_tensor = torch.stack(patches_list, dim=0) # [N, 3, H, W]
        else:
            # Fallback (should rarely happen with dynamic_preprocess)
            patches_tensor = torch.zeros((1, 3, self.image_size, self.image_size), dtype=torch.bfloat16)

        # 构造 images 输入: (patches, global_view)
        # 注意: ori_image 需要是 [1, 3, H, W]
        image_tuple = (patches_tensor, global_view_tensor.unsqueeze(0))

        # 2. 构建 Prompt 和 Tokens
        # 我们需要手动插入 Image Token 占位符，以便模型 forward 时能够替换特征
        
        # 计算占位符数量
        num_queries = math.ceil((self.image_size // self.patch_size) // self.downsample_ratio) # 10
        num_queries_base = math.ceil((self.base_size // self.patch_size) // self.downsample_ratio) # 16
        
        # 构造 Image Tokens 序列
        # Base tokens
        tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
        tokenized_image += [IMAGE_TOKEN_ID]
        
        # Crop tokens
        if width_crop_num > 1 or height_crop_num > 1:
            tokenized_image += ([IMAGE_TOKEN_ID] * (num_queries * width_crop_num) + [IMAGE_TOKEN_ID]) * (
                        num_queries * height_crop_num)
        
        # 拼接文本
        # 格式: <bos> User: <image_placeholder> \n Prompt \n Assistant: Response <eos>
        # 这里简化处理，你可以根据需要调整 Chat Template

        user_prefix_ids = self.tokenizer.encode("<|User|>", add_special_tokens=False)
        assistant_prefix_ids = self.tokenizer.encode("<|Assistant|>", add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]

        # 组合 Input IDs
        # [User] [Image_Tokens] [Prompt] [Assistant] [Response]
        input_ids = (
            [self.tokenizer.bos_token_id] +
            user_prefix_ids +
            tokenized_image +
            prompt_ids +
            assistant_prefix_ids +
            response_ids
        )

        # 组合 Labels (User 部分设为 -100)
        context_len = 1 + len(user_prefix_ids) + len(tokenized_image) + len(prompt_ids) + len(assistant_prefix_ids)
        labels = [IGNORE_INDEX] * context_len + response_ids

        # 组合 Images Seq Mask (Image Token 位置为 True)
        # Mask 长度必须与 input_ids 一致
        seq_mask = [False] * (1 + len(user_prefix_ids)) + \
                   [True] * len(tokenized_image) + \
                   [False] * (len(prompt_ids) + len(assistant_prefix_ids) + len(response_ids))

        # 截断 (简单截断，实际可能需要更复杂的逻辑)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            seq_mask = seq_mask[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images": image_tuple, # (Patches, Global)
            "images_seq_mask": torch.tensor(seq_mask, dtype=torch.bool),
            "images_spatial_crop": torch.tensor([width_crop_num, height_crop_num], dtype=torch.long)
        }

# ==========================================
# 3. 自定义 DataCollator
# ==========================================
class DeepSeekOCRCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        images_seq_mask = [f["images_seq_mask"] for f in features]
        
        # Padding
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        images_seq_mask_padded = torch.nn.utils.rnn.pad_sequence(
            images_seq_mask, batch_first=True, padding_value=False
        )
        
        attention_mask = input_ids_padded.ne(self.pad_token_id)

        # 图像数据直接作为列表传递
        batch_images = [f["images"] for f in features]
        
        # Spatial Crop 信息 stack
        images_spatial_crop = torch.stack([f["images_spatial_crop"] for f in features])

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
            "images": batch_images,
            "images_seq_mask": images_seq_mask_padded,
            "images_spatial_crop": images_spatial_crop,
            "return_dict": True
        }

# ==========================================
# 4. 主训练流程
# ==========================================
def train():
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    # 注意: 显存不足时可以使用 load_in_4bit=True (需要 bitsandbytes)
    # 使用本地导入的 DeepseekOCRForCausalLM 类直接加载
    model = DeepseekOCRForCausalLM.from_pretrained(
        MODEL_PATH,
        # trust_remote_code=True, # 使用本地类时通常不需要，除非 config 也是远程且未本地导入
        torch_dtype=torch.bfloat16,
        # _attn_implementation="flash_attention_2"
    )
    print(model)

    # 冻结 Vision Encoder (通常不需要微调 SAM 和 CLIP)
    for name, param in model.named_parameters():
        if "sam_model" in name or "vision_model" in name:
            param.requires_grad = False
    
    # 启用 LoRA (推荐)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 准备数据 (示例)
    train_data = [
        {
            "image": "example1.jpg", 
            "prompt": "<|grounding|>Extract text.", 
            "response": "This is example text."
        },
        # 添加更多数据...
    ]
    
    dataset = DeepSeekOCRDataset(train_data, tokenizer, image_root=DATA_ROOT)
    collator = DeepSeekOCRCollator(tokenizer)

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=2, # 根据显存调整
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True, # 推荐使用 bf16
        remove_unused_columns=False, # 必须为 False，否则 'images' 字段会被过滤掉
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model("./final_model")

if __name__ == "__main__":
    # train()
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 准备数据 (示例)
    train_data = [
        {
            "image": "/Users/zhangyf/PycharmProjects/cfel/deepseek-ocr/data/images/00009_hr-0.png",
            "prompt": "<|grounding|>Extract text.",
            "response": "This is example text."
        },
    ]

    dataset = DeepSeekOCRDataset(train_data, tokenizer, image_root="/Users/zhangyf/PycharmProjects/cfel/deepseek-ocr/data/images/")
    result = dataset[0]
    print(result)