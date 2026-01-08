import json
import os
import numpy as np
import scipy
import scipy.signal
import torch
import math
import io
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch.utils.data import Dataset

# 请确保 models 文件夹下有对应的文件

from models.deepencoder import build_sam_vit_b, build_clip_l
from models.modeling_deepencode_ecg_qwen3 import (
    BasicImageTransform,
    dynamic_preprocess,
    DeepencoderEcgOCRForCausalLM
)
from models.ecgencoder import _build_ecg_tower
try:
    import wfdb
except ImportError:
    print("Warning: 'wfdb' library not found. Please install it using `pip install wfdb` to load .dat/.hea files.")
    wfdb = None


class ECGInstructionDataset(Dataset):
    def __init__(self, jsonl_file):
        """
        初始化数据集。
        """
        self.data = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if "messages" in item:
                                self.data.append(item)
                        except json.JSONDecodeError:
                            pass
            print(f"成功加载 {len(self.data)} 条数据。")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {jsonl_file}")
            self.data = []

        # 保持 Qwen 的角色名，不要加 <|User|> 等装饰，后续在 Collator 中统一处理
        self.role_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system"
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_item = self.data[idx]
        image_paths = raw_item.get("images", [])
        ecg_file = raw_item.get("ecg_file", "")  # 获取 ECG 文件路径

        new_messages = []
        raw_messages = raw_item.get("messages", [])

        for i, msg in enumerate(raw_messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            role = self.role_map.get(role, role)

            new_msg = {
                "role": role,
                "content": content
            }

            # 将图片注入到第一条消息
            if i == 0 and image_paths:
                new_msg["images"] = image_paths

            new_messages.append(new_msg)

        return {
            "messages": new_messages,
            "ecg_file": ecg_file
        }


@dataclass
class QwenOCRDataCollator:
    """
    适配 Qwen ChatML 格式 (<|im_start|>...) 且支持 ECG 信号注入的 DataCollator
    """
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 151655  # Qwen <|image_pad|>
    train_on_responses_only: bool = True
    ignore_index: int = -100

    # ECG 参数
    ecg_root_path: str = ""
    ecg_token_len: int = 101
    ecg_dropout_prob: float = 0.0

    def __init__(
            self,
            tokenizer,
            model,
            image_size: int = 640,
            base_size: int = 1024,
            crop_mode: bool = True,
            train_on_responses_only: bool = True,
            # ECG数据参数
            ecg_root_path: str = "",
            ecg_token_len: int = 101,
            ecg_dropout_prob: float = 0.0,
            max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 151655
        self.dtype = model.dtype
        self.train_on_responses_only = train_on_responses_only
        self.ignore_index = -100
        self.max_length = max_length

        # ECG 设置
        self.ecg_root_path = ecg_root_path
        self.ecg_token_len = ecg_token_len
        self.ecg_dropout_prob = ecg_dropout_prob

        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        # 动态获取 Qwen 特殊 Token ID
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
        self.nl_id = nl_tokens[-1] if nl_tokens else 198

        if self.im_start_id is None: self.im_start_id = 151644
        if self.im_end_id is None: self.im_end_id = 151645

    def load_ecg_data(self, rel_path: str) -> Optional[torch.Tensor]:
        """
        读取 .dat/.hea 文件并预处理为 [12, 5000] 的 Tensor
        移植自 fix_ecg_dataset.py
        """
        if not wfdb:
            return None

        full_path = os.path.join(self.ecg_root_path, rel_path)
        record_path = os.path.splitext(full_path)[0]

        if not (os.path.exists(record_path + ".hea") or os.path.exists(record_path + ".dat")):
            if not (os.path.exists(full_path + ".hea")):
                return None
            record_path = full_path

        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal  # (N, 12)

            # NaN/Inf 清洗
            if np.isnan(signal).any() or np.isinf(signal).any():
                signal[np.isnan(signal)] = 0
                signal[np.isinf(signal)] = 0

            original_fs = record.fs
            target_fs = 500
            target_len = 5000  # 10秒

            # 1. 重采样
            if original_fs != target_fs:
                new_len = int(signal.shape[0] * target_fs / original_fs)
                signal = scipy.signal.resample(signal, new_len, axis=0)

            # 2. 截断或填充
            L, C = signal.shape
            if L > target_len:
                signal = signal[:target_len, :]
            elif L < target_len:
                pad_len = target_len - L
                padding = np.zeros((pad_len, C))
                signal = np.concatenate([signal, padding], axis=0)

            # 3. 转置为 [12, 5000]
            signal_tensor = torch.tensor(signal.T, dtype=torch.bfloat16)

            # 4. Instance Normalization (Z-Score)
            mean = signal_tensor.mean(dim=1, keepdim=True)
            std = signal_tensor.std(dim=1, keepdim=True) + 1e-5
            signal_tensor = (signal_tensor - mean) / std

            if torch.isnan(signal_tensor).any():
                return torch.zeros((12, 5000), dtype=torch.bfloat16)

            return signal_tensor

        except Exception as e:
            print(f"Error loading ECG {record_path}: {e}")
            return None

    def deserialize_image(self, image_data) -> Image.Image:
        try:
            if isinstance(image_data, Image.Image):
                return image_data.convert("RGB")
            elif isinstance(image_data, str):
                return Image.open(image_data).convert("RGB")
            elif isinstance(image_data, dict) and 'bytes' in image_data:
                image_bytes = image_data['bytes']
                image = Image.open(io.BytesIO(image_bytes))
                return image.convert("RGB")
        except Exception as e:
            return Image.new("RGB", (self.image_size, self.image_size))

    def process_image(self, image: Image.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num=2, max_num=9,
                    image_size=self.image_size, use_thumbnail=False
                )

            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_transform(crop_img).to(self.dtype))

            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                        num_queries * height_crop_num)

        else:
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])
            if self.base_size <= 640:
                resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean)
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单条样本，融合 ECG 和 Qwen ChatML 格式
        """
        messages = item_dict.get('messages', [])
        ecg_file = item_dict.get('ecg_file', "")

        # --- 1. 预加载图片 ---
        images = []
        for message in messages:
            if "images" in message and message["images"]:
                for img_path in message["images"]:
                    if img_path:
                        images.append(self.deserialize_image(img_path))

        # --- 2. 准备 ECG 数据 ---
        ecg_tensor = torch.zeros((12, 5000), dtype=torch.bfloat16)
        has_ecg = False
        drop_ecg = (self.ecg_dropout_prob > 0) and (random.random() < self.ecg_dropout_prob)

        if ecg_file and not drop_ecg:
            loaded_ecg = self.load_ecg_data(ecg_file)
            if loaded_ecg is not None:
                ecg_tensor = loaded_ecg
                has_ecg = True

        # --- 3. 构建 ID 和 Mask ---
        input_ids = []
        labels = []
        images_seq_mask = []
        ecg_seq_mask = []  # 新增: 记录哪些位置是 ECG

        images_list, images_crop_list, images_spatial_crop = [], [], []
        image_idx = 0
        ecg_injected = False  # 标记 ECG 是否已注入

        for message in messages:
            role = message["role"]
            content = message["content"]

            content = content.replace("<image>", "<|image_pad|>")

            # --- 构建 Header: <|im_start|>role\n ---
            role_ids = self.tokenizer.encode(role, add_special_tokens=False)
            header_ids = [self.im_start_id] + role_ids + [self.nl_id]

            # Header 的 Mask 设置
            header_len = len(header_ids)
            images_seq_mask.extend([False] * header_len)
            ecg_seq_mask.extend([False] * header_len)

            # --- 构建 Content ---
            content_ids = []
            content_img_mask = []
            content_ecg_mask = []

            # >>> 核心逻辑：ECG 占位符注入 (在第一个 User 回合) <<<
            if has_ecg and role == "user" and not ecg_injected:
                # 按照 DeepSeek 逻辑：1 (Start) + 101 (Features) + 1 (End) = 103
                total_ecg_tokens = 1 + self.ecg_token_len + 1
                # 使用 image_token_id 作为占位符，或者如果模型有专用 token 可替换
                ecg_tokens = [self.image_token_id] * total_ecg_tokens

                content_ids.extend(ecg_tokens)
                content_img_mask.extend([False] * total_ecg_tokens)  # ECG 不是图片，不触发 Image Mask
                content_ecg_mask.extend([True] * total_ecg_tokens)  # 触发 ECG Mask

                ecg_injected = True

            text_splits = content.split('<|image_pad|>')
            for i, text_sep in enumerate(text_splits):
                if text_sep:
                    sep_ids = self.tokenizer.encode(text_sep, add_special_tokens=False)
                    content_ids.extend(sep_ids)
                    content_img_mask.extend([False] * len(sep_ids))
                    content_ecg_mask.extend([False] * len(sep_ids))

                if i < len(text_splits) - 1:
                    # 处理图片
                    if image_idx >= len(images):
                        # 容错：如果没有图片了，就只补一个占位符，防止崩溃
                        content_ids.append(self.image_token_id)
                        content_img_mask.append(True)
                        content_ecg_mask.append(False)
                    else:
                        image = images[image_idx]
                        img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                        images_list.extend(img_list)
                        images_crop_list.extend(crop_list)
                        images_spatial_crop.extend(spatial_crop)

                        content_ids.extend(tok_img)
                        content_img_mask.extend([True] * len(tok_img))  # 标记为图片
                        content_ecg_mask.extend([False] * len(tok_img))
                        image_idx += 1

            # --- 构建 Footer: <|im_end|>\n ---
            footer_ids = [self.im_end_id, self.nl_id]
            footer_len = len(footer_ids)

            # --- 合并当前 Turn ---
            current_ids = header_ids + content_ids + footer_ids
            input_ids.extend(current_ids)

            # Extend Masks
            images_seq_mask.extend(content_img_mask + [False] * footer_len)
            ecg_seq_mask.extend(content_ecg_mask + [False] * footer_len)

            # --- 构建 Labels ---
            if role == "assistant":
                # Header Mask
                current_labels = [self.ignore_index] * header_len
                # Content 需要训练 (除了 Image/ECG，稍后会被 mask 覆盖，这里先填 ids)
                current_labels.extend(content_ids)
                # Footer 需要训练
                current_labels.extend(footer_ids)
            else:
                # User / System 全部 mask
                current_labels = [self.ignore_index] * len(current_ids)

            labels.extend(current_labels)

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            images_seq_mask = images_seq_mask[:self.max_length]
            ecg_seq_mask = ecg_seq_mask[:self.max_length]

        # Prepare Image Tensors
        if images_list:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((len(images_list), 3, self.base_size, self.base_size), dtype=self.dtype)
        else:
            images_ori = torch.zeros((0, 3, self.image_size, self.image_size), dtype=self.dtype)
            images_spatial_crop_tensor = torch.empty((0, 2), dtype=torch.long)
            images_crop = torch.zeros((0, 3, self.base_size, self.base_size), dtype=self.dtype)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
            "ecg_seq_mask": torch.tensor(ecg_seq_mask, dtype=torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
            "ecg_values": ecg_tensor
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_data = []
        for feature in features:
            try:
                processed = self.process_single_sample(feature)
                batch_data.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        # 1. Pad Sequences
        input_ids = pad_sequence([x['input_ids'] for x in batch_data], batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence([x['labels'] for x in batch_data], batch_first=True, padding_value=self.ignore_index)

        images_seq_mask = pad_sequence([x['images_seq_mask'] for x in batch_data], batch_first=True,
                                       padding_value=False)
        ecg_seq_mask = pad_sequence([x['ecg_seq_mask'] for x in batch_data], batch_first=True, padding_value=False)

        # 2. Attention Mask & Labels Cleaning
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # 强制 Mask 掉特殊区域
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        labels[images_seq_mask] = self.ignore_index
        labels[ecg_seq_mask] = self.ignore_index  # ECG 区域也不计算 Loss

        # 3. Aggregate Images
        images_batch = []
        for item in batch_data:
            images_batch.append((item['images_crop'], item['images_ori']))

        if batch_data and batch_data[0]['images_spatial_crop'].numel() > 0:
            images_spatial_crop = torch.cat([item['images_spatial_crop'] for item in batch_data], dim=0)
        else:
            images_spatial_crop = torch.empty((0, 2), dtype=torch.long)

        # 4. Aggregate ECG
        ecg_values = torch.stack([x['ecg_values'] for x in batch_data])
        if torch.isnan(ecg_values).any():
            print("[FATAL] NaN in ecg_values batch, zerofilling.")
            ecg_values = torch.nan_to_num(ecg_values)

        # Debug Stats
        # total_tokens = labels.numel()
        # masked_tokens = (labels == -100).sum().item()
        # print(f"[DEBUG] Batch: Valid Tokens={total_tokens - masked_tokens} / {total_tokens}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
            "ecg_values": ecg_values,  # 模型 forward 需要接收此参数
            "ecg_seq_mask": ecg_seq_mask,  # 模型 forward 需要接收此参数
        }


if __name__ == '__main__':
    # 请确保路径正确
    MODEL_PATH = "/Users/zhangyf/llm/Qwen3-0.6B"  # 示例路径
    data_path = "/Users/zhangyf/PycharmProjects/cfel/deepseekocr/data/deepseek_ocr_ecg_future.jsonl"  # 示例路径

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    dataset = ECGInstructionDataset(data_path)


    # 模拟模型 (如果本地没有模型，注释掉下面这行)
    model = DeepencoderEcgOCRForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)

    print("Freezing Vision Modules (SAM + CLIP)...")
    # 2. 定义需要解冻（训练）的模块关键字
    modules_to_train = [
        "projector",  # 视觉 Projector
        "ecg_projector",  # ECG Projector
        "image_newline",  # 特殊 Token 参数
        "view_seperator",
        "ecg_start_embed",
        "ecg_end_embed",
        "model.layers",
        "model.norm"
    ]

    # 3. 遍历参数，只开启命中关键字的部分
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in modules_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 4. (重要) 再次检查验证
    # 打印所有 requires_grad = True 的参数名，确保没有混入 embed_tokens 或 norm
    print("=== Trainable Parameters ===")
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            print(f"{name}: {param.shape}")

    # # 遍历所有参数
    # for name, param in model.named_parameters():
    #     # QwenOCRModel 定义了 self.sam_model 和 self.vision_model
    #     if "sam_model" in name or "vision_model" in name or "ecg_encoder" in name or "model.layers" in name or "model.norm" in name or "model.embed_tokens" in name:
    #         param.requires_grad = False
    #     else:
    #         # LLM (model.layers), Projector (model.projector), Embeddings 保持训练
    #         param.requires_grad = True
    #
    # # 再次确认 projector 是开启梯度的
    # # model.model.projector 是对齐层
    # for name, param in model.model.projector.named_parameters():
    #     param.requires_grad = True
    # for name, param in model.model.ecg_projector.named_parameters():
    #     param.requires_grad = True

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)

    # 加载 deepencode权重
    model.model.sam_model = build_sam_vit_b().to(dtype=model.dtype)
    model.model.vision_model = build_clip_l().to(dtype=model.dtype)
    root_path = "/Users/zhangyf/llm/DeepEncoder/"
    model.model.sam_model.load_state_dict(torch.load(root_path + "sam_encoder.pth"))
    model.model.vision_model.load_state_dict(torch.load(root_path + "clip_encoder.pth"))
    # 初始化ecgencoder
    model.model.ecg_encoder = _build_ecg_tower().to(dtype=model.dtype)
    missing_keys, unexpected_keys = model.model.ecg_encoder.load_state_dict(torch.load("/Users/zhangyf/PycharmProjects/cfel/deepseekocr/test/ecg_encoder.pth"))
    print(missing_keys, unexpected_keys)
    # 初始化 特殊嵌入token
    embed_std = 1 / torch.sqrt(torch.tensor(model.config.hidden_size, dtype=torch.bfloat16))
    model.model.image_newline = torch.nn.Parameter(torch.randn(model.config.hidden_size, dtype=model.dtype) * embed_std)
    model.model.view_seperator = torch.nn.Parameter(torch.randn(model.config.hidden_size, dtype=model.dtype) * embed_std)
    model.model.ecg_start_embed = torch.nn.Parameter(torch.randn(model.config.hidden_size, dtype=model.dtype) * embed_std)
    model.model.ecg_end_embed = torch.nn.Parameter(torch.randn(model.config.hidden_size, dtype=model.dtype) * embed_std)

    # 初始化 Collator (包含 ECG 设置)
    data_collator = QwenOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        ecg_root_path="/Users/zhangyf/ecg_data/",  # 你的 ECG 数据根目录
        ecg_token_len=101
    )

    # 取一个 Batch 测试
    result = data_collator(dataset)
    #
    # print("\n=== Batch Output Checks ===")
    # print(f"Input IDs Shape: {result['input_ids'].shape}")
    # print(f"Labels Shape: {result['labels'].shape}")
    # print(f"ECG Values Shape: {result['ecg_values'].shape}")
    # print(f"ECG Mask Shape: {result['ecg_seq_mask'].shape}")
    #
    # # 验证 ECG Mask 是否生效 (有 True 值)
    # print(f"Has ECG tokens in mask? {result['ecg_seq_mask'].any().item()}")
    #
    # # 验证 Image 处理
    # if len(result['images']) > 0:
    #     print(f"Image batch size: {len(result['images'])}")
    #
    out = model(**result)
    print(out.loss)

    # prompt = "<image>\n Interpret the provided ECG image, identify key features and abnormalities in each lead, and generate a clinical diagnosis that is supported by the observed evidence."
    # # image_file = './image_ds/gen_images/mimic_gen/p1079/p10792254/s42187992/42187992-0.png'
    # image_file = '/Users/zhangyf/Documents/cfel/images/p1968/p19680126/s41060732/41060732-0.png'
    # ecg_file = '/Users/zhangyf/Documents/cfel/ts_data/files/p1000/p10000032/s40689238/40689238'
    # output_path = './output'
    # #
    # # # 执行推理
    # res = model.infer(
    #     tokenizer=tokenizer,
    #     prompt=prompt,
    #     image_file=image_file,
    #     ecg_file=ecg_file,
    #     output_path=output_path,
    #     base_size=1024,
    #     image_size=640,
    #     crop_mode=True
    # )
    # #
    # model.eval().to("cpu")
    # print(f"result:{res}")