import json
import torch
import math
import io
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch.utils.data import Dataset

# 【修改点1】移除错误的 text_encode 引用
from models.modeling_qwen_ocr import (
    BasicImageTransform,
    dynamic_preprocess,
    QwenOCRForCausalLM
)


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
                        self.data.append(json.loads(line))
            print(f"成功加载 {len(self.data)} 条数据。")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {jsonl_file}")
            self.data = []

        # 【修改点2】保持原始角色名，不要加 <|User|> 等装饰，后续在 Collator 中统一处理
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
        new_messages = []
        raw_messages = raw_item.get("messages", [])

        for i, msg in enumerate(raw_messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # 简单映射，确保是 user/assistant
            role = self.role_map.get(role, role)

            new_msg = {
                "role": role,
                "content": content
            }

            # 将图片注入到第一条消息（通常是User）
            if i == 0 and image_paths:
                new_msg["images"] = image_paths

            new_messages.append(new_msg)

        return {"messages": new_messages}


@dataclass
class QwenOCRDataCollator:
    """
    适配 Qwen ChatML 格式 (<|im_start|>...) 的 DataCollator
    """
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 151655  # Qwen <|image_pad|>
    train_on_responses_only: bool = True
    ignore_index: int = -100

    def __init__(
            self,
            tokenizer,
            model,
            image_size: int = 640,
            base_size: int = 1024,
            crop_mode: bool = True,
            train_on_responses_only: bool = True,
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

        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        # 【修改点3】动态获取 Qwen 特殊 Token ID，不再硬编码 0/1
        # Qwen tokenizer_config.json/vocab.json 对应的值：
        # <|im_start|> = 151644
        # <|im_end|>   = 151645
        # \n           = 198 (通常情况)
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # 获取换行符的 token id
        nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
        self.nl_id = nl_tokens[-1] if nl_tokens else 198

        # 验证一下，防止取不到
        if self.im_start_id is None: self.im_start_id = 151644
        if self.im_end_id is None: self.im_end_id = 151645

    def deserialize_image(self, image_data) -> Image.Image:
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            return Image.open(io.BytesIO(image_data['bytes'])).convert("RGB")
        elif isinstance(image_data, str):
            return Image.open(image_data).convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

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

            # 构建占位符 token 序列
            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                        num_queries * height_crop_num)

        else:
            # Non-crop mode logic
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

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        处理单条样本，构建符合 Qwen ChatML 格式的输入：
        <|im_start|>system\n...<|im_end|>\n
        <|im_start|>user\n...<|im_end|>\n
        <|im_start|>assistant\n...<|im_end|>\n
        """

        # 1. 预加载图片
        images = []
        for message in messages:
            if "images" in message and message["images"]:
                for img_path in message["images"]:
                    if img_path:
                        images.append(self.deserialize_image(img_path))

        input_ids = []
        labels = []  # 用于计算 loss
        images_seq_mask = []

        images_list, images_crop_list, images_spatial_crop = [], [], []
        image_idx = 0

        # Qwen 不需要句首的 BOS (151643)，直接以 im_start 开始

        for message in messages:
            role = message["role"]
            content = message["content"]

            # 替换 dataset 中的 <image> 为切割符
            content = content.replace("<image>", "<|image_pad|>")

            # --- 构建 Header: <|im_start|>role\n ---
            role_ids = self.tokenizer.encode(role, add_special_tokens=False)
            header_ids = [self.im_start_id] + role_ids + [self.nl_id]

            # --- 构建 Content ---
            content_ids = []
            content_mask_flags = []  # True 表示是图片token，需要mask

            text_splits = content.split('<|image_pad|>')
            for i, text_sep in enumerate(text_splits):
                if text_sep:
                    # 文本部分
                    sep_ids = self.tokenizer.encode(text_sep, add_special_tokens=False)
                    content_ids.extend(sep_ids)
                    content_mask_flags.extend([False] * len(sep_ids))  # 文本部分不是图片占位符

                if i < len(text_splits) - 1:
                    # 图片部分
                    if image_idx >= len(images):
                        raise ValueError("Found '<image>' token but no corresponding image.")

                    image = images[image_idx]
                    img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial_crop)

                    content_ids.extend(tok_img)
                    content_mask_flags.extend([True] * len(tok_img))  # 标记为图片token
                    image_idx += 1

            # --- 构建 Footer: <|im_end|>\n ---
            footer_ids = [self.im_end_id, self.nl_id]

            # --- 拼接这一轮的完整 IDs ---
            current_ids = header_ids + content_ids + footer_ids
            input_ids.extend(current_ids)

            # --- 构建 Labels (Masking 逻辑) ---
            # 1. Image Token 位置对应的 mask (seq_mask)
            # Header 和 Footer 都不是图片
            current_img_mask = [False] * len(header_ids) + content_mask_flags + [False] * len(footer_ids)
            images_seq_mask.extend(current_img_mask)

            # 2. Loss Labels
            if role == "assistant":
                # Assistant 的回复需要训练
                # Header 部分 mask 掉
                current_labels = [self.ignore_index] * len(header_ids)

                # Content 部分保留 (除了图片token)
                # 图片 token 在 collator 最后一步会再次统一用 images_seq_mask 盖住，
                # 这里先保留 content_ids，如果这里设为 -100 也可以，双重保险。
                current_labels.extend(content_ids)

                # Footer 部分 (<|im_end|>) 需要训练，让模型学会停止
                current_labels.extend(footer_ids)
            else:
                # User / System 的话全部 mask
                current_labels = [self.ignore_index] * len(current_ids)

            labels.extend(current_labels)

        # Final checks
        if images and image_idx != len(images):
            # 简单的容错，或者 print warning
            pass

        # Prepare Tensors
        if images_list:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)
        else:
            images_ori = torch.zeros((0, 3, self.image_size, self.image_size), dtype=self.dtype)
            images_spatial_crop_tensor = torch.empty((0, 2), dtype=torch.long)
            images_crop = torch.zeros((0, 3, self.base_size, self.base_size), dtype=self.dtype)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_data = []
        for feature in features:
            try:
                processed = self.process_single_sample(feature['messages'])
                batch_data.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        input_ids = pad_sequence([x['input_ids'] for x in batch_data], batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence([x['labels'] for x in batch_data], batch_first=True, padding_value=self.ignore_index)
        images_seq_mask = pad_sequence([x['images_seq_mask'] for x in batch_data], batch_first=True,
                                       padding_value=False)

        # 双重保险：将所有 padding 位置和 image token 位置的 label 设为 -100
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        labels[images_seq_mask] = self.ignore_index

        # 聚合图片
        images_batch = []
        for item in batch_data:
            images_batch.append((item['images_crop'], item['images_ori']))

        if batch_data and batch_data[0]['images_spatial_crop'].numel() > 0:
            images_spatial_crop = torch.cat([item['images_spatial_crop'] for item in batch_data], dim=0)
        else:
            images_spatial_crop = torch.empty((0, 2), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }


if __name__ == '__main__':
    # 请确保路径正确
    MODEL_PATH = "/Users/zhangyf/llm/Qwen2.5-0.5B-Instruct"
    data_path = "/checkpoint/tt.jsonl"

    # 初始化
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QwenOCRForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)

    dataset = ECGInstructionDataset(data_path)
    data_collator = QwenOCRDataCollator(tokenizer=tokenizer, model=model)

    # if len(dataset) > 0:
    #     batch = data_collator([dataset[0]])
    #     print("Input IDs:", batch['input_ids'].shape)
    #     print("Labels:", batch['labels'].shape)
    #
    #     # 简单验证解码 (只打印前50个token)
    #     ids = batch['input_ids'][0].tolist()
    #     print("Decoded start:", tokenizer.decode(ids[:50]))
    #
    #     # 验证 Labels 是否正确 mask 掉了 user 部分
    #     lbs = batch['labels'][0].tolist()
    #     # 找到第一个不为 -100 的地方
    #     try:
    #         first_trainable = next(i for i, x in enumerate(lbs) if x != -100)
    #         print(f"Training starts at index {first_trainable}, token: {tokenizer.decode([ids[first_trainable]])}")
    #     except StopIteration:
    #         print("Warning: All labels are masked!")
    batch = data_collator(dataset)
    # output = model.forward(**batch)
    # out = model.infer()
    # print(out)
