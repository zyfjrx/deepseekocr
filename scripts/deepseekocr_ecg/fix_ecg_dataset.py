# ecg_dataset.py
import json
import torch
import math
import numpy as np
import os
import random
import io
import scipy.signal
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch.utils.data import Dataset

# 导入模型相关依赖
# 请确保您的目录结构中 models/ 文件夹下有对应的文件
from models.modeling_deepseekocr import (
    text_encode,
    BasicImageTransform,
    dynamic_preprocess,
)
from models.ecg_encoder import ECGEncoderWrapper
from models.modeling_deepseekocr_ecg import DeepseekOCRForCausalLM

# 尝试导入 wfdb，处理 .dat/.hea 文件
try:
    import wfdb
except ImportError:
    print("Warning: 'wfdb' library not found. Please install it using `pip install wfdb` to load .dat/.hea files.")
    wfdb = None


class ECGInstructionDataset(Dataset):
    def __init__(self, jsonl_file):
        """
        初始化数据集。
        :param jsonl_file: 原始数据的 jsonl 文件路径
        """
        self.data = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():  # 跳过空行
                        try:
                            item = json.loads(line)
                            # 简单的预检查，确保有 messages
                            if "messages" in item and isinstance(item["messages"], list):
                                self.data.append(item)
                            else:
                                print(f"[Warning] Line {line_num}: Missing 'messages' field.")
                        except json.JSONDecodeError:
                            print(f"[Warning] Line {line_num}: Invalid JSON.")
            print(f"成功加载 {len(self.data)} 条数据。")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {jsonl_file}")
            self.data = []

        # 定义角色映射关系，增强兼容性
        self.role_map = {
            "user": "<|User|>",
            "User": "<|User|>",
            "assistant": "<|Assistant|>",
            "Assistant": "<|Assistant|>",
            "model": "<|Assistant|>"
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单条数据并进行格式转换
        """
        raw_item = self.data[idx]

        # 1. 获取原始图片路径列表
        image_paths = raw_item.get("images", [])

        # 2. 获取 ECG 文件路径
        ecg_file = raw_item.get("ecg_file", "")

        # 3. 构建新的 messages 列表
        new_messages = []
        raw_messages = raw_item.get("messages", [])

        for i, msg in enumerate(raw_messages):
            old_role = msg.get("role", "")
            content = msg.get("content", "")

            # 映射角色
            new_role = self.role_map.get(old_role, old_role)

            new_msg = {
                "role": new_role,
                "content": content
            }

            # 将图片路径注入到第一条 User 消息中
            if i == 0 and image_paths:
                new_msg["images"] = image_paths

            new_messages.append(new_msg)

        # 返回字典，DataCollator 会进一步处理
        return {
            "messages": new_messages,
            "ecg_file": ecg_file
        }


@dataclass
class DeepSeekOCRDataCollator:
    """
    数据整理器，负责处理图像、文本和 ECG 信号的 Batch 组装。
    """
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    train_on_responses_only: bool = True
    ecg_root_path: str = ""  # ECG 数据根目录
    ecg_token_len: int = 101  # ECG Encoder 输出特征长度
    ecg_dropout_prob: float = 0.0  # 模态 Dropout 概率
    max_length: int = 8192  # 最大长度限制

    def __init__(
            self,
            tokenizer,
            model,
            image_size: int = 640,
            base_size: int = 1024,
            crop_mode: bool = True,
            train_on_responses_only: bool = True,
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
        self.image_token_id = 128815  # 使用 <image> token ID 作为通用占位符
        self.dtype = model.dtype
        self.train_on_responses_only = train_on_responses_only
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

        # 获取 BOS token ID
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0
            print(f"Warning: tokenizer has no bos_token_id, using default: {self.bos_id}")

    def load_ecg_data(self, rel_path: str) -> Optional[torch.Tensor]:
        """
        读取 .dat/.hea 文件并预处理为 [12, 5000] 的 Tensor
        包含 NaN 清洗和 Instance Normalization
        """
        if not wfdb:
            return None

        # 拼接完整路径
        full_path = os.path.join(self.ecg_root_path, rel_path)
        record_path = os.path.splitext(full_path)[0]

        # 检查文件是否存在
        if not (os.path.exists(record_path + ".hea") or os.path.exists(record_path + ".dat")):
            if not (os.path.exists(full_path + ".hea")):
                return None
            record_path = full_path

        try:
            # 读取记录
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal  # shape: (Samples, Channels) 通常是 (N, 12)

            # === Step 1: NaN/Inf 清洗
            if np.isnan(signal).any() or np.isinf(signal).any():
                # print(f"[Warning] NaN/Inf detected in {record_path}, cleaning...")
                signal[np.isnan(signal)] = 0
                signal[np.isinf(signal)] = 0

            original_fs = record.fs
            target_fs = 500
            target_len = 5000  # 10秒

            # === Step 2: 重采样 (Resample) ===
            if original_fs != target_fs:
                new_len = int(signal.shape[0] * target_fs / original_fs)
                signal = scipy.signal.resample(signal, new_len, axis=0)

            # === Step 3: 截断或填充 (Pad/Crop) ===
            L, C = signal.shape
            if L > target_len:
                signal = signal[:target_len, :]
            elif L < target_len:
                pad_len = target_len - L
                padding = np.zeros((pad_len, C))
                signal = np.concatenate([signal, padding], axis=0)

            # === Step 4: 转置为 [Channels, Length] -> [12, 5000] ===
            signal_tensor = torch.tensor(signal.T, dtype=torch.float32)

            # === Step 5: 归一化 (Instance Normalization / Z-Score) ===
            # 防止 Loss NaN 的关键步骤。
            # 即使 constants.py 中 mean=0, std=1，为了安全起见，推荐使用 Z-Score。
            mean = signal_tensor.mean(dim=1, keepdim=True)
            std = signal_tensor.std(dim=1, keepdim=True) + 1e-5  # 加 epsilon 防止除零
            signal_tensor = (signal_tensor - mean) / std

            # 再次检查 Tensor 是否有 NaN
            if torch.isnan(signal_tensor).any():
                print(f"[Error] NaN generated during normalization for {record_path}")
                return torch.zeros((12, 5000), dtype=torch.float32)

            return signal_tensor

        except Exception as e:
            print(f"Error loading ECG {record_path}: {e}")
            return None

    def deserialize_image(self, image_data) -> Image.Image:
        """转换图片数据为 RGB PIL Image"""
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
            # 返回全黑图片作为占位，避免报错崩溃
            print(f"Error deserializing image: {e}")
            return Image.new("RGB", (self.image_size, self.image_size))

    def process_image(self, image: Image.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
        """
        处理单张图片
        """
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            # Determine crop ratio based on image size
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num=2, max_num=9,
                    image_size=self.image_size, use_thumbnail=False
                )

            # Process global view with padding
            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # Process local views (crops)
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(
                        self.image_transform(crop_img).to(self.dtype)
                    )

            # Calculate image tokens
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                        num_queries * height_crop_num)

        else:  # crop_mode = False
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
        将单条对话数据处理为模型输入
        """
        messages = item_dict.get('messages', [])
        ecg_file = item_dict.get('ecg_file', "")

        # --- 1. 准备图片数据 ---
        images = []
        for message in messages:
            if "images" in message and message["images"]:
                for img_path in message["images"]:
                    if img_path is not None:
                        pil_image = self.deserialize_image(img_path)
                        images.append(pil_image)

        # 如果没有图片，创建一个纯黑图片占位，防止逻辑崩溃（DeepSeekOCR 通常需要至少一张图）
        if not images:
            # print(f"[Warning] No images in sample {ecg_file}, using placeholder.")
            images.append(Image.new("RGB", (self.image_size, self.image_size)))

        # --- 2. 准备 ECG 数据 ---
        ecg_tensor = torch.zeros((12, 5000), dtype=torch.float32)
        has_ecg = False

        # ECG Dropout 逻辑
        drop_ecg = (self.ecg_dropout_prob > 0) and (random.random() < self.ecg_dropout_prob)

        if ecg_file and not drop_ecg:
            loaded_ecg = self.load_ecg_data(ecg_file)
            if loaded_ecg is not None:
                ecg_tensor = loaded_ecg
                has_ecg = True

        # --- 3. 构建 Token 序列 ---
        tokenized_str = []
        images_seq_mask = []
        images_list, images_crop_list, images_spatial_crop = [], [], []

        ecg_seq_mask = []  # 新增 ECG Mask

        prompt_token_count = 0  # 默认为 0
        assistant_started = False
        image_idx = 0

        # 添加 BOS
        tokenized_str.append(self.bos_id)
        images_seq_mask.append(False)
        ecg_seq_mask.append(False)

        for message in messages:
            role = message["role"]
            content = message["content"]

            # 标记 Assistant 开始 (用于 Loss Masking)
            if role == "<|Assistant|>":
                if not assistant_started:
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True
                content = f"{content.strip()} {self.tokenizer.eos_token}"

            # 检测未知角色
            elif role != "<|User|>":
                # print(f"[Debug] Unknown role '{role}', treating as text.")
                pass

            # >>> 核心逻辑：ECG 占位符注入 <<<
            if has_ecg and role == "<|User|>" and ecg_seq_mask.count(True) == 0:
                # 总长度 = 1 (Start) + 101 (Features) + 1 (End) = 103
                total_ecg_tokens = 1 + self.ecg_token_len + 1
                ecg_tokens = [self.image_token_id] * total_ecg_tokens

                tokenized_str.extend(ecg_tokens)
                ecg_seq_mask.extend([True] * total_ecg_tokens)  # 标记为 ECG 区域
                images_seq_mask.extend([False] * total_ecg_tokens)

            # 处理图片标签 <image> 切分
            text_splits = content.split('<image>')

            for i, text_sep in enumerate(text_splits):
                # 编码文本
                tokenized_sep = text_encode(self.tokenizer, text_sep, bos=False, eos=False)
                tokenized_str.extend(tokenized_sep)
                images_seq_mask.extend([False] * len(tokenized_sep))
                ecg_seq_mask.extend([False] * len(tokenized_sep))

                # 如果后面有 <image> 标签
                if i < len(text_splits) - 1:
                    if image_idx < len(images):
                        image = images[image_idx]
                        # 处理图片
                        img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                        images_list.extend(img_list)
                        images_crop_list.extend(crop_list)
                        images_spatial_crop.extend(spatial_crop)

                        tokenized_str.extend(tok_img)
                        images_seq_mask.extend([True] * len(tok_img))  # 图像 Mask 为 True
                        ecg_seq_mask.extend([False] * len(tok_img))  # ECG Mask 为 False

                        image_idx += 1

        # 堆叠图片 Tensor
        if len(images_list) > 0:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((len(images_list), 3, self.base_size, self.base_size), dtype=self.dtype)
        else:
            # Dummy tensors
            images_ori = torch.zeros((1, 3, 640, 640), dtype=self.dtype)
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)
            images_spatial_crop_tensor = torch.zeros((1, 2), dtype=torch.long)

        # 修复 Loss=0 问题：如果没找到 Assistant，打印警告，但不要 Mask 全部（可以尝试学习 User 的输入，或者只 Mask 到最后）
        if not assistant_started:
            print(f"[Warning] No assistant message found in sample {ecg_file}. Masking logic may be incorrect.")
            # 策略：如果不训练 user 部分，且没有 assistant，则整条数据无法训练，返回 -100
            # 这里保持原逻辑，但因为 role_map 修复了，应该很少进入
            prompt_token_count = len(tokenized_str)

        # === 修复截断逻辑 ===
        if len(tokenized_str) > self.max_length:
            tokenized_str = tokenized_str[:self.max_length]
            ecg_seq_mask = ecg_seq_mask[:self.max_length]
            images_seq_mask = images_seq_mask[:self.max_length]
            # 如果截断导致 prompt_token_count 越界，修正它
            if prompt_token_count > self.max_length:
                prompt_token_count = self.max_length

        return {
            "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
            "prompt_token_count": prompt_token_count,
            "ecg_values": ecg_tensor,
            "ecg_seq_mask": torch.tensor(ecg_seq_mask, dtype=torch.bool)
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Batch 组装"""
        batch_data = []

        for feature in features:
            try:
                processed = self.process_single_sample(feature)
                batch_data.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        # 1. Padding Sequences
        input_ids = pad_sequence([x['input_ids'] for x in batch_data], batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        images_seq_mask = pad_sequence([x['images_seq_mask'] for x in batch_data], batch_first=True,
                                       padding_value=False)
        ecg_seq_mask = pad_sequence([x['ecg_seq_mask'] for x in batch_data], batch_first=True, padding_value=False)

        # 2. Build Labels (Loss Masking)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Pad 不算
        labels[images_seq_mask] = -100  # Image 不算
        labels[ecg_seq_mask] = -100  # ECG 不算

        if self.train_on_responses_only:
            prompt_counts = [x['prompt_token_count'] for x in batch_data]
            for i, count in enumerate(prompt_counts):
                if count > 0:
                    labels[i, :count] = -100

        # 3. Stack ECG
        ecg_values = torch.stack([x['ecg_values'] for x in batch_data])

        # 最后的防线：检查 Batch 级 NaN
        if torch.isnan(ecg_values).any():
            print("[FATAL] NaN in ecg_values batch, zerofilling.")
            ecg_values = torch.nan_to_num(ecg_values)

        # 4. Process Images list of tuples
        images_batch = []
        for item in batch_data:
            images_batch.append((item['images_crop'], item['images_ori']))

        images_spatial_crop = torch.cat([item['images_spatial_crop'] for item in batch_data], dim=0)

        # 调试信息打印 (可选，频繁打印可注释)
        # masked_tokens = (labels == -100).sum().item()
        # total_tokens = labels.numel()
        # print(f"[DEBUG] Batch Info: Valid Tokens: {total_tokens - masked_tokens} / {total_tokens}")

        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).long(),
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
            "ecg_values": ecg_values,
            "ecg_seq_mask": ecg_seq_mask,
        }

if __name__ == '__main__':
    MODEL_PATH = "deepseek-ai/DeepSeek-OCR"  # 或者是本地路径
    data_path = "/Users/zhangyf/PycharmProjects/cfel/deepseekocr/data/deepseek_ocr_ecg_future.jsonl"
    dataset = ECGInstructionDataset(data_path)
    print(dataset[0])
    print(len(dataset))
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

    clean_ecg_encoder = ECGEncoderWrapper(
        pretrained="/Users/zhangyf/Documents/cfel/epoch_20.pt"
    )

    # 确保它是 FP32 (默认就是，但保险起见)
    clean_ecg_encoder.to(torch.float32)
    # 确保冻结
    for p in clean_ecg_encoder.parameters():
        p.requires_grad = False

    # 3. 替换模型中的模块
    model.model.ecg_encoder = clean_ecg_encoder
    # samples = []
    # with open(data_path, 'r', encoding='utf-8') as f:
    #     for line_num, line in enumerate(f, 1):
    #         data = json.loads(line.strip())
    #         samples.append(data)

    data_collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=640,
        base_size=1024,
        crop_mode=True,
        train_on_responses_only=True,
    )

    result = data_collator(dataset)
    print(result['input_ids'].shape)
    print(result['labels'].shape)
    print(result['images'])
    print(result['images_seq_mask'].shape)
    print(result['images_spatial_crop'].shape)
    output = model.forward(**result)
    print(output.loss.item())
    print(output)
    # model.save_pretrained("deepseek-ai/DeepSeek-OCR-ecg")
