import json
import os
import random
import copy
from torch.utils.data import Dataset


class ECGInstructionDataset(Dataset):
    def __init__(
            self,
            jsonl_file,
            is_train=False,
            ecg_root="",
            image_root="",
    ):
        """
        轻量级 Dataset：只负责 Prompt 注入和格式转换，不负责加载 Tensor。
        """
        self.data = []
        self.is_train = is_train
        self.ecg_root = ecg_root
        self.image_root = image_root
        print(f"Loading data from {jsonl_file}...")
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
            print(f"Loaded {len(self.data)} samples.")
        except FileNotFoundError:
            print(f"Error: File {jsonl_file} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ---------------------------------------------------
        # 1. 获取 & 处理 Measurements (离线字段)
        # ---------------------------------------------------
        meas_str = item.get('measurements', '')
        meas_source = item.get('meas_source', 'estimated')

        # 策略：训练时 20% 概率丢弃测量值 (Modality Dropout)
        # 即使不加载 Tensor，我们依然可以在 Prompt 层面做 Dropout
        inject_measurements = True
        if self.is_train and random.random() < 0.2:
            inject_measurements = False

        # ---------------------------------------------------
        # 2. 格式转换: conversations -> messages & Prompt 注入
        # ---------------------------------------------------
        # DeepCopy 防止修改原始缓存
        raw_convs = copy.deepcopy(item.get('conversations', []))
        messages = []

        # 准备注入的文本
        prompt_text = ""
        if inject_measurements and meas_str and meas_str != "Measurements: N/A":
            prefix = "[Verified]" if meas_source == 'verified' else "[Estimated]"
            prompt_text = f"\n{prefix} Measurements: {meas_str}\n"

        for i, turn in enumerate(raw_convs):
            # 映射 role: human -> user, gpt -> assistant
            role = 'user' if turn['from'] == 'human' else 'assistant'
            content = turn['value']

            # 仅在第一条 User 消息中注入 Measurements
            if i == 0 and role == 'user' and prompt_text:
                if "<image>" in content:
                    content = content.replace("<image>", "<image>" + prompt_text)
                else:
                    content = prompt_text + content

            messages.append({
                "role": role,
                "content": content
            })

        # ---------------------------------------------------
        # 3. 路径处理 (拼接根目录)
        # ---------------------------------------------------
        # 处理 ECG 路径
        raw_ecg_path = item.get('ecg', '')
        ecg_file_path = os.path.join(self.ecg_root, raw_ecg_path) if raw_ecg_path else ""

        # 处理 Image 路径 (转为 List 格式，适配 DeepSeekOCRDataCollator)
        raw_img_path = item.get('image', '')
        image_list = []
        if raw_img_path:
            full_img_path = os.path.join(self.image_root, raw_img_path)
            image_list.append(full_img_path)

        # ---------------------------------------------------
        # 4. 返回 Collator 所需的 Raw Dict
        # ---------------------------------------------------
        return {
            "messages": messages,  # 已注入 Measurements 的 messages 列表
            "images": image_list,  # 图片路径列表 ["/path/to/img"]
            "ecg_file": ecg_file_path  # ECG 文件路径 "/path/to/ecg"
        }

if __name__ == '__main__':
    dataset = ECGInstructionDataset(
        jsonl_file="/Users/zhangyf/PycharmProjects/cfel/deepseekocr/utils/grounding_train_30k.jsonl",
        is_train=True
    )
    print(dataset[0])