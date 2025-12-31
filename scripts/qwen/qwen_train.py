import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed
)
import logging

# 导入你本地的模型和数据集代码
# 假设你之前修改好的 dataset 代码保存为了 dataset.py
from qwen_dataset import ECGInstructionDataset, QwenOCRDataCollator
from models.modeling_qwen_ocr import QwenOCRForCausalLM


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    freeze_vision: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the vision encoder (SAM + CLIP)."}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="/workspace/deepseekocr/deepseek_ocr_instruct.jsonl",
        metadata={"help": "Path to the jsonl dataset file."}
    )
    image_size: int = field(
        default=640,
        metadata={"help": "Image size for processing."}
    )
    base_size: int = field(
        default=1024,
        metadata={"help": "Base size for global view."}
    )


def print_trainable_parameters(model):
    """
    打印模型的可训练参数数量
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # 设置随机种子
    set_seed(training_args.seed)

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"  # 训练时通常 Padding 在右侧（除了某些生成任务）
    )
    # Qwen 的 pad_token 处理
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # 也可以指定 e.g. tokenizer.pad_token_id = 151643

    # 2. 加载数据集

    full_dataset = ECGInstructionDataset(jsonl_file=data_args.data_path)

    # 手动按比例切分 (例如 99% 训练，1% 验证)
    train_size = int(0.99 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # 3. 加载模型
    print(f"Loading model from {model_args.model_name_or_path}...")
    model = QwenOCRForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,  # H800 建议使用 bfloat16
        trust_remote_code=True
    )

    # 开启梯度检查点 (节省显存，H800 显存够大可选择关闭以换取速度)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 4. 冻结 Vision 模块逻辑
    if model_args.freeze_vision:
        print("Freezing Vision Modules (SAM + CLIP)...")
        # 遍历所有参数
        for name, param in model.named_parameters():
            # 这里的命名依据是 modeling_qwen_ocr.py 中的结构
            # QwenOCRModel 定义了 self.sam_model 和 self.vision_model
            if "sam_model" in name or "vision_model" in name:
                param.requires_grad = False
            else:
                # LLM (model.layers), Projector (model.projector), Embeddings 保持训练
                param.requires_grad = True

        # 再次确认 projector 是开启梯度的
        # model.model.projector 是对齐层
        for name, param in model.model.projector.named_parameters():
            param.requires_grad = True

    print_trainable_parameters(model)

    # 5. Data Collator
    data_collator = QwenOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=data_args.image_size,
        base_size=data_args.base_size,
        crop_mode=True,
        train_on_responses_only=True
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7. 开始训练
    if list(path.iterdir()) if (
    path := torch.load(os.path.join(training_args.output_dir, "trainer_state.json")) if os.path.exists(
            os.path.join(training_args.output_dir, "trainer_state.json")) else None) else None:
        # 简单的断点续训逻辑，实际使用可依赖 HF 的 resume_from_checkpoint 参数
        pass

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # 8. 保存模型
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()