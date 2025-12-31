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
from ecg_dataset import ECGInstructionDataset, DeepSeekOCRDataCollator
from models.ecg_encoder import ECGEncoderWrapper
from models.modeling_deepseekocr_ecg import DeepseekOCRForCausalLM


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="deepseek-ai/DeepSeek-OCR",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    freeze_vision: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the vision encoder (SAM + CLIP)."}
    )
    ecg_weight_path: str = field(
        default="/Users/zhangyf/Documents/cfel/epoch_20.pt",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    max_length: int = field(
        default=4096,
        metadata={"help": "Base size for global view."}
    )

#
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


# def print_trainable_parameters(model):
#     trainable_params = 0
#     all_param = 0
#     print("\n=== 当前开启梯度的层 (Trainable Layers) ===")
#     for name, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#             print(f"✅ {name} ({param.numel()} params)")
#
#     print(f"\n=== 统计结果 ===")
#     print(f"总参数量: {all_param}")
#     print(f"可训练参数: {trainable_params}")
#     print(f"可训练比例: {100 * trainable_params / all_param:.4f}%")

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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载数据集
    full_dataset = ECGInstructionDataset(jsonl_file=data_args.data_path)

    # 手动按比例切分 (例如 99% 训练，1% 验证)
    train_size = int(0.99 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # 3. 加载模型
    print(f"Loading model from {model_args.model_name_or_path}...")
    model = DeepseekOCRForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,  # H800 建议使用 bfloat16
        trust_remote_code=True
    )

    fp32_ecg_encoder = ECGEncoderWrapper(
        pretrained=model_args.ecg_weight_path,
    )

    # 确保它是 FP32 (默认就是，但保险起见)
    fp32_ecg_encoder.to(torch.float32)
    # # 确保冻结
    # for p in fp32_ecg_encoder.parameters():
    #     p.requires_grad = False
    # 3. 替换模型中的模块
    model.model.ecg_encoder = fp32_ecg_encoder

    # 开启梯度检查点 (节省显存，H800 显存够大可选择关闭以换取速度)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 4. 冻结 Vision 模块逻辑
    if model_args.freeze_vision:
        print("Freezing Vision Modules (SAM + CLIP)...")
        for name, param in model.named_parameters():
            # self.sam_model 和 self.vision_model ,ecg_encoder
            if "sam_model" in name or "vision_model" in name or "ecg_encoder" in name:
                param.requires_grad = False
            else:
                # LLM (model.layers), Projector (model.projector), Embeddings 保持训练
                param.requires_grad = True


        # projector,ecg_projector 是对齐层
        for name, param in model.model.projector.named_parameters():
            print( name)
            param.requires_grad = True
        for name, param in model.model.ecg_projector.named_parameters():
            print( name)
            param.requires_grad = True

    print_trainable_parameters(model)

    # 5. Data Collator
    data_collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=data_args.image_size,
        base_size=data_args.base_size,
        crop_mode=True,
        train_on_responses_only=True,
        max_length=data_args.max_length,
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
    # 7. start training
    if list(path.iterdir()) if (
    path := torch.load(os.path.join(training_args.output_dir, "trainer_state.json")) if os.path.exists(
            os.path.join(training_args.output_dir, "trainer_state.json")) else None) else None:
        # 简单的断点续训逻辑，实际使用可依赖 HF 的 resume_from_checkpoint 参数
        pass

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # 8. save model and tokenizer
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()