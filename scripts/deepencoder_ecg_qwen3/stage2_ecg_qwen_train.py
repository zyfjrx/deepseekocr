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
from models.ecgencoder import _build_ecg_tower
from models.deepencoder import build_sam_vit_b, build_clip_l
from ecg_qwen_dataset import ECGInstructionDataset, QwenOCRDataCollator
from models.modeling_deepencode_ecg_qwen3 import DeepencoderEcgOCRForCausalLM


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        # default="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
        default="/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
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
    model = DeepencoderEcgOCRForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,  # H800 建议使用 bfloat16
        trust_remote_code=True
    )

    # 加载 deepencode权重
    model.model.sam_model = build_sam_vit_b().to(dtype=model.dtype)
    model.model.vision_model = build_clip_l().to(dtype=model.dtype)
    root_path = "/root/.cache/huggingface/hub/deepencoder/"
    model.model.sam_model.load_state_dict(torch.load(root_path + "sam_encoder.pth"))
    model.model.vision_model.load_state_dict(torch.load(root_path + "clip_encoder.pth"))
    # 初始化ecgencoder
    model.model.ecg_encoder = _build_ecg_tower().to(dtype=model.dtype)
    missing_keys, unexpected_keys = model.model.ecg_encoder.load_state_dict(
        torch.load("/root/.cache/huggingface/hub/ecgencoder/ecg_encoder.pth"))
    print(missing_keys, unexpected_keys)
    # 初始化 特殊嵌入token
    print("Initializing Special Embedding Tokens...")
    print(f"model.config.hidden_size:{model.config.hidden_size}")
    embed_std = 1 / torch.sqrt(torch.tensor(model.config.hidden_size, dtype=torch.bfloat16))
    model.model.image_newline = torch.nn.Parameter(torch.randn(model.config.hidden_size, dtype=model.dtype) * embed_std)
    model.model.view_seperator = torch.nn.Parameter(torch.randn(model.config.hidden_size, dtype=model.dtype) * embed_std)
    model.model.ecg_start_embed = torch.nn.Parameter(torch.randn(model.config.hidden_size, dtype=model.dtype) * embed_std)
    model.model.ecg_end_embed = torch.nn.Parameter(torch.randn(model.config.hidden_size, dtype=model.dtype) * embed_std)

    # 开启梯度检查点
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 4. 冻结 Vision 模块逻辑
     # 开启梯度的模块
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

    # 遍历参数，只开启命中关键字的部分
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in modules_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False
    # if model_args.freeze_vision:
    #     print("Freezing Vision Modules (SAM + CLIP)...")
    #     # 遍历所有参数
    #     for name, param in model.named_parameters():
    #         # QwenOCRModel 定义了 self.sam_model 和 self.vision_model
    #         if "sam_model" in name or "vision_model" in name or "ecg_encoder" in name:
    #             param.requires_grad = False
    #         else:
    #             # LLM (model.layers), Projector (model.projector), Embeddings 保持训练
    #             param.requires_grad = True
    #
    #     # 再次确认 projector 是开启梯度的
    #     # model.model.projector 是对齐层
    #     for name, param in model.model.projector.named_parameters():
    #         param.requires_grad = True
    #     for name, param in model.model.ecg_projector.named_parameters():
    #         param.requires_grad = True

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
    path := torch.load(os.path.join(training_args.output_dir, "trainer_state.json"),weights_only=False) if os.path.exists(
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