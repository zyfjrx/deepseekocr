import torch
from transformers import AutoTokenizer,Trainer,TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from dataset import ECGInstructionDataset,DeepSeekOCRDataCollator
from models.modeling_deepseekocr import DeepseekOCRForCausalLM



MODEL_PATH = "deepseek-ai/DeepSeek-OCR" # 或者是本地路径
data_path = "/Users/zhangyf/PycharmProjects/cfel/deepseekocr/data/deepseek_ocr_ecg_future.jsonl"
output_dir = "./checkpoints_ocr"

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    print("\n=== 当前开启梯度的层 (Trainable Layers) ===")
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"✅ {name} ({param.numel()} params)")

    print(f"\n=== 统计结果 ===")
    print(f"总参数量: {all_param}")
    print(f"可训练参数: {trainable_params}")
    print(f"可训练比例: {100 * trainable_params / all_param:.4f}%")

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
        torch_dtype=torch.bfloat16,
    )
    # print(model)
    # print_trainable_parameters(model)



    # 冻结 Vision Encoder (通常不需要微调 SAM 和 CLIP)
    for name, param in model.named_parameters():
        if "sam_model" in name or "vision_model" in name:
            param.requires_grad = False
    print_trainable_parameters(model)
    
    # 启用 LoRA (推荐)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        # target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj","projector.layers"]
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["projector.layers"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=6,
        per_device_train_batch_size=16,  # 根据显存调整，OOM就减小
        gradient_accumulation_steps=16,  # 显存小就增大这个，保持总BatchSize不变
        learning_rate=2e-4,  # LoRA常用
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,  # 必开，省显存
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=5,  # 只留最近3个模型
        report_to="tensorboard",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,  # DDP多卡防报错
        dataloader_num_workers=8
    )
    dataset = ECGInstructionDataset(data_path)
    data_collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir=output_dir)
    trainer.save_state()

if __name__ == "__main__":
    train()
