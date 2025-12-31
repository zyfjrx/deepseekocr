import torch
from transformers import AutoTokenizer



# from models.modeling_deepseekocr import DeepseekOCRForCausalLM
from models.modeling_qwen_ocr import QwenOCRForCausalLM
QWEN_MODEL_PATH = "/Users/zhangyf/llm/Qwen2.5-0.5B-Instruct" # 或者是本地路径


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
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QwenOCRForCausalLM.from_pretrained(
        QWEN_MODEL_PATH,
        # trust_remote_code=True, # 使用本地类时通常不需要，除非 config 也是远程且未本地导入
        torch_dtype=torch.bfloat16,
        # _attn_implementation="flash_attention_2"
    )
    model = model.eval().to("cpu")
    print(model)
    print_trainable_parameters(model)

    # for name,model in model.named_modules():
    #     print(name)
    # print(model)
    # 冻结 Vision Encoder (通常不需要微调 SAM 和 CLIP)
    # for name, param in model.named_parameters():
    #     if "sam_model" in name or "vision_model" in name:
    #         param.requires_grad = False
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    # # prompt = "<image>\nFree OCR. "
    prompt = "<|image_pad|>\nnFree OCR."
    image_file = '/Users/zhangyf/大模型学习/深度学习/智图寻宝/资料/logo/logo.jpg'
    output_path = './outputs'

    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path=output_path, base_size=1024,
                      image_size=640, crop_mode=True, save_results=True, test_compress=True)
    print("**1000")
    print(res)
    print("**1000")


if __name__ == "__main__":
    train()