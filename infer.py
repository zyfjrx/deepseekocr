import torch
from transformers import AutoTokenizer



from modeling_deepseekocr import DeepseekOCRForCausalLM

MODEL_PATH = "deepseek-ai/DeepSeek-OCR" # 或者是本地路径
DATA_ROOT = "./data" # 图片存放根目录
MAX_LENGTH = 2048
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815 # DeepSeek-OCR 的 image token id
PAD_TOKEN_ID = 100015 # 或者是 tokenizer.pad_token_id
IGNORE_INDEX = -100


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
    model = model.eval().to("cpu")
    # for name,model in model.named_modules():
    #     print(name)
    # print(model)

    prompt = "<image>\nFree OCR. "
    prompt = "<image>\nFree OCR. "
    image_file = '/Users/zhangyf/PycharmProjects/cfel/deepseek-ocr/data/images/00009_hr-0.png'
    output_path = './outputs'

    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path=output_path, base_size=1024,
                      image_size=640, crop_mode=True, save_results=True, test_compress=True)


if __name__ == "__main__":
    train()