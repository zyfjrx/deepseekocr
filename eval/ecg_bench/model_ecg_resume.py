import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
import transformers
from typing import Dict, Optional, Sequence, List
import re
from PIL import Image
import wfdb
import math
from transformers import AutoTokenizer
from models.modeling_deepseekocr_ecg import DeepseekOCRForCausalLM

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    #im_start, im_end = tokenizer.additional_special_tokens_ids
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


def eval_model(args):
    # Model
    disable_torch_init()



    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    # 注意: 显存不足时可以使用 load_in_4bit=True (需要 bitsandbytes)
    # 使用本地导入的 DeepseekOCRForCausalLM 类直接加载
    model = DeepseekOCRForCausalLM.from_pretrained(
        args.model_path,
        # trust_remote_code=True, # 使用本地类时通常不需要，除非 config 也是远程且未本地导入
        torch_dtype=torch.bfloat16,
        # _attn_implementation="flash_attention_2"
    )

    
    questions = []
    with open(args.question_file, "r") as f:
        # json_data = json.load(f)
        _, ext = os.path.splitext(args.question_file)
        if ext.lower() == '.json':
            json_data = json.load(f)
        elif ext.lower() == '.jsonl':
            json_data = []
            for line in f:
                if line.strip():
                    json_data.append(json.loads(line))
        else:
            raise ValueError("Unsupported file format: {}".format(ext))
        
        for line in json_data:
            questions.append({"question_id": line["id"], 
                              "image": line["image"], 
                              "text": line["conversations"][0]["value"].replace("<image>\n",""),
                              "ans": line["conversations"][1]["value"],
                              "ecg": line["ecg"],
                              "conversations": line["conversations"],
                              },
                             )

    existing_question_ids = set()
    
    if os.path.exists(args.answers_file):
        with open(args.answers_file, "r") as ans_file:
            for line in ans_file:
                existing_data = json.loads(line)
                existing_question_ids.add(existing_data["question_id"])  # Track existing question_ids

    output_file = open(args.answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        # Skip if the answer for this question_id already exists
        if idx in existing_question_ids:
            print(f"Skipping question {idx}, already exists.")
            continue

        image_file = line["image"]
        ecg_file = line["ecg"]
        qs = "<image>" + '\n' + line["text"]

        ecg_file = os.path.join(args.ecg_folder, ecg_file)
        image_file = os.path.join(args.image_folder, image_file)

            
        try:
            response = model.infer(
                tokenizer=tokenizer,
                prompt=qs,
                image_file=image_file,
                ecg_file=ecg_file,
                output_path='./tmp_infer',  # 占位，实际上 save_results=False 不会用
                base_size=1024,
                image_size=640,
                crop_mode=True,  # 必须开启 Gundam Mode
                save_results=False,  # 只返回文本
                test_compress=False,
                eval_mode=True
            )
        except Exception as e:
            print(f"Error inferring {idx}: {e}")
            response = "Error"

        # --- 5. 构造输出 (严格保持 model_ecg_resume.py 格式) ---
        ans_id = shortuuid.uuid()
        new_answer = {
            "question_id": idx,  # 评估脚本 evaluate_ecgbench.py 需要这个作为 Key
            "prompt": qs,
            "text": response,  # 评估脚本 evaluate_ecgbench.py 读取这个字段作为预测值
            "answer_id": ans_id,
            "model_id": "deepseek_ocr",
            "metadata": {}
        }
        output_file.write(json.dumps(new_answer) + "\n")
        output_file.flush()

    output_file.close()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/Users/zhangyf/PycharmProjects/cfel/deepseekocr/checkpoint/deepseek-ai/DeepSeek-OCR-ecg")
    parser.add_argument("--image-folder", type=str, default="/gem")
    parser.add_argument("--ecg-folder", type=str, default="/ts_ds")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)


    args = parser.parse_args()

    eval_model(args)

