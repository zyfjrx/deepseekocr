import json
import os
import json
import requests
from openai import  OpenAI
# import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
from tqdm import tqdm
import numpy as np
from prompts import report_eval_prompt

# os.environ["AZURE_OPENAI_API_KEY"] = "api key"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "endpoint"
# os.environ["OPENAI_API_KEY"] = ""
# openai.api_key = "sk-72a06376ac8b46209ecb43bda39cf"
# openai.base_url = "http://pg154v8s.aistudio.cq-pub.puhui.chengfengerlai.com/openai/default-workspace/ikll7p8y35da/v1"
# api_key = "sk-72a06376ac8b46209ecb43bda39cf"
# base_url = "http://pg154v8s.aistudio.cq-pub.puhui.chengfengerlai.com/openai/default-workspace/ikll7p8y35da/v1"

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def extract_json_from_text(text):
    # Find the start and end of the JSON object
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        return None

    # Extract the JSON string
    json_str = text[start:end]

    try:
        # Parse the JSON string
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None


def process(datum, ptb_golden_report, output_dir, eval_model_name, client):
    if "question_id" in datum:
        ecg_id = datum['question_id'].split('-')[-1]
        generated_report = datum['text']
    elif "id" in datum:
        ecg_id = datum['id'].split('-')[-1]
        generated_report = datum['response']

    golden_report = ptb_golden_report[ecg_id]

    report_score_prompt = report_eval_prompt

    prompt = f"{report_score_prompt} \n [The Start of Ground Truth Report]\n {golden_report}\n [The End of Ground Truth Report]\n [The Start of Generated Report]\n {generated_report}\n [The End of Generated Report]"
    api_key = "sk-72a06376ac8b46209ecb43bda39cf"
    base_url = "http://pg154v8s.aistudio.cq-pub.puhui.chengfengerlai.com/openai/default-workspace/ikll7p8y35da/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=eval_model_name,
        messages=[
            {"role": "user", "content": prompt+ "/no_think"},
        ],
        temperature=0,
        # response_format={"type": "json_object"},
    )


    # response = openai.ChatCompletion.create(
    #     model=eval_model_name,
    #     messages=[
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature=0,
    # )

    # Save the JSON response directly to a .json file
    with open(f'{output_dir}/{ecg_id}.json', 'w') as f:
        text = response.choices[0].message.content
        text = text.replace("<think>","")
        text = text.replace("</think>","")
        f.write(text)

def run_pairwise_comparison(ptb_test_generated_report_file, ptb_golden_report,  output_dir, eval_model_name, client):

    ptb_test_generated_report = load_jsonl(ptb_test_generated_report_file)
    # print(ptb_test_generated_report[0])

    existing_files = os.listdir(output_dir)
    existing_images = [file.split('.')[0] for file in existing_files]
    if "question_id" in ptb_test_generated_report[0]:
        filtered_ptb_test_generated_report = [datum for datum in ptb_test_generated_report if datum['question_id'].split('-')[-1] not in existing_images]
    elif "id" in ptb_test_generated_report[0]:
        filtered_ptb_test_generated_report = [datum for datum in ptb_test_generated_report if datum['id'].split('-')[-1] not in existing_images]
    # filtered_ptb_test_generated_report = [datum for datum in ptb_test_generated_report if datum['question_id'].split('-')[-1] not in existing_images]
    print(len(filtered_ptb_test_generated_report))
    print(f"eval_model_name: {eval_model_name}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process, datum, ptb_golden_report, output_dir, eval_model_name, client) for datum in filtered_ptb_test_generated_report]
        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Wait for the result to handle any exceptions that might occur


def compute_score(output_dir):
    report_scores = {}
    all_scores = {}
    for file in os.listdir(output_dir):
        with open(f"{output_dir}/{file}", 'r',encoding='utf-8') as f:
            # print(file)
            # Due to the output may start with ```json
            try:
                report_score = json.load(f)
            except:
                f.seek(0)
                content = f.read()
                content = content.strip()
                if content.startswith("```"):
                    first_newline = content.find("\n")
                    if first_newline != -1:
                        content = content[first_newline:].strip()
                    else:
                        content = ""
                if content.endswith("```"):
                    last_backtick_idx = content.rfind("\n```")
                    if last_backtick_idx != -1:
                        content = content[:last_backtick_idx].strip()
                    else:
                        content = content[:-3].strip()
                    report_score = json.loads(content)
            # sum the scores
            for key, value in report_score.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(value['Score'])
            report_scores[file.split('.')[0]] = sum([value['Score'] for key, value in report_score.items()])/len(report_score) * 10

    for key, value in all_scores.items():
        print(f"{key}: {np.mean(value)*10}")
    # print the average scores
    print(f'Lenght of report_scores: {len(report_scores)}')
    print(f"Average Score: {np.mean(list(report_scores.values()))}")


# main function
def main():
    # golden report file
    # ptb_golden_report_file = "/root/.cache/huggingface/hub/datasets--PULSE-ECG--ECGBench/snapshots/cc7bfe06da6b7ca5b4e890d95e1a4099e4b248e2/ptb-test-report"
    ptb_golden_report_file = "/root/.cache/huggingface/hub/datasets--PULSE-ECG--ECGBench/snapshots/cc7bfe06da6b7ca5b4e890d95e1a4099e4b248e2/ptb-test-report/ptb_test_report.jsonl"
    with open(ptb_golden_report_file, 'r') as f:
        ptb_golden_reports = []
        for line in f:
            if line.strip():
                ptb_golden_reports.append(json.loads(line))
        # ptb_golden_report_file = json.load(f)
    ptb_golden_report = {datum["id"].split("-")[-1]: datum['conversations'][-1]['value'] for datum in ptb_golden_reports}

    # model generated report file
    test_model_name = 'step-final'
    model_name = "deepseekocr"
    # ptb_test_generated_report_file = f'../eval_outputs/{model_name}/ptb-test-report/{test_model_name}.jsonl'
    ptb_test_generated_report_file = "/workspace/GEM/eval_bench/eval_outputs/deepseek-ocr/ecg-bench-test/ptb-test-report"

    # report score save directory
    output_dir = f'/workspace/GEM/eval_bench/eval_outputs/deepseek-ocr/ecg-bench-test/ptb-test-report/report-{model_name}-{test_model_name}'
    # output_dir = "/Users/zhangyf/PycharmProjects/cfel/GEM/data/ecg-bench/ptb-report/out"
    os.makedirs(output_dir, exist_ok=True)

    # client = AzureOpenAI(
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_version="2024-08-01-preview",
    #     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    # )

    client = None
    eval_model_name="aj5cxr2h2l"

    print(f"Pairwise Comparison: ecg-chat-{model_name}-{test_model_name}")
    run_pairwise_comparison(ptb_test_generated_report_file, ptb_golden_report, output_dir, eval_model_name, client)

    print(f"ECG Report Score: {output_dir}")
    compute_score(output_dir)

if __name__ == '__main__':
    main()
