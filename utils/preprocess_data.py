import json
import os
import argparse
from tqdm import tqdm
from data_processor import ECGDataProcessor  # 确保 data_processor.py 在同级目录


def preprocess_dataset(input_file, output_file, ecg_root):
    """
    离线处理核心逻辑：
    1. 读取原始 JSON 数组文件
    2. 计算/聚合指标
    3. 转换为 JSONL 格式输出 (更适合 DataLoader 流式读取)
    """
    # 初始化处理器
    processor = ECGDataProcessor()

    print(f"正在读取原始数据: {input_file} ...")

    try:
        with open(input_file, 'r', encoding='utf-8') as fin:
            # 【关键修改】针对 JSON Array 格式，使用 json.load 读取整个列表
            data_list = json.load(fin)

        if not isinstance(data_list, list):
            raise ValueError("输入文件不是 JSON 数组格式 (应以 '[' 开头)")

        print(f"成功加载 {len(data_list)} 条样本，开始预处理...")

    except json.JSONDecodeError:
        print("错误：文件解析失败。请确认文件是标准的 JSON 数组格式。")
        return
    except Exception as e:
        print(f"读取错误: {e}")
        return

    # 准备输出
    print(f"ECG 根目录: {ecg_root}")
    print(f"输出目标: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as fout:

        # 统计计数器
        stats = {'silver': 0, 'gold': 0, 'failed': 0, 'skipped': 0}

        for item in tqdm(data_list):
            try:
                # --- 核心判断逻辑 ---
                meas_str = "Measurements: N/A"
                source_type = "estimated"  # 默认为估算值

                # 1. 检查是否为 Gold 数据 (有 machine_measurements 且不为空)
                if "machine_measurements" in item and item["machine_measurements"]:
                    # Gold: 直接聚合 JSON
                    meas_str = processor.process(item["machine_measurements"], mode='gold')
                    source_type = "verified"  # 标记为金标准
                    stats['gold'] += 1

                # 2. 否则视为 Silver 数据 (需要计算 ECG 文件)
                elif "ecg" in item:
                    # Silver: 读取 .dat 文件计算
                    # 拼接完整路径
                    ecg_rel_path = item["ecg"]
                    ecg_path = os.path.join(ecg_root, ecg_rel_path)

                    # 检查文件是否存在 (兼容 .dat 后缀或无后缀)
                    # 有些数据集路径里已经带了后缀，有些没带，做个兼容检查
                    candidates = [ecg_path, ecg_path + ".dat"]
                    found_path = None
                    for p in candidates:
                        if os.path.exists(p):
                            found_path = p
                            break

                    if found_path:
                        # 传入不带 .dat 后缀的路径给 wfdb
                        wfdb_path = os.path.splitext(found_path)[0]
                        meas_str = processor.process(wfdb_path, mode='silver')
                        stats['silver'] += 1
                    else:
                        # 文件找不到，跳过计算，但在训练中可以用全0填充
                        meas_str = "Measurements: N/A"
                        stats['failed'] += 1
                        # print(f"Warning: ECG file not found: {ecg_path}")

                else:
                    stats['skipped'] += 1
                    continue

                # 3. 写入新字段
                item['measurements'] = meas_str
                item['meas_source'] = source_type

                # 【关键】将处理后的对象转为 JSONL 的一行写入
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')

            except Exception as e:
                # print(f"处理单条样本出错 ID {item.get('id', 'unknown')}: {e}")
                stats['failed'] += 1
                # 出错也尽量写入，避免数据丢失，measurements 保持 N/A 即可
                if 'measurements' not in item:
                    item['measurements'] = "Measurements: N/A"
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"预处理完成！\n统计信息: {stats}")
    print(f"结果已保存为 JSONL 格式 (每行一个 JSON)，可直接用于 ecg_dataset.py")


if __name__ == "__main__":
    # 使用示例
    # python preprocess_data.py --input raw_train_array.json --output offline_train.jsonl --ecg_root /path/to/data
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="原始 JSON 数组文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--ecg_root", type=str, default="", help="ECG 文件根目录")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 {args.input}")
    else:
        preprocess_dataset(args.input, args.output, args.ecg_root)