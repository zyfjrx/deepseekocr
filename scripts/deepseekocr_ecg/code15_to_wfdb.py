import os
import glob
import h5py
import numpy as np
import pandas as pd
import wfdb
import argparse
from tqdm import tqdm


def get_keys(h5_file):
    """
    自动检测 HDF5 键名
    """
    keys = list(h5_file.keys())

    # 检测信号键名
    if 'tracings' in keys:
        sig_key = 'tracings'
    elif 'signal' in keys:
        sig_key = 'signal'
    else:
        raise KeyError(f"Cannot find signal dataset. Available keys: {keys}")

    # 检测ID键名
    if 'exam_id' in keys:
        id_key = 'exam_id'
    elif 'id_exam' in keys:
        id_key = 'id_exam'
    else:
        raise KeyError(f"Cannot find exam_id dataset. Available keys: {keys}")

    return sig_key, id_key


def get_diagnosis_string(row):
    """
    格式化诊断标签
    """
    diagnoses = []
    dx_cols = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

    for dx in dx_cols:
        if dx in row and row[dx]:
            diagnoses.append(dx)

    if row.get('normal_ecg'):
        diagnoses.append('Normal')

    return ",".join(diagnoses) if diagnoses else "Unknown"


def prepare_code15(args):
    data_dir = args.data_dir
    base_output_dir = os.path.join(data_dir, "records")

    # 1. 读取 Metadata CSV
    csv_path = os.path.join(data_dir, "exams.csv")
    if not os.path.exists(csv_path):
        print(f"Error: exams.csv not found in {data_dir}")
        return

    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    # 将 exam_id 转为字符串并设为索引
    df['exam_id'] = df['exam_id'].astype(str)
    df.set_index('exam_id', inplace=True)

    # 2. 准备基础输出目录
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Created base output directory: {base_output_dir}")

    # 3. 寻找所有 HDF5 分片文件
    h5_files = sorted(glob.glob(os.path.join(data_dir, "exams_part*.hdf5")))
    if not h5_files:
        h5_files = sorted(glob.glob(os.path.join(data_dir, "records", "exams_part*.hdf5")))

    if not h5_files:
        print("No exams_part*.hdf5 files found.")
        return

    print(f"Found {len(h5_files)} HDF5 files. Starting conversion...")

    # CODE-15% 标准参数
    sig_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    units = ['mV'] * 12
    fs = 400

    total_processed = 0

    for h5_path in h5_files:
        # 获取文件名 stem，例如 "exams_part16"
        file_stem = os.path.splitext(os.path.basename(h5_path))[0]

        # --- 修改核心：为每个 part 创建子目录 ---
        part_output_dir = os.path.join(base_output_dir, file_stem)
        if not os.path.exists(part_output_dir):
            os.makedirs(part_output_dir)
        # -------------------------------------

        print(f"Processing {file_stem} -> {part_output_dir} ...")

        try:
            with h5py.File(h5_path, 'r') as f:
                sig_key, id_key = get_keys(f)

                tracings = f[sig_key]
                exam_ids = f[id_key]

                num_records = len(exam_ids)

                for i in tqdm(range(num_records), desc=file_stem, unit="rec"):
                    current_id = str(exam_ids[i])

                    # 信号处理
                    signal = tracings[i]
                    if np.isnan(signal).any():
                        signal = np.nan_to_num(signal)

                    # 转 int16
                    d_signal = (signal * 1000).astype('int16')

                    # 元数据处理
                    comments = []
                    if current_id in df.index:
                        row = df.loc[current_id]

                        age = row.get('age')
                        if not pd.isna(age):
                            comments.append(f"Age: {int(age)}")

                        is_male = row.get('is_male')
                        if not pd.isna(is_male):
                            sex = 'M' if is_male else 'F'
                            comments.append(f"Sex: {sex}")

                        dx_str = get_diagnosis_string(row)
                        comments.append(f"Dx: {dx_str}")
                    else:
                        comments.append("Metadata not found in CSV")

                    # 写入 WFDB (注意 write_dir 变成了子目录)
                    wfdb.wrsamp(
                        record_name=current_id,
                        fs=fs,
                        units=units,
                        sig_name=sig_names,
                        fmt=['16'] * 12,
                        adc_gain=[1000.0] * 12,
                        baseline=[0] * 12,
                        d_signal=d_signal,
                        comments=comments,
                        write_dir=part_output_dir  # <--- 指向子文件夹
                    )

                    total_processed += 1

        except Exception as e:
            print(f"Failed to process file {h5_path}: {e}")

    print(f"\nSuccessfully converted {total_processed} records.")
    print(f"Data is organized in subdirectories under: {base_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CODE-15% dataset to WFDB format (Organized by parts)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing exams.csv and exams_part*.hdf5 files")

    args = parser.parse_args()
    prepare_code15(args)