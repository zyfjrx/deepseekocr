import os
from typing import Optional
import scipy
import torch
import wfdb
import numpy as np

def load_ecg_data(full_path: str) -> Optional[torch.Tensor]:
    """
    读取 .dat/.hea 文件并预处理为 [12, 5000] 的 Tensor
    """
    if not wfdb:
        return None



    # wfdb 读取时不带后缀
    record_path = os.path.splitext(full_path)[0]

    if not (os.path.exists(record_path + ".hea") or os.path.exists(record_path + ".dat")):
        # 尝试直接读取（如果 rel_path 本身就不带后缀）
        if not (os.path.exists(full_path + ".hea")):
            return None
        record_path = full_path

    try:
        # 读取记录
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # shape: (Samples, Channels)通常是 (N, 12)
        original_fs = record.fs

        # 目标参数
        target_fs = 500
        target_len = 5000  # 10秒

        # 1. 重采样 (Resample)
        if original_fs != target_fs:
            new_len = int(signal.shape[0] * target_fs / original_fs)
            # 使用 scipy 进行重采样
            signal = scipy.signal.resample(signal, new_len, axis=0)

        # 2. 截断或填充 (Pad/Crop)
        L, C = signal.shape
        if L > target_len:
            # 截取中间部分或开头，这里取开头
            signal = signal[:target_len, :]
        elif L < target_len:
            # 零填充
            pad_len = target_len - L
            padding = np.zeros((pad_len, C))
            signal = np.concatenate([signal, padding], axis=0)

        # 3. 转置为 [Channels, Length] -> [12, 5000]
        signal_tensor = torch.tensor(signal.T, dtype=torch.float32)

        # 4. 简单归一化 (Z-Score per channel) - 可选
        # mean = signal_tensor.mean(dim=1, keepdim=True)
        # std = signal_tensor.std(dim=1, keepdim=True) + 1e-5
        # signal_tensor = (signal_tensor - mean) / std

        return signal_tensor

    except Exception as e:
        print(f"Error loading ECG {record_path}: {e}")
        return None

if __name__ == '__main__':
    result = load_ecg_data("/Users/zhangyf/Documents/cfel/code15/h5py/records_wfdb/1819953")
    print(result.shape)