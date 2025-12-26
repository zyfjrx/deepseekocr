import sys
import os
import pandas as pd
import numpy as np
import torch
import unittest.mock as mock
from torch.utils.data import DataLoader

# 添加路径以导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入待测试的模块
try:
    from training.data import load_mimic_iv_ecg, ECGTextDataset
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def create_dummy_csvs(base_dir):
    """创建模拟的 MIMIC-IV CSV 文件结构"""
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. machine_measurements.csv (存储报告)
    # 结构: study_id, report_0 ... report_17, 以及波形特征列
    data = {
        'study_id': [1001, 1002, 1003],
        'report_0': ['Sinus rhythm', 'Atrial fibrillation', 'Normal ECG'],
        'report_1': ['Normal axis', 'Rapid ventricular response', ''],
        # 添加 wfep 需要的特征列
        'RR_Interval': [800, 600, 900],
        'PR_Interval': [160, 0, 150],
        'QRS_Complex': [90, 80, 85],
        'QT_Interval': [360, 320, 380],
        'QTc_Interval': [400, 410, 390],
        'P_Wave_Peak': [0.1, 0, 0.1],
        'R_Wave_Peak': [1.0, 0.8, 1.1],
        'T_Wave_Peak': [0.2, 0.1, 0.25]
    }
    # 填充剩余的 report 列
    for i in range(2, 18):
        data[f'report_{i}'] = [''] * 3
        
    df_measurements = pd.DataFrame(data)
    df_measurements.to_csv(os.path.join(base_dir, "machine_measurements.csv"), index=False)
    print(f"Created dummy measurements at {base_dir}/machine_measurements.csv")

    # 2. new_record_list.csv (存储文件路径和划分)
    # file_name % 10 > 0 -> Train, % 10 == 0 -> Test
    records = {
        'study_id': [1001, 1002, 1003],
        'file_name': [11, 12, 20], # 11, 12 -> Train; 20 -> Test
        'path': ['data/1001', 'data/1002', 'data/1003']
    }
    df_records = pd.DataFrame(records)
    df_records.to_csv(os.path.join(base_dir, "new_record_list.csv"), index=False)
    print(f"Created dummy records at {base_dir}/new_record_list.csv")

class MockTokenizer:
    """简单的 Mock Tokenizer"""
    def __call__(self, text):
        # 返回随机的 token ids，长度 77 (CLIP 默认)
        return torch.randint(0, 1000, (1, 77))

def main():
    dummy_dir = os.path.join(current_dir, "dummy_data")
    create_dummy_csvs(dummy_dir)
    
    print("\n" + "="*30 + " Starting Data Pipeline Test " + "="*30)
    
    # Mock wfdb.rdsamp
    # 它的作用是读取心电文件，我们直接拦截它，返回随机数组
    with mock.patch('wfdb.rdsamp') as mock_rdsamp:
        # 设置返回值: (signals, fields)
        # signals shape: (5000, 12)
        mock_rdsamp.return_value = (np.random.randn(5000, 12).astype(np.float32), {})
        
        print("1. Testing load_mimic_iv_ecg function...")
        # 调用数据加载逻辑 (启用 wfep 以测试波形特征拼接)
        train_x, train_y, val_x, val_y, test_x, test_y = load_mimic_iv_ecg(dummy_dir, wfep=True)
        
        print(f"  Train Samples: {len(train_x)}")
        print(f"  Test Samples:  {len(test_x)}")
        print(f"  Sample Path:   {train_x[0]}")
        print(f"  Sample Text:   {train_y[0]}")
        print("  (Note: Check if text contains 'RR:' etc. to verify WFEP logic)")
        
        print("\n2. Testing ECGTextDataset...")
        # 实例化 Dataset
        dataset = ECGTextDataset(
            path=train_x, 
            texts=train_y, 
            tokenizer=MockTokenizer(),
            is_train=True
        )
        
        print(f"  Dataset Length: {len(dataset)}")
        
        # 获取一个样本
        ecg_tensor, text_tokens = dataset[0]
        print(f"  ECG Tensor Shape: {ecg_tensor.shape} (Expected: 12, 5000 or similar)")
        print(f"  Text Tokens Shape: {text_tokens.shape}")
        
        print("\n3. Testing DataLoader Iteration...")
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        for batch_ecg, batch_text in loader:
            print(f"  Batch ECG Shape: {batch_ecg.shape}")
            print(f"  Batch Text Shape: {batch_text.shape}")
            break
            
    print("\n" + "="*30 + " Test Finished Successfully " + "="*30)
    
    # 清理生成的临时文件 (可选)
    # import shutil
    # shutil.rmtree(dummy_dir)

if __name__ == "__main__":
    # main()
    import wfdb
    signals, fields = wfdb.rdsamp("/Users/zhangyf/Documents/cfel/ts_data/files/p1000/p10000032/s40689238")
    print( signals)
    print( fields)