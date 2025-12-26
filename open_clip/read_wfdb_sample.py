import wfdb
import os
import numpy as np

def read_sample_record():
    # 1. 文件夹路径 (Directory)
    study_dir = "/Users/zhangyf/Documents/cfel/ts_data/files/p1000/p10000032/s40689238"
    
    # 2. 记录名称 (Record Name)
    # 注意：根据您的文件结构，文件名是 '40689238' (没有 's')
    record_name = "40689238"
    
    # 3. 拼接完整路径 (不带后缀 .dat/.hea)
    # 结果应该是: .../s40689238/40689238
    record_path = os.path.join(study_dir, record_name)
    
    print(f"Attempting to read record from: {record_path}")
    
    try:
        # 读取信号和元数据
        signals, fields = wfdb.rdsamp(record_path)
        
        print("\n=== Read Successful ===")
        print(f"Signals Shape: {signals.shape} (Samples x Channels)")
        print(f"Sampling Frequency: {fields['fs']} Hz")
        print(f"Signal Units: {fields['units']}")
        print(f"Channel Names: {fields['sig_name']}")
        
        print("\nFirst 5 signal values:")
        print(signals[:5])
        
        # 如果你想绘制波形 (可选)
        # wfdb.plot_wfdb(record=wfdb.rdrecord(record_path), title='Record ' + record_name)
        
    except Exception as e:
        print(f"\nError reading record: {e}")
        print("Please check if the file path is correct.")

if __name__ == "__main__":
    read_sample_record()