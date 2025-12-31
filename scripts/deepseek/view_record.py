import wfdb
import matplotlib.pyplot as plt
import numpy as np

# 记录名（不带后缀）
record_name = '/Users/zhangyf/Documents/cfel/code15/h5py/records/exams_part17/1819953'

try:
    # 1. 读取记录
    # rdrecord 会自动寻找同名的 .hea 和 .dat 文件
    record = wfdb.rdrecord(record_name)

    # 2. 打印元数据信息
    print(f"=== Record: {record.record_name} ===")
    print(f"采样率 (fs): {record.fs} Hz")
    print(f"信号长度: {record.sig_len} samples")
    print(f"信号形状: {record.p_signal.shape}")  # 应该是 (4096, 12)
    print(f"导联名称: {record.sig_name}")
    print(f"包含的注释 (Comments): {record.comments}")  # 检查年龄、性别、诊断是否在里面

    # 3. 绘制波形 (使用 wfdb 自带绘图功能)
    print("正在绘图...")
    wfdb.plot_wfdb(
        record=record,
        title=f'Record {record_name} (CODE-15% Converted)',
        time_units='seconds',
        figsize=(12, 10)
    )

    #手动画前两个导联（I 和 II）以检查细节
    plt.figure(figsize=(12, 4))
    # 生成时间轴
    t = np.arange(record.sig_len) / record.fs

    plt.plot(t, record.p_signal[:, 0], label=record.sig_name[0])  # Lead I
    plt.plot(t, record.p_signal[:, 1] + 2, label=record.sig_name[1])  # Lead II (向上平移2mV以免重叠)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("Lead I & II Preview")
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"读取失败: {e}")