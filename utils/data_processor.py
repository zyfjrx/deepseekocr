import wfdb
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import warnings
import json

# 忽略 NeuroKit2 可能产生的某些 RuntimeWarning (如除以0)
warnings.filterwarnings("ignore")


class RobustECGExtractor:
    """
    【Silver 数据处理器】(100W 弱标签)
    适用场景：拥有 .dat/.hea 原始波形，但没有现成指标。
    功能：使用 NeuroKit2 算法从波形中计算指标，并进行熔断清洗。
    """

    def __init__(self):
        # 标准 12 导联名称
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def clean_value(self, val, min_v, max_v):
        """熔断清洗机制：过滤生理不可能的数值"""
        if val is None or pd.isna(val) or val == "N/A":
            return None
        try:
            val = float(val)
        except:
            return None

        if not (min_v <= val <= max_v):
            return None
        return int(val)

    def get_st_amplitude(self, signal, qrs_offsets, fs):
        """计算 ST 段幅度 (J点后 40ms)"""
        st_amps = []
        for off in qrs_offsets:
            if not np.isnan(off):
                # J point + 40ms
                j_point_idx = int(off + 0.04 * fs)
                if j_point_idx < len(signal):
                    st_amps.append(signal[j_point_idx])
        if st_amps:
            return np.mean(st_amps)
        return None

    def process_record(self, record_path):
        """
        核心处理函数：读取 .dat -> 计算 -> 清洗 -> 格式化
        """
        # 兼容带后缀或不带后缀的路径
        base_path = os.path.splitext(record_path)[0]

        try:
            # 1. 读取数据
            record = wfdb.rdrecord(base_path)
            signals = record.p_signal
            fs = record.fs

            # 容器
            hr_val = None
            lead_metrics = {lead: {'PR': None, 'QRS': None, 'QT': None, 'ST': None} for lead in self.lead_names}

            # A. 计算全局心率 (优先 II 导联)
            ref_lead_idx = 1 if signals.shape[1] > 1 else 0
            try:
                ref_sig = signals[:, ref_lead_idx]
                cleaned_ref = nk.ecg_clean(ref_sig, sampling_rate=fs)
                _, rpeaks = nk.ecg_peaks(cleaned_ref, sampling_rate=fs)
                rpeaks_idx = rpeaks['ECG_R_Peaks']
                if len(rpeaks_idx) > 1:
                    rates = nk.signal_rate(rpeaks_idx, sampling_rate=fs, desired_length=len(cleaned_ref))
                    hr_val = int(np.nanmean(rates))
            except:
                pass

            # B. 逐导联计算
            for i, lead_name in enumerate(self.lead_names):
                if i >= signals.shape[1]: break
                try:
                    sig = signals[:, i]
                    clean_sig = nk.ecg_clean(sig, sampling_rate=fs)
                    _, waves = nk.ecg_delineate(clean_sig, sampling_rate=fs, method='dwt')

                    p_onsets = waves.get('ECG_P_Onsets', [])
                    qrs_onsets = waves.get('ECG_R_Onsets', [])
                    qrs_offsets = waves.get('ECG_R_Offsets', [])
                    t_offsets = waves.get('ECG_T_Offsets', [])

                    metrics = lead_metrics[lead_name]

                    # PR Interval
                    valid_pr = []
                    for p, r in zip(p_onsets, qrs_onsets):
                        if not np.isnan(p) and not np.isnan(r):
                            diff = (r - p) / fs * 1000
                            if 80 <= diff <= 400: valid_pr.append(diff)
                    if valid_pr: metrics['PR'] = np.mean(valid_pr)

                    # QRS Duration
                    valid_qrs = []
                    for on, off in zip(qrs_onsets, qrs_offsets):
                        if not np.isnan(on) and not np.isnan(off):
                            diff = (off - on) / fs * 1000
                            if 40 <= diff <= 250: valid_qrs.append(diff)
                    if valid_qrs: metrics['QRS'] = np.mean(valid_qrs)

                    # QT Interval
                    valid_qt = []
                    min_len = min(len(qrs_onsets), len(t_offsets))
                    for k in range(min_len):
                        on, off = qrs_onsets[k], t_offsets[k]
                        if not np.isnan(on) and not np.isnan(off):
                            diff = (off - on) / fs * 1000
                            if 200 <= diff <= 700: valid_qt.append(diff)
                    if valid_qt: metrics['QT'] = np.mean(valid_qt)

                    # ST Segment
                    st_amp = self.get_st_amplitude(clean_sig, qrs_offsets, fs)
                    if st_amp is not None: metrics['ST'] = st_amp

                except Exception:
                    continue

            # C. 聚合与清洗
            all_pr = [m['PR'] for m in lead_metrics.values() if m['PR'] is not None]
            all_qrs = [m['QRS'] for m in lead_metrics.values() if m['QRS'] is not None]
            all_qt = [m['QT'] for m in lead_metrics.values() if m['QT'] is not None]

            final_hr = self.clean_value(hr_val, 30, 250)
            final_pr = self.clean_value(np.mean(all_pr) if all_pr else None, 80, 400)
            final_qrs = self.clean_value(np.mean(all_qrs) if all_qrs else None, 40, 220)
            final_qt = self.clean_value(np.mean(all_qt) if all_qt else None, 200, 650)

            # ST 异常检测 (Silver 数据可以算出来)
            abnormal_st_list = []
            for lead, m in lead_metrics.items():
                if m['ST'] is not None and abs(m['ST']) > 0.1:
                    abnormal_st_list.append(f"{lead}({m['ST']:.2f})")

            return self._format_output(final_hr, final_pr, final_qrs, final_qt, abnormal_st_list)

        except Exception as e:
            return "Measurements: N/A"

    def _format_output(self, hr, pr, qrs, qt, st_list=None):
        """统一输出格式化"""
        parts = []
        parts.append(f"HR: {hr if hr else 'N/A'}")
        parts.append(f"PR: {pr if pr else 'N/A'}")
        parts.append(f"QRS: {qrs if qrs else 'N/A'}")
        parts.append(f"QT: {qt if qt else 'N/A'}")

        if st_list and len(st_list) > 0:
            st_str = ",".join(st_list[:5])
            parts.append(f"ST Abn: {st_str}")
        else:
            parts.append("ST: Isoelectric")

        return " | ".join(parts)


class GoldMeasurementProcessor:
    """
    【Gold 数据处理器】(3W 强标签)
    适用场景：MIMIC-IV 的 machine_measurements.csv/.json。
    功能：解析离散的数组数据，计算中位数聚合，并格式化为与 Silver 一致的字符串。
    """

    def process_json(self, json_data):
        """
        输入: 包含 measurements 的字典或 JSON 字符串
        输出: 紧凑的 Prompt 字符串
        """
        try:
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data

            # 1. 提取 Global Features
            # MIMIC 有时 key 是 "Heart Rate" 有时是 "Heart Rate (bpm)"，做个容错
            global_feats = data.get('global_features', {})
            hr = global_feats.get('Heart Rate (bpm)') or global_feats.get('Heart Rate')

            # 2. 提取并聚合 Lead Features (数组 -> 中位数)
            features = data.get('lead_features', {})
            pr_list, qrs_list, qt_list = [], [], []

            for lead, feats in features.items():
                pr = self._get_median_from_str(feats.get('PR Interval (ms)', '[]'))
                qrs = self._get_median_from_str(feats.get('QRS Duration (ms)', '[]'))
                qt = self._get_median_from_str(feats.get('QT Interval (ms)', '[]'))

                if pr: pr_list.append(pr)
                if qrs: qrs_list.append(qrs)
                if qt: qt_list.append(qt)

            # 3. 计算全局平均
            final_pr = int(np.mean(pr_list)) if pr_list else None
            final_qrs = int(np.mean(qrs_list)) if qrs_list else None
            final_qt = int(np.mean(qt_list)) if qt_list else None

            # Gold 数据通常没有 ST Amplitude，填 N/A
            return self._format_output(hr, final_pr, final_qrs, final_qt)

        except Exception as e:
            return "Measurements: N/A"

    def _get_median_from_str(self, val_str):
        """解析 '[1, 2, 3]' 字符串并取中位数"""
        try:
            if isinstance(val_str, str) and val_str.startswith('['):
                vals = json.loads(val_str)
                vals = [v for v in vals if v is not None]
                if vals:
                    return np.median(vals)
            elif isinstance(val_str, (int, float)):
                return val_str
        except:
            pass
        return None

    def _format_output(self, hr, pr, qrs, qt):
        """与 Silver 保持完全一致的格式"""
        parts = []
        parts.append(f"HR: {hr if hr else 'N/A'}")
        parts.append(f"PR: {pr if pr else 'N/A'}")
        parts.append(f"QRS: {qrs if qrs else 'N/A'}")
        parts.append(f"QT: {qt if qt else 'N/A'}")
        parts.append("ST: N/A")  # Gold 数据默认没有 ST 电压值
        return " | ".join(parts)


class ECGDataProcessor:
    """
    【统一入口类】
    在 ecg_dataset.py 或 预处理脚本中调用此类。
    """

    def __init__(self):
        self.silver_extractor = RobustECGExtractor()
        self.gold_processor = GoldMeasurementProcessor()

    def process(self, input_data, mode='silver'):
        """
        通用处理接口
        :param input_data:
            - mode='silver': 文件路径 (str), 例如 "mimic-iv/files/..."
            - mode='gold': 测量数据 (dict 或 json str)
        :param mode: 'silver' | 'gold'
        :return: 格式化后的 Prompt 字符串 (如 "HR: 90 | PR: 151 ...")
        """
        if mode == 'silver':
            return self.silver_extractor.process_record(input_data)
        elif mode == 'gold':
            return self.gold_processor.process_json(input_data)
        else:
            raise ValueError("Mode must be 'silver' or 'gold'")


# ================= 使用示例 =================
if __name__ == "__main__":
    processor = ECGDataProcessor()

    # 示例 1: Silver 数据 (只有文件路径)
    # path = "/path/to/ecg_file"
    # print(processor.process(path, mode='silver'))

    # 示例 2: Gold 数据 (有 JSON 指标)
    # json_data = '{"global_features": {"Heart Rate (bpm)": "71"}, "lead_features": {...}}'
    # print(processor.process(json_data, mode='gold'))