import numpy as np
import pandas as pd
import os
import wfdb


def get_wave_info(data):
    keys = ['RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval',
            'QTc_Interval', 'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak']
    text_describe = ""
    text_describe += f" RR: {data['RR_Interval']}"
    text_describe += f" PR: {data['PR_Interval']}"
    text_describe += f" QRS: {data['QRS_Complex']}"
    text_describe += f" QT/QTc: {data['QT_Interval']}/{data['QTc_Interval']}"
    text_describe += f" P/R/T Wave: {data['P_Wave_Peak']}/{data['R_Wave_Peak']}/{data['T_Wave_Peak']}"
    return text_describe

def load_mimic_iv_ecg(path, wfep=True):
    database = pd.read_csv(os.path.join(path, "machine_measurements.csv")).set_index("study_id")
    record_list = pd.read_csv(os.path.join(path, "new_record_list.csv"))

    indexes = record_list.index.values
    np.random.seed(0)
    np.random.shuffle(indexes)

    train_list = record_list.loc[np.where(record_list["file_name"].values % 10 > 0)].set_index("study_id")
    test_list = record_list.loc[np.where(record_list["file_name"].values % 10 == 0)].set_index("study_id")
    train_indexes = train_list.index.values
    val_indexes = test_list.index.values[-10000:-5000]
    test_indexes = test_list.index.values[-5000:]
    train_indexes = np.append(train_indexes, test_indexes[:-10000])

    def data(index_list):
        reports = []
        X = []
        n_reports = 18
        bad_reports = ["--- Warning: Data quality may affect interpretation ---",
                       "--- Recording unsuitable for analysis - please repeat ---",
                       "Analysis error",
                       "conduction defect",
                       "*** report made without knowing patient's sex ***",
                       "--- Suspect arm lead reversal",
                       "--- Possible measurement error ---",
                       "--- Pediatric criteria used ---",
                       "--- Suspect limb lead reversal",
                       "-------------------- Pediatric ECG interpretation --------------------",
                       "Lead(s) unsuitable for analysis:",
                       "LEAD(S) UNSUITABLE FOR ANALYSIS:",
                       "PACER DETECTION SUSPENDED DUE TO EXTERNAL NOISE-REVIEW ADVISED",
                       "Pacer detection suspended due to external noise-REVIEW ADVISED"]

        for i in index_list:
            row = record_list.loc[i]
            m_row = database.loc[i]
            report_txt = ""
            for j in range(n_reports):
                report = m_row[f"report_{j}"]
                if type(report) == str:
                    is_bad = False
                    for bad_report in bad_reports:
                        if report.find(bad_report) > -1:
                            is_bad = True
                            break
                    report_txt += (report + " ") if not is_bad else ""
            if report_txt == "":
                continue
            report_txt = report_txt[:-1].lower()
            report_txt = (report_txt.replace("---", "")
                          .replace("***", "")
                          .replace(" - age undetermined", ""))

            report_txt = (report_txt.replace('rbbb', 'right bundle branch block')
                          .replace('lbbb', 'light bundle branch block')
                          .replace('lvh', 'left ventricle hypertrophy')
                          .replace("mi", "myocardial infarction")
                          .replace("lafb", "left anterior fascicular block")
                          .replace("pvc(s)", "ventricular premature complex")
                          .replace("pvcs", "ventricular premature complex")
                          .replace("pac(s)", "atrial premature complex")
                          .replace("pacs", "atrial premature complex"))
            if wfep:
                report_txt = report_txt + get_wave_info(row)
            reports.append(report_txt)
            X.append(os.path.join(path, row["path"]))
        return X, reports

    record_list = record_list.set_index("study_id")
    train_x, train_y = data(train_indexes)

    val_x, val_y = data(val_indexes)
    test_x, test_y = data(test_indexes)
    return train_x, train_y, val_x, val_y, test_x, test_y

if __name__ == '__main__':
    load_mimic_iv_ecg("/Users/zhangyf/Documents/cfel/ts_data")
