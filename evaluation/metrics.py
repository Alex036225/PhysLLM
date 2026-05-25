import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
# from evaluation.post2 import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman
import os


class _RunningStats:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, x):
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        self.n += arr.size
        self.sum += float(np.sum(arr))
        self.sumsq += float(np.sum(arr * arr))

    def mean(self):
        if self.n == 0:
            return np.nan
        return self.sum / self.n

    def std(self):
        if self.n == 0:
            return np.nan
        mean = self.mean()
        var = max(self.sumsq / self.n - mean * mean, 0.0)
        return np.sqrt(var)


def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    hr_pred_stats = _RunningStats()
    hr_gt_stats = _RunningStats()
    snr_stats = _RunningStats()
    macc_stats = _RunningStats()
    abs_err_stats = _RunningStats()
    sq_err_stats = _RunningStats()
    ape_stats = _RunningStats()
    # Pearson running sums
    corr_n = 0
    corr_sumx = 0.0
    corr_sumy = 0.0
    corr_sumx2 = 0.0
    corr_sumy2 = 0.0
    corr_sumxy = 0.0

    print("Calculating metrics!")
    need_macc = "MACC" in config.TEST.METRICS
    subject_keys = list(predictions.keys())
    for index in tqdm(subject_keys, ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak', need_macc=need_macc)
                gt_hr = gt_hr_peak
                pred_hr = pred_hr_peak
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT', need_macc=need_macc)
                gt_hr = gt_hr_fft
                pred_hr = pred_hr_fft
            else:
                raise ValueError("Inference evaluation method name wrong!")

            err = pred_hr - gt_hr
            hr_pred_stats.update(pred_hr)
            hr_gt_stats.update(gt_hr)
            snr_stats.update(SNR)
            if need_macc:
                macc_stats.update(macc)
            abs_err_stats.update(abs(err))
            sq_err_stats.update(err * err)
            if gt_hr != 0:
                ape_stats.update(abs(err / gt_hr) * 100)

            corr_n += 1
            corr_sumx += float(pred_hr)
            corr_sumy += float(gt_hr)
            corr_sumx2 += float(pred_hr * pred_hr)
            corr_sumy2 += float(gt_hr * gt_hr)
            corr_sumxy += float(pred_hr * gt_hr)

        # subject 流式释放，降低峰值内存
        del predictions[index]
        del labels[index]
    
    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'multi_train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    metrics_dict = {}
    num_test_samples = hr_pred_stats.n
    if num_test_samples == 0:
        raise ValueError("No valid test windows were found for metric calculation.")

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = abs_err_stats.mean()
                standard_error = abs_err_stats.std() / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(sq_err_stats.mean())
                standard_error = np.sqrt(sq_err_stats.std() / np.sqrt(num_test_samples))
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
                metrics_dict['RMSE'] = "FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error)
            elif metric == "MAPE":
                MAPE_FFT = ape_stats.mean()
                standard_error = ape_stats.std() / np.sqrt(num_test_samples)
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
                metrics_dict['MAPE'] = "FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error)
            elif metric == "Pearson":
                denom_x = corr_n * corr_sumx2 - corr_sumx * corr_sumx
                denom_y = corr_n * corr_sumy2 - corr_sumy * corr_sumy
                denom = np.sqrt(max(denom_x * denom_y, 0.0))
                if denom == 0:
                    correlation_coefficient = 0.0
                else:
                    correlation_coefficient = (corr_n * corr_sumxy - corr_sumx * corr_sumy) / denom
                if num_test_samples > 2:
                    standard_error = np.sqrt(max((1 - correlation_coefficient**2) / (num_test_samples - 2), 0.0))
                else:
                    standard_error = np.nan
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = snr_stats.mean()
                standard_error = snr_stats.std() / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = macc_stats.mean()
                standard_error = macc_stats.std() / np.sqrt(num_test_samples)
                print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            # elif "BA" in metric:  
                # compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                # compare.scatter_plot(
                #     x_label='GT PPG HR [bpm]',
                #     y_label='rPPG HR [bpm]',
                #     show_legend=True, figure_size=(5, 5),
                #     the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                #     file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                # compare.difference_plot(
                #     x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                #     y_label='Average of rPPG HR and GT PPG HR [bpm]',
                #     show_legend=True, figure_size=(5, 5),
                #     the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                #     file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
            # else:
            #     raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = abs_err_stats.mean()
                standard_error = abs_err_stats.std() / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(sq_err_stats.mean())
                standard_error = np.sqrt(sq_err_stats.std() / np.sqrt(num_test_samples))
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
                metrics_dict['RMSE'] = "PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error)
            elif metric == "MAPE":
                MAPE_PEAK = ape_stats.mean()
                standard_error = ape_stats.std() / np.sqrt(num_test_samples)
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
                metrics_dict['MAPE'] = "PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error)
            elif metric == "Pearson":
                denom_x = corr_n * corr_sumx2 - corr_sumx * corr_sumx
                denom_y = corr_n * corr_sumy2 - corr_sumy * corr_sumy
                denom = np.sqrt(max(denom_x * denom_y, 0.0))
                if denom == 0:
                    correlation_coefficient = 0.0
                else:
                    correlation_coefficient = (corr_n * corr_sumxy - corr_sumx * corr_sumy) / denom
                if num_test_samples > 2:
                    standard_error = np.sqrt(max((1 - correlation_coefficient**2) / (num_test_samples - 2), 0.0))
                else:
                    standard_error = np.nan
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = snr_stats.mean()
                standard_error = snr_stats.std() / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = macc_stats.mean()
                standard_error = macc_stats.std() / np.sqrt(num_test_samples)
                print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
    
    # save_metrics_to_file(metrics_dict, config)
    return metrics_dict


def save_metrics_to_file(metrics_dict, config, epoch, comment=""):
    # 获取基本路径
    base_path = os.path.dirname(config.TEST.OUTPUT_SAVE_DIR)
    
    # 创建新的文件夹名称
    folder_name = f"{config.MODEL.FACE_MODEL}_{config.MODEL.ENV_MODEL}_{config.MODEL.VIDEO_ENC}"
    
    # 创建新的文件夹路径
    folder_path = os.path.join(base_path, folder_name)
    
    # 创建文件夹（如果已存在则不报错）
    os.makedirs(folder_path, exist_ok=True)
    
    file_name = f"LR_{config.TRAIN.LR}_BS_{config.TRAIN.BATCH_SIZE}"
    # 定义保存指标的文件路径
    metrics_file = os.path.join(folder_path, f"{file_name}.txt")
    
    # 将指标写入文件
    with open(metrics_file, "a") as f:
        f.write(F"Epoch:{epoch}.{comment}\n")
        for metric_name, metric_value in metrics_dict.items():
            f.write(f"{metric_name}: {metric_value}\n")
        f.write("\n")
