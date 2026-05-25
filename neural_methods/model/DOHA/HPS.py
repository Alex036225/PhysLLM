# Copyright ©2022 Sun weiyu and Chen ying. All Rights Reserved.
import os
import torch

import numpy as np
from scipy.signal import hilbert
import math


def self_similarity_calc(ippg: torch.Tensor):
    """
    计算 ippg 信号的 phase 相似性矩阵，可导。
    ippg: shape [T]，1D tensor，模型输出通道之一
    返回: shape [T, T] 的 similarity 矩阵
    """

    # 1. Hilbert 变换 - 生成解析信号
    analytic_signal = torch.fft.fft(ippg)  # complex tensor
    analytic_signal = torch.fft.ifft(analytic_signal)  # 变换回 time domain，保持复数

    # 2. 获取相位（phase）
    phase = torch.atan2(analytic_signal.imag, analytic_signal.real)  # [-π, π]

    # 3. phase similarity matrix，使用余弦相似度
    phase_i = phase.unsqueeze(0)  # [1, T]
    phase_j = phase.unsqueeze(1)  # [T, 1]
    sim_matrix = torch.cos(phase_i - phase_j)  # [T, T]

    return sim_matrix  # 保持在计算图中


# def self_similarity_calc(ippg):
#     ippg_phase0 = myhilbert(ippg)
#     ippg_phase = amass_hilbort(ippg_phase0)[1:]
#     result_list = []
#     for i in range(len(ippg_phase)):
#         tmp_list = []
#         for j in range(len(ippg_phase)):
#             similarity = np.cos(ippg_phase[i] - ippg_phase[j])
#             tmp_list.append(similarity)
#         tmp_list = torch.FloatTensor(tmp_list).unsqueeze(-1)
#         result_list.append(tmp_list)
#     result = torch.cat(result_list, dim=-1)
#     return result


def amass_hilbort(ippg):
    peak_record = [(0, 0)]
    current_sum = 0
    for i in range(1, len(ippg)):
        if (ippg[i] - ippg[i - 1]) < 0:
            current_sum += ippg[i - 1] - ippg[i]
            peak_record.append((i, current_sum))

    sum_list = [peak_record[i][1] for i in range(len(peak_record))]
    peak_record.append((len(ippg), None))
    record_list = [(peak_record[i][0], peak_record[i + 1][0]) for i in range(len(peak_record) - 1)]
    result =[]
    for i in range(len(ippg)):
        for j in range(len(record_list)):
            if record_list[j][0] <= i < record_list[j][1]:
                ans = ippg[i] + sum_list[j]
                while ans > 2 * np.pi:
                    ans -= 2 * np.pi
                result.append(ans)

    return result

def myhilbert(ippg_test):
    ippg_hilbert = hilbert(ippg_test)
    N = len(ippg_test)
    ippg_hilbert_phase = np.zeros(N)
    ippg_hilbert_phase_shift = np.zeros(N)
    for i in range(N):
        if ippg_hilbert[i].real == 0:
            if (ippg_hilbert[i].imag > 0):
                ippg_hilbert_phase[i] = 1.57
            elif (ippg_hilbert[i].imag < 0):
                ippg_hilbert_phase[i] = -1.57
            else:
                ippg_hilbert_phase[i] = 0
        else:
            ippg_hilbert_phase[i] = math.atan(ippg_hilbert[i].imag / ippg_hilbert[i].real)
    k = 1
    for i in range(N):
        if i != 0 and ippg_hilbert_phase[i] - ippg_hilbert_phase[i - 1] < -1.5:
            k = -k
        ippg_hilbert_phase_shift[i] = k * ippg_hilbert_phase[i]
    return ippg_hilbert_phase