# 频域损失的简单实现示例
import torch
import torch.nn as nn

class SpectralLoss(nn.Module):
    def __init__(self, Fs=30.0, high_pass=0.75, low_pass=3.0):
        super(SpectralLoss, self).__init__()
        self.Fs = Fs
        self.high_pass_idx = int(high_pass / (self.Fs / 2) * (128 // 2)) # frames需要传入
        self.low_pass_idx = int(low_pass / (self.Fs / 2) * (128 // 2))

    def forward(self, pred_signal, true_signal):
        # FFT
        pred_fft = torch.fft.rfft(pred_signal, dim=-1)
        true_fft = torch.fft.rfft(true_signal, dim=-1)
        pred_power = torch.abs(pred_fft) ** 2
        true_power = torch.abs(true_fft) ** 2

        # 关注心率区间
        pred_power_band = pred_power[:, self.high_pass_idx:self.low_pass_idx]
        true_power_band = true_power[:, self.high_pass_idx:self.low_pass_idx]

        # 可以用L1或MSE等计算损失
        loss = nn.L1Loss()(pred_power_band, true_power_band)
        return loss