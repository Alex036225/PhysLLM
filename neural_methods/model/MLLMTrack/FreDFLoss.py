import torch
import torch.nn as nn

class FreDLyLoss(nn.Module):
    def __init__(self, dim=1):
        """
        初始化频率域损失函数。

        参数:
        - dim: 指定进行FFT变换的维度,默认为1。
        """
        super(FreDLyLoss, self).__init__()
        self.dim = dim

    def forward(self, outputs, target):
        """
        计算频率域损失。

        参数:
        - outputs: 模型的输出张量 (形状与target一致)
        - target: 真实的目标张量

        返回:
        - loss: 频率域上的平均绝对误差损失
        """
        # 对outputs和target进行快速傅里叶变换（RFFT）
        fft_outputs = torch.fft.rfft(outputs, dim=self.dim)
        fft_target = torch.fft.rfft(target, dim=self.dim)

        # 计算频域差异并取绝对值
        freq_diff = (fft_outputs - fft_target).abs()

        # 计算平均值作为损失
        loss = freq_diff.mean()

        return loss