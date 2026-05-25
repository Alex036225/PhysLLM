# -*- coding: utf-8 -*-
"""
ram.py
Residual Amortized Transport (RAM)
----------------------------------
将一次 Sinkhorn 计算到的 OT 重心 “摊销”为一个可学习的前馈映射，
训练阶段以 stop-grad 的 OT 重心 y★ 作为教师信号，
推理阶段直接用 Tθ(x) 输出对齐特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAmortizedTransport(nn.Module):
    """Residual Amortized Transport: C→C 的轻量映射"""
    def __init__(self, feat_dim: int, hidden_mul: float = 1.5, dropout: float = 0.1):
        super().__init__()
        H = max(8, int(hidden_mul * feat_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, H),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(H, feat_dim)
        )
        self.gate = nn.Parameter(torch.tensor(0.7))  # 融合教师比例（sigmoid范围0~1）

    def forward(self, x_btc: torch.Tensor, y_star_btc: torch.Tensor = None):
        """x_btc:(B,T,C), y_star_btc:(B,T,C) 为 OT 重心(stop_grad)"""
        y_pred = x_btc + self.net(x_btc)
        if y_star_btc is None:
            return y_pred, torch.tensor(0., device=x_btc.device)
        with torch.no_grad():
            y_star = y_star_btc
        lam = torch.sigmoid(self.gate)
        y_mix = lam * y_pred + (1 - lam) * y_star
        L_map = F.mse_loss(y_pred, y_star)
        return y_mix, L_map
