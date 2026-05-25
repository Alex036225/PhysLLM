# -*- coding: utf-8 -*-
"""
transportnet.py
---------------------------------------
面向 rPPG 域泛化的“时序保形 + 条件 OT + 重心原型”模块集合。
新增：不依赖输入维度的核心组件（对齐器/原型/代价/Sinkhorn/频带能量），
真正的 3D/4D/5D 适配逻辑放在 wrapper.py 的 ShapeAdapter 中统一处理。

核心时序接口仍为 (B, T, C)。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1) 时序保形对齐器
# =========================

class DSConv1d(nn.Module):
    """Depthwise-Separable Conv1d + GLU + InstanceNorm 残差块（保持 (B,T,C)）"""
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.pw_in = nn.Conv1d(channels, 2 * channels, kernel_size=1, bias=True)
        self.pw_out = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        # 原: GroupNorm -> 现: InstanceNorm1d（对 (B,C,T) 生效）
        self.norm = nn.InstanceNorm1d(channels, affine=True, track_running_stats=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_btC: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B,C,T)
        x = x_btC.transpose(1, 2)
        y = self.dw(x)
        y = self.pw_in(y)
        a, b = y.chunk(2, dim=1)          # GLU
        y = a * torch.sigmoid(b)
        y = self.pw_out(y)
        y = self.norm(y)                  # (B,C,T)
        y = self.dropout(y)
        return (x + y).transpose(1, 2)     # 残差 + 回到 (B,T,C)



class TemporalAligner(nn.Module):
    """
    轻量时序对齐器：N 层 DS-Conv1d 残差堆叠，可选 1 层自注意力。
    目的：在时间维上进行“保形”的域内对齐，避免打平破坏生理结构。
    统一使用 (B,T,C)。
    """
    def __init__(self,
                 feat_dim: int,
                 layers: int = 3,
                 kernel_size: int = 3,
                 use_attention: bool = False,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            DSConv1d(feat_dim, kernel_size, dropout) for _ in range(layers)
        ])
        self.use_attention = use_attention
        if use_attention:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=num_heads,
                dim_feedforward=feat_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu"
            )
            self.att = nn.TransformerEncoder(enc_layer, num_layers=1)
        else:
            self.att = None
        self.post = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        """
        y = x
        for block in self.blocks:
            y = block(y)
        if self.use_attention:
            y = self.att(y)  # (B, T, C)
        return self.post(y)


# =========================
# 2) 重心原型
# =========================

class PrototypeBank(nn.Module):
    """
    可学习的重心原型集合：K 个 C 维原型，以及对应的 HR 原型值。
    作为“多源统一对齐的共同空间”（barycenter 近似）。
    """
    def __init__(self, feat_dim: int, num_prototypes: int = 64, init_scale: float = 0.02):
        super().__init__()
        self.prototypes = nn.Parameter(init_scale * torch.randn(num_prototypes, feat_dim))
        hr_init = 80.0 + 10.0 * torch.randn(num_prototypes)  # 初值落在 60~100 BPM 左右
        self.hr_proto = nn.Parameter(hr_init)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            P: (K, C)
            h: (K,)
        """
        return self.prototypes, self.hr_proto


# =========================
# 3) 条件代价函数
# =========================

@dataclass
class CostConfig:
    lambda_hr: float = 1.0
    sigma_hr: float = 10.0        # HR 高斯核带宽
    use_learnable_w: bool = True  # 是否学习频带/通道权重 W


class ConditionalCost(nn.Module):
    """
    c(x,y) = ||x - y||^2_W  +  λ_hr * (1 - exp(-(hr_x - hr_y)^2 / (2σ^2)))

    - X: (B, T, C)
    - Y: (K, C)
    - hr_x: (B,) 或 (B,T)（默认 (B,) 并广播到 T）
    - hr_y: (K,)
    输出 (B, T, K) 的成对代价
    """
    def __init__(self, feat_dim: int, cfg: CostConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_learnable_w:
            self.w = nn.Parameter(torch.ones(feat_dim))
        else:
            self.register_buffer("w", torch.ones(feat_dim))

    def feature_cost(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        加权 L2: ||x - y||^2_W
        X: (B, T, C), Y: (K, C)
        Return: (B, T, K)
        """
        w = self.w.abs()  # 非负
        Xw = X * w        # (B,T,C)
        Yw = Y * w        # (K,C)
        x2 = (Xw ** 2).sum(dim=-1, keepdim=True)            # (B,T,1)
        y2 = (Yw ** 2).sum(dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,K)
        xy = torch.einsum('btc,kc->btk', Xw, Yw)            # (B,T,K)
        return x2 + y2 - 2.0 * xy

    def hr_kernel_term(self, hr_x: torch.Tensor, hr_y: torch.Tensor, T: int) -> torch.Tensor:
        """
        hr_x: (B,) or (B,T)
        hr_y: (K,)
        Return: (B, T, K) = λ_hr * (1 - k_hr)
        """
        if hr_x.dim() == 1:
            hr_x = hr_x.unsqueeze(1).expand(-1, T)  # (B,T)
        diff2 = (hr_x.unsqueeze(-1) - hr_y.view(1, 1, -1)) ** 2  # (B,T,K)
        k_hr = torch.exp(- diff2 / (2 * (self.cfg.sigma_hr ** 2) + 1e-8))
        return self.cfg.lambda_hr * (1.0 - k_hr)

    def forward(self, X: torch.Tensor, Y: torch.Tensor,
                hr_x: torch.Tensor, hr_y: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            cost(X,Y): (B, T, K)
        """
        B, T, _ = X.shape
        c_feat = self.feature_cost(X, Y)               # (B,T,K)
        c_hr = self.hr_kernel_term(hr_x, hr_y, T)      # (B,T,K)
        return c_feat + c_hr


# =========================
# 4) Sinkhorn & 去偏 SinkhornDivergence
# =========================

@dataclass
class SinkhornConfig:
    epsilon: float = 0.08   # entropic 正则
    n_iters: int = 50
    tol: float = 1e-3
    debiased: bool = True   # 使用去偏的 Sinkhorn divergence
    stable: bool = True     # 使用 log-domain 稳定实现


class Sinkhorn(nn.Module):
    """
    Batched Sinkhorn (source=(B,T), target=(K)) with uniform weights by default.
    允许传入权重 a=(B,T) 与 b=(K,).
    成本矩阵 C: (B, T, K)
    返回：
      - π: (B, T, K) 运输计划
      - cost: 标量，总 OT 代价 sum(π*C)
    """
    def __init__(self, cfg: SinkhornConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, C: torch.Tensor,
                a: Optional[torch.Tensor] = None,
                b: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        C: (B, T, K)
        a: (B, T)  源分布，默认均匀
        b: (K,)    目标分布，默认均匀
        """
        eps = self.cfg.epsilon
        n_iters = self.cfg.n_iters
        B, T, K = C.shape

        if a is None:
            a = torch.full((B, T), 1.0 / T, device=C.device, dtype=C.dtype)
        if b is None:
            b = torch.full((K,), 1.0 / K, device=C.device, dtype=C.dtype)

        if not self.cfg.stable:
            Kmat = torch.exp(-C / eps)          # (B,T,K)
            u = torch.ones((B, T), device=C.device, dtype=C.dtype) / T
            v = torch.ones((B, K), device=C.device, dtype=C.dtype) / K
            for _ in range(n_iters):
                Kv = torch.bmm(Kmat, v.unsqueeze(-1)).squeeze(-1) + 1e-12
                u = a / Kv
                Ktu = torch.bmm(Kmat.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-12
                v = (b.unsqueeze(0)) / Ktu
            P = u.unsqueeze(-1) * Kmat * v.unsqueeze(1)  # (B,T,K)
            ot_cost = (P * C).sum()
            return P, ot_cost

        # log-domain 实现（更稳定）
        log_a = torch.log(a + 1e-12)            # (B,T)
        log_b = torch.log(b + 1e-12)            # (K,)
        log_u = torch.zeros_like(log_a)         # (B,T)
        log_v = torch.zeros((B, K), device=C.device, dtype=C.dtype)  # (B,K)

        for _ in range(n_iters):
            # log_u = log_a - logsumexp_j( -C/eps + log_v_j )
            logKv = torch.logsumexp((-C / eps) + log_v.unsqueeze(1), dim=2)  # (B,T)
            log_u = log_a - logKv

            # log_v = log_b - logsumexp_i( -C/eps + log_u_i )
            logKtu = torch.logsumexp((-C / eps) + log_u.unsqueeze(2), dim=1)  # (B,K)
            log_v = log_b.unsqueeze(0) - logKtu

        logP = (-C / eps) + log_u.unsqueeze(2) + log_v.unsqueeze(1)  # (B,T,K)
        P = torch.exp(logP)
        ot_cost = (P * C).sum()
        return P, ot_cost


class SinkhornDivergence(nn.Module):
    """
    去偏的 Sinkhorn 发散：
      S_ε(μ,ν) = W_ε(μ,ν) - 0.5 W_ε(μ,μ) - 0.5 W_ε(ν,ν)

    这里 μ 是 (B,T) 上的经验分布，ν 是 (K) 上的原型分布。
    为简洁与数值稳定，自项用特征项近似（实践上足够稳）。
    """
    def __init__(self, sinkhorn: Sinkhorn):
        super().__init__()
        self.sinkhorn = sinkhorn

    def forward(self,
                C_xy: torch.Tensor,
                X: torch.Tensor, hr_x: torch.Tensor,
                Y: torch.Tensor, hr_y: torch.Tensor,
                w: torch.Tensor,
                a: Optional[torch.Tensor] = None,
                b: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        C_xy: (B,T,K) 由 ConditionalCost 计算
        返回：
          - Sε(μ,ν) 标量
          - 诊断信息 dict
        """
        # cross term
        P_xy, W_xy = self.sinkhorn(C_xy, a=a, b=b)

        if not self.sinkhorn.cfg.debiased:
            return W_xy, {"W_xy": W_xy, "P_xy": P_xy}

        # self-terms（仅特征项近似，自项 HR 项影响很小，避免额外 O(T^2)/O(K^2) 代价）
        # μ-μ
        B, T, C = X.shape
        Xw = X * w
        x2 = (Xw ** 2).sum(dim=-1, keepdim=True)               # (B,T,1)
        y2 = (Xw ** 2).sum(dim=-1).unsqueeze(1)                 # (B,1,T)
        xy = torch.bmm(Xw, Xw.transpose(1, 2))                  # (B,T,T)
        C_xx_feat = x2 + y2 - 2.0 * xy

        a_x = torch.full((B, T), 1.0 / T, device=X.device, dtype=X.dtype)
        b_x = torch.full((T,), 1.0 / T, device=X.device, dtype=X.dtype)
        W_xx = 0.0
        for bidx in range(B):
            _, W_xx_b = self.sinkhorn(C_xx_feat[bidx:bidx+1, :, :], a=a_x[bidx:bidx+1, :], b=b_x)
            W_xx = W_xx + W_xx_b

        # ν-ν
        Yw = Y * w
        x2y = (Yw ** 2).sum(dim=-1, keepdim=True)               # (K,1)
        y2y = (Yw ** 2).sum(dim=-1).unsqueeze(0)                 # (1,K)
        xyy = torch.matmul(Yw, Yw.transpose(0, 1))               # (K,K)
        C_yy_feat = x2y + y2y - 2.0 * xyy

        C_yy_b = C_yy_feat.unsqueeze(0).expand(B, -1, -1)  # (B,K,K)
        a_y = torch.full((B, Y.shape[0]), 1.0 / Y.shape[0], device=Y.device, dtype=Y.dtype)
        b_y = torch.full((Y.shape[0],), 1.0 / Y.shape[0], device=Y.device, dtype=Y.dtype)
        _, W_yy = self.sinkhorn(C_yy_b, a=a_y, b=b_y)

        S = W_xy - 0.5 * W_xx - 0.5 * W_yy
        diag = {
            "W_xy": W_xy.detach(),
            "W_xx": W_xx.detach(),
            "W_yy": W_yy.detach(),
            "P_xy": P_xy.detach(),
        }
        return S, diag


# =========================
# 5) 简易频带能量（正则用）
# =========================

class BandEnergy(nn.Module):
    """
    在时间维 T 上估计每个通道的频带能量，用于“对齐前后频带能量保持”正则。
    - 需要提供采样率 fps 与目标频带（Hz）
    - 对 (B,T,C) 先在 T 上做 FFT，得到功率谱，再按频率索引积分
    """
    def __init__(self, fps: float = 30.0, band: Tuple[float, float] = (0.7, 3.0)):
        super().__init__()
        self.fps = fps
        self.band = band

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B,T,C)
        Return: (B,C) 频带能量
        """
        B, T, C = X.shape
        # FFT on T
        Xf = torch.fft.rfft(X.transpose(1, 2), dim=-1)  # (B,C,F)
        power = (Xf.real ** 2 + Xf.imag ** 2)           # (B,C,F)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(X.device)  # (F,)

        fmin, fmax = self.band
        mask = (freqs >= fmin) & (freqs <= fmax)        # (F,)
        band_power = power[:, :, mask].sum(dim=-1)      # (B,C)
        return band_power
