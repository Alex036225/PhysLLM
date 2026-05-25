from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================
# 工具：Hilbert 解析信号（FFT 版，强制 fp32，避开 ComplexHalf）
# =============================

def hilbert_1d(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, T)  (允许上游是 fp16/bf16/fp32)
    return: complex (N, T)  —— 返回复数仍在 fp32 复数域
    """
    # 关掉 autocast，强制 fp32 做 FFT，避免 ComplexHalf
    with torch.cuda.amp.autocast(enabled=False):
        x32 = x.float()
        N, T = x32.shape
        Xf = torch.fft.rfft(x32, n=T, dim=-1)            # complex64
        Nh = Xf.shape[-1]
        h = torch.zeros(Nh, device=x32.device, dtype=Xf.dtype)
        if T % 2 == 0:
            h[0] = 1.0
            h[1:-1] = 2.0
            h[-1] = 1.0
        else:
            h[0] = 1.0
            h[1:] = 2.0
        Xf_h = Xf * h
        z = torch.fft.irfft(Xf_h, n=T, dim=-1)           # real fp32
        # 返回复数（实部=原信号，虚部=希尔伯特）
        return torch.complex(x32, z)


# =============================
# 工具：相位展开
# =============================

def unwrap_phase(angle: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pi = math.pi
    two_pi = 2 * pi
    slice1 = [slice(None)] * angle.dim()
    slice2 = [slice(None)] * angle.dim()
    slice1[dim] = slice(1, None)
    slice2[dim] = slice(0, -1)
    dphi = angle[tuple(slice1)] - angle[tuple(slice2)]
    dphi_mod = (dphi + pi) % (two_pi) - pi
    fix = (dphi_mod == -pi) & (dphi > 0)
    dphi_mod = torch.where(fix, torch.full_like(dphi_mod, pi), dphi_mod)
    zeros_shape = list(angle.shape)
    zeros_shape[dim] = 1
    zeros = torch.zeros(zeros_shape, device=angle.device, dtype=angle.dtype)
    corr = dphi_mod - dphi
    corr_cs = torch.cumsum(torch.cat([zeros, corr], dim=dim), dim=dim)
    return angle + corr_cs


# =============================
# 工具：线性插值到统一相位网格（向量化 + 可选“最后完整周期”）
# =============================

def resample_to_phase_grid(
    x: torch.Tensor,
    phase: torch.Tensor,
    K: int,
    use_last_cycle: bool = True
) -> torch.Tensor:
    """
    向量化重采样到统一相位网格（无 Python for 循环）
    x:     (N, T, C)
    phase: (N, T) — 已展开相位
    ->     (N, K, C)
    """
    N, T, C = x.shape
    device = x.device
    two_pi = 2 * math.pi

    # 相位对齐到起点
    phase0 = phase - phase[:, :1]                       # (N,T)

    if use_last_cycle:
        # 取最后一个完整 2π 周期（不足则回退全序列）
        span = phase0[:, -1]                             # (N,)
        start_cycle = (torch.floor(span / two_pi) - 1.0).clamp(min=0) * two_pi  # (N,)
        end_cycle = start_cycle + two_pi                                                     # (N,)
        low = start_cycle[:, None]                                                            # (N,1)
        high = end_cycle[:, None]                                                             # (N,1)
        # 将相位夹到 [low, high]，相当于用边界延拓掩码外的点，保持向量化插值
        phase0 = phase0.clamp(min=low, max=high)                                             # (N,T)
        # 归一化到 [0,2π]
        ph_n = (phase0 - low) * (two_pi / (high - low + 1e-8))                                # (N,T)
    else:
        span = (phase0[:, -1:] + 1e-8).clamp_min(1e-6)                                       # (N,1)
        ph_n = phase0 * (two_pi / span)                                                       # (N,T)

    # 统一网格
    grid = torch.linspace(0, two_pi, K, device=device)  # (K,)
    le_mask = (ph_n[:, None, :] <= grid[None, :, None]) # (N,K,T)
    right_idx = le_mask.long().sum(dim=-1).clamp(min=1, max=T-1)  # (N,K)
    left_idx  = right_idx - 1                                     # (N,K)

    # 取出左右相位与特征，线性插值
    b_idx = torch.arange(N, device=device)[:, None].expand(N, K)  # (N,K)

    ph_left  = ph_n[b_idx, left_idx]           # (N,K)
    ph_right = ph_n[b_idx, right_idx]          # (N,K)

    x_left  = x[b_idx, left_idx]               # (N,K,C)
    x_right = x[b_idx, right_idx]              # (N,K,C)

    # 权重 w = (grid - ph_left) / (ph_right - ph_left)
    w = (grid[None, :] - ph_left) / (ph_right - ph_left + 1e-8)   # (N,K)
    w = w.unsqueeze(-1)                                           # (N,K,1)

    out = (1 - w) * x_left + w * x_right                          # (N,K,C)
    return out


# =============================
# 相位头：代理波 -> 相位 -> 重采样
# =============================

class PhaseHead(nn.Module):
    def __init__(self, in_channels: int, K: int = 96, use_last_cycle: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_channels, 1)
        self.K = K
        self.use_last_cycle = use_last_cycle

    @staticmethod
    def moving_avg(x: torch.Tensor, win: int = 15) -> torch.Tensor:
        if win <= 1:
            return x
        pad = win // 2
        xpad = F.pad(x, (pad, pad), mode="reflect")
        kernel = torch.ones(1, 1, win, device=x.device, dtype=x.dtype) / win
        y = F.conv1d(xpad.unsqueeze(1), kernel, padding=0).squeeze(1)
        return y

    def forward(self, f_ntc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        f_ntc: (N, T, C)
        return:
          f_phase_ntc: (N, K, C)
          phase_unwrapped: (N, T)
        """
        N, T, C = f_ntc.shape
        # 用 fp32 做代理波与滤波，稳定 Hilbert
        s = self.proj(f_ntc).squeeze(-1).float()        # (N, T) -> fp32
        s_hp = s - self.moving_avg(s, win=max(5, T//30))  # 稍更强一点的平滑
        s_hp = s_hp / (s_hp.std(dim=-1, keepdim=True) + 1e-6)
        z = hilbert_1d(s_hp)                             # complex64
        phase = unwrap_phase(torch.angle(z), dim=-1)     # fp32
        # 重采样仍可接住上游 dtype（f_ntc 可能是 fp16）
        f_phase_ntc = resample_to_phase_grid(
            f_ntc, phase, K=self.K, use_last_cycle=self.use_last_cycle
        )
        return f_phase_ntc, phase


# =============================
# 物理微分算子 L_tau
# =============================

class PhysioOperator(nn.Module):
    def __init__(self, K: int, learnable_tau: bool = True, n_domains: Optional[int] = None,
                 tau_init: float = 0.6, tau_bounds: Tuple[float, float]=(0.3, 0.9)):
        super().__init__()
        self.K = K
        self.tau_min, self.tau_max = tau_bounds
        if n_domains is None:
            self.tau = nn.Parameter(torch.tensor([tau_init], dtype=torch.float32), requires_grad=learnable_tau)
            self.per_domain = False
        else:
            self.tau = nn.Parameter(torch.full((n_domains,), tau_init, dtype=torch.float32), requires_grad=learnable_tau)
            self.per_domain = True

    def forward(self, x_phase_ntd: torch.Tensor, domain_ids_n: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        x_phase_ntd: (N, K, D)
        domain_ids_n: (N,) 若 per_domain
        return (N, K, D)
        """
        Bn, K, D = x_phase_ntd.shape
        dx = torch.roll(x_phase_ntd, shifts=-1, dims=1) - x_phase_ntd
        if self.per_domain:
            assert domain_ids_n is not None, "per-domain τ 需要提供 domain_ids"
            tau_b = self.tau[domain_ids_n].view(Bn, 1, 1)
        else:
            tau_b = self.tau.view(1, 1, 1)
        tau_b = torch.clamp(tau_b, self.tau_min, self.tau_max)
        return dx + (x_phase_ntd / tau_b)


# =============================
# Sinkhorn-OT（熵正则，对数域稳定版）
# =============================

@dataclass
class SinkhornConfig:
    eps: float = 0.15   # 更强的 OT（温度更低）
    iters: int = 20

def sinkhorn(a: torch.Tensor, b: torch.Tensor, C: torch.Tensor, cfg: SinkhornConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对数域 Sinkhorn（稳定）
    a, b: (m,)  —— 非负且和为 1
    C:    (m,m)
    返回：
      pi:  (m,m)
      u,v: (m,), (m,)
    """
    with torch.cuda.amp.autocast(enabled=False):  # 避免 fp16 下的数值问题
        log_a = torch.log(a + 1e-38)
        log_b = torch.log(b + 1e-38)
        logK = -C / cfg.eps

        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)

        for _ in range(cfg.iters):
            log_u = log_a - torch.logsumexp(logK + log_v[None, :], dim=1)
            log_v = log_b - torch.logsumexp(logK.t() + log_u[None, :], dim=1)

        log_pi = log_u[:, None] + logK + log_v[None, :]
        log_pi = torch.clamp(log_pi, min=-80.0, max=40.0)
        pi = torch.exp(log_pi)

        u = torch.exp(log_u)
        v = torch.exp(log_v)
        return pi, u, v


# =============================
# P_c：心动相关子空间（可学习正交）
# =============================

class CardioSubspace(nn.Module):
    def __init__(self, in_c: int, r: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_c, r) * (1.0 / math.sqrt(in_c)))

    def forward(self, x_lastc: torch.Tensor) -> torch.Tensor:
        return x_lastc @ self.W

    def orth_loss(self, eps: float = 1e-6) -> torch.Tensor:
        # 在 fp32 中计算，避免 AMP 下溢
        W = self.W.float()
        WT_W = W.t() @ W
        I = torch.eye(WT_W.shape[0], device=WT_W.device, dtype=WT_W.dtype)
        return F.mse_loss(WT_W, I)


# ============================================================
# 维度适配：任意 3~5D -> (B, T, *S, C)；再从 (B, K, *S, C) 还原
# ============================================================

def permute_to_BTSXC(x: torch.Tensor, time_dim: int, channel_dim: int):
    """
    把任意 3~5D 形状规范为 (B, T, *S, C)
    - 批维 B 固定是 dim 0
    - time_dim 指向时间维
    - channel_dim 指向通道维
    其它 0~2 个维度作为 *S（空间维）
    返回：
      x_perm: (B, T, *S, C)
      S_shape: tuple
      inv: 用于还原的元信息（包含 inv_perm、orig_shape 等）
    """
    assert 3 <= x.dim() <= 5, "Only support 3~5D input"
    D = x.dim()
    if time_dim < 0: time_dim += D
    if channel_dim < 0: channel_dim += D
    assert time_dim != 0 and channel_dim != 0, "batch 维必须是 0"
    assert time_dim != channel_dim, "time_dim 与 channel_dim 不能相同"

    spatial_dims = [d for d in range(1, D) if d not in (time_dim, channel_dim)]
    perm = [0, time_dim] + spatial_dims + [channel_dim]
    x_perm = x.permute(*perm).contiguous()

    # 逆置换
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i

    inv = {
        'orig_shape': tuple(x.shape),
        'perm': perm,
        'inv_perm': inv_perm,
        'orig_dim': D,
        'time_dim': time_dim,
        'channel_dim': channel_dim,
        'spatial_dims': spatial_dims
    }
    S_shape = tuple(x_perm.shape[2:-1])  # () or (S,) or (H,W)
    return x_perm, S_shape, inv


def invert_from_BKSXC(x_bksxc: torch.Tensor, inv: Dict) -> torch.Tensor:
    """
    把 (B, K, *S, C) 还原回**原维度顺序**（把原时间维长度替换为 K）
    """
    x_orig_order = x_bksxc.permute(*inv['inv_perm']).contiguous()
    return x_orig_order


def flatten_to_NTC(f_btcx: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, Tuple[int, ...]]]:
    """
    f_btcx: (B, T, *S, C)
    return: f_ntc: (N, T, C), meta=(B, S_shape)
    """
    assert 3 <= f_btcx.dim() <= 5, "Only support 3~5D"
    B = f_btcx.shape[0]
    T = f_btcx.shape[1]
    C = f_btcx.shape[-1]
    S_shape = f_btcx.shape[2:-1]
    if len(S_shape) == 0:
        f_ntc = f_btcx.reshape(B, T, C)
    else:
        S = 1
        for s in S_shape:
            S *= s
        f_ntc = f_btcx.reshape(B, T, S, C).permute(0, 2, 1, 3).reshape(B * S, T, C)
    return f_ntc, (B, S_shape)


def unflatten_from_NKC(x_nkc: torch.Tensor, meta: Tuple[int, Tuple[int, ...]]) -> torch.Tensor:
    """
    x_nkc: (N, K, C)
    meta: (B, S_shape)
    return: (B, K, *S, C)
    """
    B, S_shape = meta
    K = x_nkc.shape[1]
    C = x_nkc.shape[-1]
    if len(S_shape) == 0:
        return x_nkc.reshape(B, K, C)
    S = 1
    for s in S_shape:
        S *= s
    x_bskc = x_nkc.reshape(B, S, K, C)
    x_bkSc = x_bskc.permute(0, 2, 1, 3).contiguous()
    return x_bkSc.reshape(B, K, *S_shape, C)


def repeat_meta_vector(vec_b: torch.Tensor, S_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    vec_b: (B,) -> (B*S,)
    """
    if len(S_shape) == 0:
        return vec_b
    S = 1
    for s in S_shape:
        S *= s
    return vec_b.repeat_interleave(S)


# =============================
# POTA 主模块（3~5D 通用 + 可配置维度）
# =============================

class POTA(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 K: int = 96,
                 subspace_dim: int = 16,
                 n_domains: Optional[int] = None,
                 alpha: float = 0.5,
                 beta: float = 0.1,
                 gamma: float = 0.05,
                 lambda_ot: float = 0.8,          # 提升 OT 权重
                 lambda_orth: float = 1e-2,       # orth 力度更可见
                 sinkhorn_cfg: SinkhornConfig = SinkhornConfig(),
                 use_conditional_kernel: bool = True,   # 默认开启
                 kernel_sigma: float = 12.0,            # 放宽一点
                 pair_m: int = 24,
                 time_dim: int = 1,        # 可配置：时间维
                 channel_dim: int = -1,    # 可配置：通道维
                 use_last_cycle: bool = True):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ot = lambda_ot
        self.lambda_orth = lambda_orth
        self.use_conditional_kernel = use_conditional_kernel
        self.kernel_sigma = kernel_sigma
        self.pair_m = pair_m
        self.sinkhorn_cfg = sinkhorn_cfg
        self.time_dim = time_dim
        self.channel_dim = channel_dim

        self.phase_head = PhaseHead(feat_dim, K=K, use_last_cycle=use_last_cycle)
        self.Pc = CardioSubspace(feat_dim, subspace_dim)
        self.phys = PhysioOperator(K=K, learnable_tau=True, n_domains=n_domains)

    # ------- 向量化：相位中心（用于代价项） -------
    def _phase_center_batch(self, f_phase_nkr: torch.Tensor) -> torch.Tensor:
        """
        f_phase_nkr: (N, K, r)
        return c_n:  (N,) 每个样本的相位中心
        """
        N, K, _ = f_phase_nkr.shape
        theta = torch.linspace(0, 2*math.pi, K, device=f_phase_nkr.device)
        w = (f_phase_nkr ** 2).mean(dim=-1)      # (N,K)
        num = (w * theta[None, :]).sum(dim=-1)   # (N,)
        den = w.sum(dim=-1) + 1e-9
        return num / den

    def _pair_indices(self, idx_a: torch.Tensor, idx_b: torch.Tensor, m: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ma = min(m, idx_a.numel())
        mb = min(m, idx_b.numel())
        ia = idx_a[torch.randint(0, idx_a.numel(), (ma,), device=idx_a.device)]
        ib = idx_b[torch.randint(0, idx_b.numel(), (mb,), device=idx_b.device)]
        return ia, ib

    def _build_cost_matrix(self, Fd: torch.Tensor, FdL: torch.Tensor, y_d: torch.Tensor,
                           Fp: torch.Tensor, FpL: torch.Tensor, y_p: torch.Tensor) -> torch.Tensor:
        """
        Fd/Fp:  (m, K, r)  子空间表征
        FdL/FpL:(m, K, r)  物理算子作用后的表征
        y_d/y_p:(m,)       条件标签（如 HR）
        return: (m, m)     成本矩阵（已归一/裁剪，数值稳定；兼容老版 torch）
        """
        # 统一到 float32，避免 cdist 的 Half 限制
        Fd  = Fd.float()
        FdL = FdL.float()
        Fp  = Fp.float()
        FpL = FpL.float()
        y_d = y_d.float()
        y_p = y_p.float()

        m = Fd.shape[0]

        # L2 展平距离
        Fi = Fd.reshape(m, -1)
        Fj = Fp.reshape(m, -1)
        base = torch.cdist(Fi, Fj, p=2) ** 2

        Li = FdL.reshape(m, -1)
        Lj = FpL.reshape(m, -1)
        diff = torch.cdist(Li, Lj, p=2) ** 2

        C = base + self.alpha * diff

        # 相位中心项
        ci = self._phase_center_batch(Fd)  # (m,)
        cj = self._phase_center_batch(Fp)  # (m,)
        C = C + self.beta * (1.0 - torch.cos(ci.view(-1,1) - cj.view(1,-1)))

        # 条件核（如启用）
        if self.use_conditional_kernel:
            Dy = torch.cdist(y_d.view(-1,1), y_p.view(-1,1), p=2) ** 2
            Kc = torch.exp(-Dy / (2 * (self.kernel_sigma ** 2)))
            C = C * (1.0 / (Kc + 1e-6))

        # ---- 数值归一化与裁剪（兼容老版 torch） ----
        # 先把非有限数置零
        C = torch.where(torch.isfinite(C), C, torch.zeros_like(C))

        # 平移到非负
        C_min = C.min()
        C = C - C_min

        # 按中位数做尺度归一
        med = C.median().clamp(min=1e-6)
        C = C / med

        # 裁剪上界避免 exp(-C/eps) 下溢/上溢
        C_max = 80.0 / max(self.sinkhorn_cfg.eps, 1e-6)
        C = C.clamp(max=C_max)

        # 兜底一次
        C = torch.nan_to_num(C, nan=0.0, posinf=C_max, neginf=0.0)
        return C

    # ------- 对外：仅相位对齐（保持原布局） -------
    def phase_align(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: 任意 3~5D，指定 time_dim / channel_dim
        return:
          out: 与输入同维度顺序，但时间维被 K 替换
          phase_unwrap: (N, T)（展平样本的相位）
        """
        x_btsxc, S_shape, inv = permute_to_BTSXC(x, self.time_dim, self.channel_dim)
        f_ntc, meta = flatten_to_NTC(x_btsxc)
        f_phase_ntc, phase_unwrap = self.phase_head(f_ntc)    # (N,K,C)
        f_bksxc = unflatten_from_NKC(f_phase_ntc, meta)       # (B,K,*S,C)
        out = invert_from_BKSXC(f_bksxc, inv)                 # 回到原维度顺序（时间维长度改为 K）
        return out, phase_unwrap

    # ------- 训练前向：相位 + 子空间 + 物理正则 + 条件 OT -------
    def forward(self, x: torch.Tensor, domain_ids_b: torch.Tensor, y_b: Optional[torch.Tensor]=None):
        """
        x: 任意 3~5D，指定 time_dim / channel_dim
        domain_ids_b: (B,)
        y_b: (B,) 或 None
        return:
          out: 与输入同维度顺序，但时间维被 K 替换
          losses: dict 包含 ot_loss 与 orth_loss
        """
        device = x.device
        # 1) 任意布局 -> (B,T,*S,C)
        x_btsxc, S_shape, inv = permute_to_BTSXC(x, self.time_dim, self.channel_dim)
        # 2) (B,T,*S,C) -> (N,T,C)
        f_ntc, meta = flatten_to_NTC(x_btsxc)
        B, _Sshape = meta
        # 3) 广播域与标签到 (N,)
        domain_ids_n = repeat_meta_vector(domain_ids_b.to(device), _Sshape)
        y_n = repeat_meta_vector(y_b.to(device), _Sshape) if (y_b is not None) else None
        # 4) 相位对齐 + 子空间 + 物理算子
        f_phase_ntc, phase_unwrap = self.phase_head(f_ntc)    # (N,K,C)
        f_pc_ntd = self.Pc(f_phase_ntc)                       # (N,K,r)
        Lf_ntd = self.phys(f_pc_ntd, domain_ids_n)            # (N,K,r)

        # 5) 跨域 OT 正则
        unique_domains = torch.unique(domain_ids_n)
        ot_total = f_ntc.new_zeros(())
        n_pairs = 0

        pairs = [(di.item(), dj.item()) for i, di in enumerate(unique_domains) for dj in unique_domains[i+1:]]
        # 若要进一步提速：训练时随机只取一对域
        # if self.training and len(pairs) > 0:
        #     pairs = [pairs[torch.randint(len(pairs), (1,), device=device).item()]]

        # 在 OT 区域关闭 AMP，更稳
        with torch.cuda.amp.autocast(enabled=False):
            for (di, dj) in pairs:
                di = torch.tensor(di, device=device)
                dj = torch.tensor(dj, device=device)
                idx_i = torch.nonzero(domain_ids_n == di, as_tuple=False).squeeze(-1)
                idx_j = torch.nonzero(domain_ids_n == dj, as_tuple=False).squeeze(-1)
                if idx_i.numel() == 0 or idx_j.numel() == 0:
                    continue
                ia, ib = self._pair_indices(idx_i, idx_j, self.pair_m)
                Fi, Fj = f_pc_ntd[ia], f_pc_ntd[ib]
                Li, Lj = Lf_ntd[ia], Lf_ntd[ib]
                yi = y_n[ia] if (y_n is not None) else torch.zeros(len(ia), device=device, dtype=torch.float32)
                yj = y_n[ib] if (y_n is not None) else torch.zeros(len(ib), device=device, dtype=torch.float32)
                m = min(Fi.shape[0], Fj.shape[0])
                if m <= 1:
                    continue
                Fi, Fj, Li, Lj, yi, yj = Fi[:m], Fj[:m], Li[:m], Lj[:m], yi[:m], yj[:m]

                # 统一到 float32（即便上游可能是 half）
                Fi = Fi.float(); Fj = Fj.float()
                Li = Li.float(); Lj = Lj.float()
                yi = yi.float(); yj = yj.float()

                Cmat = self._build_cost_matrix(Fi, Li, yi, Fj, Lj, yj)
                a = torch.full((m,), 1.0/m, device=device, dtype=torch.float32)
                b = torch.full((m,), 1.0/m, device=device, dtype=torch.float32)
                pi, _, _ = sinkhorn(a, b, Cmat, self.sinkhorn_cfg)
                ot_total = ot_total + (pi * Cmat).sum()
                n_pairs += 1

        if n_pairs > 0:
            ot_total = ot_total / n_pairs

        # orth 用 fp32 计算，避免 AMP 下溢
        with torch.cuda.amp.autocast(enabled=False):
            orth = self.Pc.orth_loss()

        losses = {
            'ot_loss': self.lambda_ot * ot_total,
            'orth_loss': self.lambda_orth * orth,
        }

        # 6) 还原到原布局（时间维变 K）
        f_bksxc = unflatten_from_NKC(f_phase_ntc, meta)   # (B,K,*S,C)
        out = invert_from_BKSXC(f_bksxc, inv)
        return out, losses


# =============================
# 用法示例（可按需改你的训练脚本）
# =============================

if __name__ == "__main__":
    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ---- 示例1：经典 3D (B, T, C) ----
    B, T, C = 8, 128, 64
    feats3 = torch.randn(B, T, C, device=device)
    domains = torch.randint(0, 3, (B,), device=device)
    hr = torch.randint(55, 110, (B,), dtype=torch.float32, device=device)

    pota1 = POTA(
        feat_dim=C, K=64, subspace_dim=12, n_domains=3, pair_m=24,
        # 更强 OT + 条件核
        sinkhorn_cfg=SinkhornConfig(eps=0.15, iters=20),
        use_conditional_kernel=True,
        kernel_sigma=12.0,
        lambda_ot=0.8,
        lambda_orth=1e-2,
        time_dim=1, channel_dim=-1,
        use_last_cycle=True
    ).to(device)

    if use_cuda:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            f3_aligned, losses3 = pota1(feats3, domains, y_b=hr)
    else:
        f3_aligned, losses3 = pota1(feats3, domains, y_b=hr)

    print("3D in :", tuple(feats3.shape), " -> out:", tuple(f3_aligned.shape))
    print({k: float(v.detach().cpu()) for k, v in losses3.items()})

    # ---- 示例2：5D NCDHW (B, C, D, H, W) ----
    B, C, D, H, W = 4, 64, 128, 4, 4
    feats5 = torch.randn(B, C, D, H, W, device=device)
    domains5 = torch.randint(0, 3, (B,), device=device)
    hr5 = torch.randint(55, 110, (B,), dtype=torch.float32, device=device)

    pota2 = POTA(
        feat_dim=C, K=64, subspace_dim=12, n_domains=3, pair_m=24,
        sinkhorn_cfg=SinkhornConfig(eps=0.15, iters=20),
        use_conditional_kernel=True,
        kernel_sigma=12.0,
        lambda_ot=0.8,
        lambda_orth=1e-2,
        time_dim=2,  # N C D H W 中 D 是时间 => 2
        channel_dim=1,  # C 在第二维 => 1
        use_last_cycle=True
    ).to(device)

    if use_cuda:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            f5_aligned, losses5 = pota2(feats5, domains5, y_b=hr5)
    else:
        f5_aligned, losses5 = pota2(feats5, domains5, y_b=hr5)

    print("5D in :", tuple(feats5.shape), " -> out:", tuple(f5_aligned.shape))  # 期望 (B, C, K, H, W)
    print({k: float(v.detach().cpu()) for k, v in losses5.items()})
