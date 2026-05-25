# -*- coding: utf-8 -*-
"""
wrapper.py
---------------------------------------
将 transportnet.py 中的组件组装为一个可插拔的“rPPG 域泛化对齐器”。
新增：ShapeAdapter，自动兼容 3D/4D/5D 输入与原样式还原。

支持的常见布局（B=0 位）：
- 3D: (B,T,C), (B,C,T)
- 4D: (B,C,H,W)（无时间，视为 T=1）, (B,T,C,S), (B,C,T,S), (B,T,H,W)
- 5D: (B,C,T,H,W), (B,T,C,H,W), (B,T,H,W,C)

策略：
- 统一将输入规范化为 (B,T,C) 进行对齐（空间维做全局平均池化）。
- 输出再按原布局还原；若原有空间维，使用“广播回原尺寸”（每帧同一通道填充）。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_methods.model.OTTrack.transportnet import (
    TemporalAligner, PrototypeBank, ConditionalCost,
    CostConfig, Sinkhorn, SinkhornConfig, SinkhornDivergence,
    BandEnergy
)


# =========================
# 0) 形状自适配器
# =========================

class ShapeAdapter:
    """
    将任意 3D/4D/5D 的张量规范化为 (B,T,C)，并能将 (B,T,C) 结果还原为原布局。
    - 若没有时间维（如 (B,C,H,W)），视为 T=1。
    - 空间维一律用全局平均池化归一；还原时采用广播填回。
    - 如布局不在内置列表，会按默认优先级猜测：3D -> (B,T,C)；4D -> (B,T,C,S)；5D -> (B,T,C,H,W)。
    你也可以显式提供 time_dim / channel_dim 来覆盖推断。
    """

    def __init__(self, time_dim: Optional[int] = None, channel_dim: Optional[int] = None, pool: str = "mean"):
        self.time_dim = time_dim
        self.channel_dim = channel_dim
        self.pool = pool  # 目前只实现 mean
        self._ctx = None

    def _global_pool(self, x: torch.Tensor, spatial_dims: List[int]) -> torch.Tensor:
        y = x
        # 按从大到小顺序做，以避免维度变化导致的索引偏移
        for d in sorted(spatial_dims, reverse=True):
            y = y.mean(dim=d, keepdim=False)
        return y

    def _broadcast_back(self, y_btc: torch.Tensor, ctx: dict) -> torch.Tensor:
        """
        将 (B,T,C) 结果按 ctx 指定的原布局进行还原。
        对于被池化掉的空间维，广播回原尺寸。
        """
        B, T, C = y_btc.shape
        layout = ctx["layout"]        # "BTC", "BCT", "BCHW", "BCTHW", "BTCHW", "BTHWC", "BTCS", "BCTS", "BTHW"
        orig_shape = ctx["orig_shape"]  # tuple
        spatial_sizes = ctx["spatial_sizes"]  # list of ints
        has_time = ctx["has_time"]

        # 先得到 (B,T,C)
        out = y_btc

        # 如原来没有时间维，则压回 (B,C)
        if not has_time:
            # T=1
            out = out.mean(dim=1)  # (B,C)

        # 广播空间维
        if len(spatial_sizes) > 0:
            # 生成 (B, T or 1, C, *spatial)
            if has_time:
                base = out.view(B, T, C, *([1] * len(spatial_sizes)))
                expand_sizes = (B, T, C, *spatial_sizes)
                out = base.expand(expand_sizes)
            else:
                base = out.view(B, C, *([1] * len(spatial_sizes)))
                expand_sizes = (B, C, *spatial_sizes)
                out = base.expand(expand_sizes)

        # 根据原布局重排维度
        if layout == "BTC":
            return out  # (B,T,C)
        elif layout == "BCT":
            return out.permute(0, 2, 1)  # (B,C,T)
        elif layout == "BCHW":
            return out  # 已是 (B,C,H,W)（无 T）
        elif layout == "BCTHW":
            # 当前 out 是 (B,T,C,H,W)
            return out.permute(0, 2, 1, 3, 4)  # (B,C,T,H,W)
        elif layout == "BTCHW":
            return out  # (B,T,C,H,W)
        elif layout == "BTHWC":
            return out.permute(0, 1, 3, 4, 2)  # (B,T,H,W,C)
        elif layout == "BTCS":
            return out  # (B,T,C,S)
        elif layout == "BCTS":
            return out.permute(0, 2, 1, 3)  # (B,C,T,S)
        elif layout == "BTHW":
            return out  # (B,T,H,W)（无 C，表示 C=1 情况不处理，这里保持 (B,T,H,W)）
        else:
            # 默认返回 (B,T,C) 或 (B,C)（没有空间维时）
            return out

    def to_BTC(self, x: torch.Tensor) -> torch.Tensor:
        """
        解析 x 的布局，返回 (B,T,C) 和上下文 ctx（保存用以还原）。
        3D/4D/5D 均可。
        """
        assert x.dim() in (3, 4, 5), f"Only support 3D/4D/5D, got {x.dim()}D"
        B = x.shape[0]
        D = x.dim()
        shape = tuple(x.shape)

        # 用户显式指定时间/通道维时，优先使用
        tdim = self.time_dim
        cdim = self.channel_dim

        has_time = True
        spatial_dims: List[int] = []

        def pool_spatial(y: torch.Tensor, dims: List[int]) -> torch.Tensor:
            return self._global_pool(y, dims)

        layout = None

        if D == 3:
            # 常见：(B,T,C) 或 (B,C,T)
            if tdim is None and cdim is None:
                # 默认优先 (B,T,C) 再 (B,C,T)
                # 经验：若第二维较大且第三维较小，可视作 T,C；否则反之
                T2, C2 = x.shape[1], x.shape[2]
                if T2 >= C2:
                    tdim, cdim = 1, 2
                    layout = "BTC"
                else:
                    tdim, cdim = 2, 1
                    layout = "BCT"
            if tdim is None: tdim = 1 if cdim == 2 else 2
            if cdim is None: cdim = 2 if tdim == 1 else 1

            # 转成 (B,T,C)
            if (tdim, cdim) == (1, 2):
                y = x
                layout = layout or "BTC"
            elif (tdim, cdim) == (2, 1):
                y = x.permute(0, 2, 1)
                layout = layout or "BCT"
            else:
                y = x.permute(0, tdim, cdim)
                layout = "BTC"
            spatial_sizes = []
            ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=spatial_sizes, has_time=True)
            self._ctx = ctx
            return y

        if D == 4:
            # 候选： (B,C,H,W)（无 T） / (B,T,C,S) / (B,C,T,S) / (B,T,H,W)
            if tdim is not None and cdim is not None:
                # 明确指定
                perm = [0, tdim, cdim] + [d for d in range(1, 4) if d not in (tdim, cdim)]
                y = x.permute(*perm)  # (B,T,C,Spatial...)
                spatial_dims = list(range(3, y.dim()))
                spatial_sizes = [y.shape[d] for d in spatial_dims]
                y = pool_spatial(y, spatial_dims)  # -> (B,T,C)
                layout = "BTCS" if len(spatial_sizes) == 1 else "BTHW" if len(spatial_sizes) == 2 and cdim not in (1,2) else "BTCS"
                ctx = dict(layout="BTCS" if len(spatial_sizes)==1 else "BTHW",
                           orig_shape=shape, spatial_sizes=spatial_sizes, has_time=True)
                self._ctx = ctx
                return y

            # 自动推断
            # 1) (B,C,H,W) 无时间
            if self._looks_like_bchw(shape):
                has_time = False
                cdim = 1
                # 池化空间 -> (B,C)
                y = pool_spatial(x, [3, 2])  # W,H
                # 塞一个 T=1: (B,1,C)
                y = y.unsqueeze(1)
                spatial_sizes = [shape[2], shape[3]]
                layout = "BCHW"
                ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=spatial_sizes, has_time=False)
                self._ctx = ctx
                return y  # (B,1,C)

            # 2) (B,T,C,S)
            # 3) (B,C,T,S)
            # 4) (B,T,H,W)
            # 优先假设第二维是 T
            # 如果第三维很小（<= 256）更像 C，否则更像 H/W，统一用均值池化处理
            # 先尝试 (B,T,C,S)
            T = shape[1]
            # 假定 (B,T,*,*)
            # 将第 2,3 维之一作为 C（若其中一个较小）
            if shape[2] <= 1024:
                # 作为 C；第3维 S 做池化
                tdim, cdim = 1, 2
                layout = "BTCS"
                y = x
                y = pool_spatial(y, [3])  # 池化 S
                y = y  # (B,T,C)
                spatial_sizes = [shape[3]]
                ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=spatial_sizes, has_time=True)
                self._ctx = ctx
                return y
            else:
                # 作为 (B,T,H,W)（无 C）：我们把 H,W 池化到 1，并设置 C=原通道=1（退化）
                layout = "BTHW"
                y = pool_spatial(x, [3, 2])  # (B,T)
                y = y.unsqueeze(-1)          # (B,T,1) 当作 C=1
                spatial_sizes = [shape[2], shape[3]]
                ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=spatial_sizes, has_time=True)
                self._ctx = ctx
                return y

        if D == 5:
            # 候选： (B,C,T,H,W) / (B,T,C,H,W) / (B,T,H,W,C)
            if tdim is not None and cdim is not None:
                # 显式指定
                perm = [0, tdim, cdim] + [d for d in range(1, 5) if d not in (tdim, cdim)]
                y = x.permute(*perm)  # (B,T,C,*,*)
                spatial_dims = list(range(3, y.dim()))
                spatial_sizes = [y.shape[d] for d in spatial_dims]
                y = pool_spatial(y, spatial_dims)  # -> (B,T,C)
                # 记录布局（常见三种按原顺序映射）
                if (tdim, cdim) == (1, 2):
                    layout = "BTCHW"
                elif (tdim, cdim) == (2, 1):
                    layout = "BCTHW"
                elif (tdim, cdim) == (1, 4):
                    layout = "BTHWC"
                else:
                    layout = "BTCHW"
                ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=spatial_sizes, has_time=True)
                self._ctx = ctx
                return y

            # 自动推断常见三种
            # (B,C,T,H,W)
            if self._match_dims(shape, order=("B", "C", "T", "H", "W")):
                layout = "BCTHW"
                # perm -> (B,T,C,H,W)
                y = x.permute(0, 2, 1, 3, 4)
                spatial_sizes = [shape[3], shape[4]]
                y = pool_spatial(y, [4, 3])  # -> (B,T,C)
                ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=spatial_sizes, has_time=True)
                self._ctx = ctx
                return y

            # (B,T,C,H,W)
            if self._match_dims(shape, order=("B", "T", "C", "H", "W")):
                layout = "BTCHW"
                spatial_sizes = [shape[3], shape[4]]
                y = pool_spatial(x, [4, 3])  # -> (B,T,C)
                ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=spatial_sizes, has_time=True)
                self._ctx = ctx
                return y

            # (B,T,H,W,C)
            layout = "BTHWC"
            # perm -> (B,T,C,H,W)
            y = x.permute(0, 1, 4, 2, 3)
            spatial_sizes = [shape[2], shape[3]]
            y = pool_spatial(y, [4, 3])  # -> (B,T,C)
            ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=spatial_sizes, has_time=True)
            self._ctx = ctx
            return y

        # fallback：粗暴认为第二维是 T，最后一维是 C（或无时间时 T=1）
        if D == 3:
            y = x
            layout = "BTC"
            ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=[], has_time=True)
            self._ctx = ctx
            return y
        elif D == 4:
            # 假设 (B,T,C,S)
            y = x.mean(dim=3, keepdim=False)
            layout = "BTCS"
            ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=[shape[3]], has_time=True)
            self._ctx = ctx
            return y
        else:
            # 假设 (B,T,C,H,W)
            y = x.mean(dim=[3, 4], keepdim=False)
            layout = "BTCHW"
            ctx = dict(layout=layout, orig_shape=shape, spatial_sizes=[shape[3], shape[4]], has_time=True)
            self._ctx = ctx
            return y

    @staticmethod
    def _looks_like_bchw(shape: Tuple[int, ...]) -> bool:
        return (len(shape) == 4) and (shape[1] <= 512) and (shape[2] >= 4) and (shape[3] >= 4)

    @staticmethod
    def _match_dims(shape: Tuple[int, ...], order: Tuple[str, ...]) -> bool:
        # 这里只用于匹配常见 5D 排列的“标签顺序”，简化处理：默认总是 True（因为没有显式标签）
        # 实际推断主要靠上面 if 分支的优先级。
        return True

    def restore_like_input(self, y_btc: torch.Tensor) -> torch.Tensor:
        assert self._ctx is not None, "No context to restore. Call to_BTC() first."
        return self._broadcast_back(y_btc, self._ctx)


# =========================
# 1) 配置
# =========================

@dataclass
class OTAlignConfig:
    feat_dim: int
    num_prototypes: int = 64
    align_layers: int = 3
    align_kernel: int = 3
    use_attention: bool = False
    align_dropout: float = 0.1

    # 条件代价
    lambda_hr: float = 1.0
    sigma_hr: float = 10.0
    learnable_w: bool = True

    # Sinkhorn
    sink_epsilon: float = 0.08
    sink_iters: int = 50
    sink_debiased: bool = True

    # 正则
    lambda_band: float = 0.1
    band_fps: float = 30.0
    band_range: Tuple[float, float] = (0.7, 3.0)

    lambda_identity: float = 0.05
    lambda_src_consistency: float = 0.05

    # HR Head（在 hr_pred=None 时启用）
    use_internal_hr_head: bool = True

    # 形状自适配（可选手动指定）
    time_dim: Optional[int] = None
    channel_dim: Optional[int] = None


# =========================
# 2) 模块
# =========================

class HRHead(nn.Module):
    """一个极轻量的 HR 估计头，避免与主回归头强耦合。"""
    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, 1)
        )

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        # 简单地对 T 做平均再回归
        pooled = x_btc.mean(dim=1)      # (B,C)
        hr = self.net(pooled).squeeze(-1)  # (B,)
        # 把范围大致映射到 40~180
        return 40.0 + 140.0 * torch.sigmoid(hr)


class OTAlignWrapper(nn.Module):
    """
    端到端：时序保形对齐 + 条件OT + 多源重心 + 正则。
    现在支持 3D/4D/5D 输入，自动规范为 (B,T,C)，输出再还原为原布局。
    """
    def __init__(self, cfg: OTAlignConfig):
        super().__init__()
        self.cfg = cfg

        # 时序对齐器
        self.aligner = TemporalAligner(
            feat_dim=cfg.feat_dim,
            layers=cfg.align_layers,
            kernel_size=cfg.align_kernel,
            use_attention=cfg.use_attention,
            num_heads=4,
            dropout=cfg.align_dropout
        )

        # 原型
        self.bank = PrototypeBank(cfg.feat_dim, cfg.num_prototypes)

        # 代价
        self.cost = ConditionalCost(
            feat_dim=cfg.feat_dim,
            cfg=CostConfig(lambda_hr=cfg.lambda_hr, sigma_hr=cfg.sigma_hr, use_learnable_w=cfg.learnable_w)
        )

        # Sinkhorn
        self.sinkhorn = Sinkhorn(SinkhornConfig(
            epsilon=cfg.sink_epsilon,
            n_iters=cfg.sink_iters,
            debiased=cfg.sink_debiased,
            stable=True
        ))
        self.sdiv = SinkhornDivergence(self.sinkhorn)

        # 正则
        self.band_energy = BandEnergy(fps=cfg.band_fps, band=cfg.band_range)

        # 内置 HR 估计（可选）
        self.hr_head = HRHead(cfg.feat_dim) if cfg.use_internal_hr_head else None

        # 形状适配器
        self.adapter = ShapeAdapter(time_dim=cfg.time_dim, channel_dim=cfg.channel_dim)

    def _compute_ot(self, x_src_btc: torch.Tensor, hr_x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        对齐到原型重心，返回：
          - x_align_btc: (B,T,C)
          - diag: 诊断字典（含 W_xy 等）
          - ot_loss: 标量
          - P_xy: (B,T,K) 运输计划（用于一致性正则）
        """
        P, hr_p = self.bank()                         # (K,C), (K,)
        C_xy = self.cost(x_src_btc, P, hr_x, hr_p)    # (B,T,K)
        # 均匀权重
        B, T, _ = x_src_btc.shape
        a = torch.full((B, T), 1.0 / T, device=x_src_btc.device, dtype=x_src_btc.dtype)
        b = torch.full((P.shape[0],), 1.0 / P.shape[0], device=x_src_btc.device, dtype=x_src_btc.dtype)

        # Sinkhorn divergence / cost
        s_eps, diag = self.sdiv(
            C_xy=C_xy, X=x_src_btc, hr_x=hr_x, Y=P, hr_y=hr_p, w=self.cost.w, a=a, b=b
        )

        # 获得运输计划
        P_xy, _ = self.sinkhorn(C_xy, a=a, b=b)   # (B,T,K)

        # 映射到重心（barycentric projection）
        x_align_btc = torch.einsum('btk,kc->btc', P_xy, P)   # (B,T,C)
        return x_align_btc, diag, s_eps, P_xy

    @staticmethod
    def _identity_reg(x_src_btc: torch.Tensor, x_align_btc: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_align_btc, x_src_btc)

    def _band_reg(self, x_src_btc: torch.Tensor, x_align_btc: torch.Tensor) -> torch.Tensor:
        be_src = self.band_energy(x_src_btc)     # (B,C)
        be_aln = self.band_energy(x_align_btc)   # (B,C)
        return F.l1_loss(be_aln, be_src)

    @staticmethod
    def _source_consistency(P_xy: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        让不同源域的“原型分配直方图”更一致：
        - 对每个样本把 (T,K) 的运输计划在 T 上平均 -> (K,)
        - 按域取平均，最后最小化各域分布之间的方差
        """
        B, T, K = P_xy.shape
        with torch.no_grad():
            hist = P_xy.mean(dim=1)                      # (B,K)
            domains = torch.unique(domain_ids)
        dists = []
        for d in domains:
            mask = (domain_ids == d).float().view(-1, 1) # (B,1)
            m = (hist * mask).sum(dim=0) / (mask.sum() + 1e-8)  # (K,)
            dists.append(m)
        if len(dists) <= 1:
            return P_xy.new_zeros(())
        mats = torch.stack(dists, dim=0)                 # (D,K)
        return mats.var(dim=0).mean()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor,
                hr_pred: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: 任意 (B,*,*,...)，支持 3D/4D/5D 常见布局
        domain_ids: (B,)
        hr_pred: (B,) or None  — 连续标签（若 None，则内部估计）

        返回：
          x_aligned_same_layout: 与 x 相同布局（空间维按广播还原）
          losses: dict
        """
        # 0) 规范为 (B,T,C)
        x_btc = self.adapter.to_BTC(x)  # (B,T,C)

        # 1) 时序保形对齐器
        x_src_btc = self.aligner(x_btc)  # (B,T,C)

        # 2) HR 条件：外部给定或内部估计
        if hr_pred is None:
            assert self.hr_head is not None, "hr_pred is None but internal HR head is disabled."
            with torch.no_grad():
                hr_x = self.hr_head(x_src_btc).detach()   # (B,)
        else:
            hr_x = hr_pred

        # 3) 条件 OT 到重心
        x_align_btc, diag, ot_loss, P_xy = self._compute_ot(x_src_btc, hr_x)  # (B,T,C)

        # 4) 正则项（在 BTC 空间中计算）
        loss_identity = self._identity_reg(x_btc, x_align_btc) * self.cfg.lambda_identity
        loss_band = self._band_reg(x_btc, x_align_btc) * self.cfg.lambda_band
        loss_src = self._source_consistency(P_xy, domain_ids) * self.cfg.lambda_src_consistency

        total = ot_loss + loss_identity + loss_band + loss_src

        # 5) 还原为原布局（对有空间维的情况用广播还原）
        x_aligned_same_layout = self.adapter.restore_like_input(x_align_btc)

        losses = {
            "total": total,
            "ot": ot_loss.detach(),
            "identity": loss_identity.detach(),
            "band": loss_band.detach(),
            "src_consistency": loss_src.detach(),
            "W_xy": diag.get("W_xy", torch.tensor(0.)).detach(),
            "W_xx": diag.get("W_xx", torch.tensor(0.)).detach(),
            "W_yy": diag.get("W_yy", torch.tensor(0.)).detach(),
        }
        return x_aligned_same_layout, losses


# =========================
# 3) 简易用法示例（可注释）
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, H, W = 4, 128, 64, 16, 16
    n_domains = 3

    # 3D: (B,T,C)
    x3 = torch.randn(B, T, C)
    # 4D: (B,C,H,W)
    x4 = torch.randn(B, C, H, W)
    # 5D: (B,C,T,H,W)
    x5 = torch.randn(B, C, T, H, W)

    domain_ids = torch.randint(0, n_domains, (B,))
    hr_soft = 60 + 40 * torch.rand(B)  # 60~100

    cfg = OTAlignConfig(
        feat_dim=C,
        num_prototypes=64,
        align_layers=3,
        lambda_hr=1.0,
        sigma_hr=10.0,
        learnable_w=True,
        sink_epsilon=0.08,
        sink_iters=50,
        sink_debiased=True,
        lambda_band=0.1,
        band_fps=30.0,
        band_range=(0.7, 3.0),
        lambda_identity=0.05,
        lambda_src_consistency=0.05,
        use_internal_hr_head=True,
        time_dim=None, channel_dim=None  # 如需强指定可设置
    )
    model = OTAlignWrapper(cfg)

    for name, xin in [("3D (B,T,C)", x3), ("4D (B,C,H,W)", x4), ("5D (B,C,T,H,W)", x5)]:
        x_aligned, losses = model(xin, domain_ids, hr_pred=hr_soft)
        print(name, "-> out shape:", tuple(x_aligned.shape),
              "| losses:", {k: float(v) for k, v in losses.items() if k in ["total","ot","identity","band","src_consistency"]})
