"""This file is a combination of Physformer.py and transformer_layer.py
   in the official PhysFormer implementation here:
   https://github.com/ZitongYu/PhysFormer

   model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

import numpy as np
from typing import Optional, Dict, Tuple
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import math

try:
    from neural_methods.model.OTTrack.wrapper import OTAlignWrapper, OTAlignConfig
except Exception as exc:
    raise ImportError(
        "PhysFormer requires neural_methods.model.OTTrack.wrapper in the public repo."
    ) from exc

def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape
            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff
            else:
                return out_normal


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, gra_sharp):    # [B, 4*4*40, 128]
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / gra_sharp
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores


class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):    # [B, 4*4*40, 128]
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        x = self.fc1(x)		              # x [B, ff_dim, 40, 4, 4]
        x = self.STConv(x)		          # x [B, ff_dim, 40, 4, 4]
        x = self.fc2(x)		              # x [B, dim, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        return x

class Block_ST_TDC_gra_sharp(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score

class Transformer_ST_TDC_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        Score = None
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score


# ===============================
# stem_3DCNN + ST-ViT + OT 对齐
# ===============================
class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):
    def __init__(
        self,
        name: Optional[str] = None,
        pretrained: bool = False,
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        in_channels: int = 3,
        frame: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None,
        # 新增：是否启用 OT 对齐模块
        enable_ot: bool = True,
        # 可选：自定义 OT 配置
        ot_cfg: Optional[OTAlignConfig] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.frame = frame
        self.dim = dim
        self.enable_ot = enable_ot

        # Image and patch sizes
        t, h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # ft = 4 ==> 160/4=40
        gt, gh, gw = t//ft, h // fh, w // fw
        seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

        # Transformer (分 3 组)
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)

        # 3D stem
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(dim//2),
            nn.ELU(),
        )

        # 最终回归头（保持输入为 (B, C=dim//2, T)）
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1, stride=1, padding=0)

        # ============ 新增：OT 对齐模块 ============
        if self.enable_ot:
            if ot_cfg is None:
                # 以上采样后通道数 dim//2 作为对齐特征维度
                ot_cfg = OTAlignConfig(
                    feat_dim=dim // 2,
                    # 其他超参采用 wrapper.py 的默认值，你也可以在这里按需覆写
                    # 例如：num_prototypes=64, sink_iters=50, lambda_band=0.1, 等
                )
            self.ot_align = OTAlignWrapper(ot_cfg)
        else:
            self.ot_align = None
        # ========================================

        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)

    def forward(
        self,
        x: Tensor,
        gra_sharp: Tensor,
        domain_ids: Optional[Tensor] = None,
        hr_pred: Optional[Tensor] = None,
        return_align_losses: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:

        # x: (B, C, T, H, W)
        b, c, t, fh, fw = x.shape

        # stem + patching + transformer
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)  # [B, dim, T, H', W']

        x = self.patch_embedding(x)                 # [B, dim, T/4, 4, 4]
        x = x.flatten(2).transpose(1, 2)            # [B, (T/4)*4*4, dim]

        Trans_features, Score1 = self.transformer1(x, gra_sharp)
        Trans_features2, Score2 = self.transformer2(Trans_features, gra_sharp)
        Trans_features3, Score3 = self.transformer3(Trans_features2, gra_sharp)

        # 还原为 (B, dim, T/4, 4, 4) 并沿时间上采样回到 T
        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t//4, 4, 4)
        features_last = self.upsample(features_last)     # (B, dim, T/2, 4, 4)
        features_last = self.upsample2(features_last)    # (B, dim//2, T, 4, 4)
        features_last = torch.mean(features_last, 3)     # -> (B, dim//2, T, 4)
        features_last = torch.mean(features_last, 3)     # -> (B, dim//2, T)   (C,T)

        align_losses = None
        if self.enable_ot and self.ot_align is not None:
            # 期望输入 (B,C,T)。OT 对齐模块内部会用 ShapeAdapter 统一到 (B,T,C)，对齐后再还原回 (B,C,T)
            if domain_ids is None:
                domain_ids = torch.zeros(features_last.size(0), dtype=torch.long, device=features_last.device)
            x_aligned, losses = self.ot_align(features_last, domain_ids=domain_ids, hr_pred=hr_pred)
            features_last = x_aligned  # (B, C, T)
            if return_align_losses:
                align_losses = losses

        # 最终 rPPG 回归
        rPPG = self.ConvBlockLast(features_last)   # (B, 1, T)
        rPPG = rPPG.squeeze(1)                     # (B, T)

        # 返回：rPPG 以及三个层的注意力分数（保持原接口）
        # 若需要 loss，设置 return_align_losses=True
        return rPPG, Score1, Score2, Score3, align_losses


class PhysFormer_encoder(nn.Module):
    def __init__(
        self,
        name: Optional[str] = None,
        pretrained: bool = False,
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        in_channels: int = 3,
        frame: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None,
        # 新增：可作为纯特征对齐编码器用
        enable_ot: bool = True,
        ot_cfg: Optional[OTAlignConfig] = None,
    ):
        super().__init__()

        self.image_size = image_size
        self.frame = frame
        self.dim = dim
        self.enable_ot = enable_ot

        t, h, w = as_tuple(image_size)
        ft, fh, fw = as_tuple(patches)
        gt, gh, gw = t//ft, h // fh, w // fw
        seq_len = gh * gw * gt

        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)

        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(dim//2),
            nn.ELU(),
        )

        # ============ 新增：OT 对齐模块（编码器阶段输出对齐后的时序特征） ============
        if self.enable_ot:
            if ot_cfg is None:
                ot_cfg = OTAlignConfig(feat_dim=dim // 2)
            self.ot_align = OTAlignWrapper(ot_cfg)
        else:
            self.ot_align = None
        # =====================================================================

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)

    def forward(
        self,
        x: Tensor,
        gra_sharp: Tensor,
        domain_ids: Optional[Tensor] = None,
        hr_pred: Optional[Tensor] = None,
        return_align_losses: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:

        # x: (B, C, T, H, W)
        b, c, t, fh, fw = x.shape

        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)  # [B, dim, T, H', W']

        x = self.patch_embedding(x)            # [B, dim, T/4, 4, 4]
        x = x.flatten(2).transpose(1, 2)       # [B, (T/4)*4*4, dim]

        Trans_features, _ = self.transformer1(x, gra_sharp)
        Trans_features2, _ = self.transformer2(Trans_features, gra_sharp)
        Trans_features3, _ = self.transformer3(Trans_features2, gra_sharp)

        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t//4, 4, 4)
        features_last = self.upsample(features_last)
        features_last = self.upsample2(features_last)
        features_last = torch.mean(features_last, 3)  # (B, dim//2, T, 4)
        features_last = torch.mean(features_last, 3)  # (B, dim//2, T)

        align_losses = None
        if self.enable_ot and self.ot_align is not None:
            if domain_ids is None:
                domain_ids = torch.zeros(features_last.size(0), dtype=torch.long, device=features_last.device)
            x_aligned, losses = self.ot_align(features_last, domain_ids=domain_ids, hr_pred=hr_pred)
            features_last = x_aligned
            if return_align_losses:
                align_losses = losses

        # 返回对齐后的时序特征 (B, C=dim//2, T)
        return features_last, align_losses
