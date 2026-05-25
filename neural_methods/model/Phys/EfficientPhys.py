import torch
import torch.nn as nn
# 直接用你已有的 wrapper（已经适配 BTCHW）
from  neural_methods.model.OTTrack.wrapper import OTAlignWrapper, OTAlignConfig


class Attention_mask(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5


class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div
    def forward(self, x, n_segment=None):
        nseg = self.n_segment if n_segment is None else int(n_segment)
        nt, c, h, w = x.size()
        assert nt % nseg == 0, f"TSM expects nt % n_segment == 0, got nt={nt}, n_segment={nseg}"
        n_batch = nt // nseg
        x = x.view(n_batch, nseg, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                  # shift left
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]      # shift right
        out[:, :, 2*fold:] = x[:, :, 2*fold:]                 # not shift
        return out.view(nt, c, h, w)


class EfficientPhys(nn.Module):
    """
    现在 EfficientPhys 直接接收 (B, T, C, H, W)，并在最前面调用 wrapper 进行对齐。
    后续与论文代码一致：时间差分 -> BN -> TSM -> conv/attn/pool -> 全连接。
    返回：out 或 (out, d3, d5, d7)
    """
    def __init__(self,
                 in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3,
                 dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2),
                 nb_dense=128, frame_depth=20, img_size=36, channel='raw',
                 ot_cfg: OTAlignConfig = None):
        super().__init__()
        # ====== OT 对齐器（wrapper，已适配 BTCHW）======
        if ot_cfg is None:
            # 关键：声明时间维=1、通道维=2，确保 wrapper 以 BTCHW 方式解析
            ot_cfg = OTAlignConfig(
                feat_dim=in_channels,
                align_layers=2,
                align_kernel=3,
                use_attention=False,
                align_dropout=0.1,
                lambda_hr=1.0, sigma_hr=10.0,
                learnable_w=True,
                sink_epsilon=0.08, sink_iters=50, sink_debiased=True,
                lambda_band=0.1, band_fps=30.0, band_range=(0.7, 3.0),
                lambda_identity=0.05, lambda_src_consistency=0.05,
                use_internal_hr_head=True,
                time_dim=1, channel_dim=2,
            )
        self.ot = OTAlignWrapper(ot_cfg)

        # ====== 原 EfficientPhys ======
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense

        # TSM
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)

        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()

        # Avg pooling & Dropout
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        # Dense (与原实现一致)
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)    # 64*7*7
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)   # 64*16*16
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)   # 64*22*22
        elif img_size == 128:
            self.final_dense_1 = nn.Linear(57600, self.nb_dense, bias=True)   # 64*30*30
        else:
            raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)
        self.batch_norm = nn.BatchNorm2d(3)
        self.channel = channel
        self.frame_depth = frame_depth  # 仅用于校验/设置 TSM

    def forward(self, inputs: torch.Tensor, domain_ids: torch.Tensor = None, hr_pred: torch.Tensor = None,
                return_intermediates: bool = False):
        """
        inputs: (B, T, C, H, W)   —— 注意：这是你要求的 BTCHW
        """
        assert inputs.dim() == 5, f"Expect (B,T,C,H,W), got {tuple(inputs.shape)}"
        B, T, C, H, W = inputs.shape

        # 1) 直接调用 wrapper（它已适配 BTCHW）
        if domain_ids is None:
            domain_ids = torch.zeros(B, dtype=torch.long, device=inputs.device)
        inputs, ot_losses = self.ot(inputs, domain_ids, hr_pred=hr_pred)   # 仍是 (B,T,C,H,W)

        # 2) EfficientPhys 原流程（把时间维当作“批次”来做 TSM 与卷积）
        x = torch.diff(inputs, dim=1)                # (B, T-1, C, H, W)
        Tm1 = T - 1
        nt = B * Tm1
        x = x.reshape(nt, C, H, W)
        x = self.batch_norm(x)

        # 让 TSM 的分段等于 (T-1)，保证还原到 batch=B 的语义
        nseg = Tm1 if Tm1 > 0 else 1

        network_input = self.TSM_1(x, n_segment=nseg)
        d1 = torch.tanh(self.motion_conv1(network_input))
        d1 = self.TSM_2(d1, n_segment=nseg)
        d2 = torch.tanh(self.motion_conv2(d1))

        g1 = torch.sigmoid(self.apperance_att_conv1(d2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        d4 = self.TSM_3(d4, n_segment=nseg)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5, n_segment=nseg)
        d6 = torch.tanh(self.motion_conv4(d5))

        g2 = torch.sigmoid(self.apperance_att_conv2(d6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)               # (B*(T-1), 1)
        out = out.view(B, Tm1, 1)

        if return_intermediates:
            def unflatten(btchw):
                Bt, Cc, Hh, Ww = btchw.shape
                return btchw.view(B, Tm1, Cc, Hh, Ww)
            return out, unflatten(d3), unflatten(d5), unflatten(d7), ot_losses
        return out, ot_losses
