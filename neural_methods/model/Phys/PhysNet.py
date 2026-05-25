# physnet_with_ot.py
# 将 OT 模块集成进 PhysNet（在 32x32 / 16x16 两个尺度的特征上做对齐）
# wrapper.py 与 transportnet.py 放在 neural_methods/model/OTTrack/ 下时，按下面 import 即可用

import torch
import torch.nn as nn

from neural_methods.model.OTTrack.wrapper import OTAlignWrapper, OTAlignConfig


class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(
        self,
        frames: int = 128,
        # ===== OT 开关与配置 =====
        use_ot: bool = True,
        num_domains: int = 3,          # 你自己的域数（domain_ids in [0, num_domains-1]）
        use_internal_hr_head: bool = True,  # 如果你不传 hr_pred，就设 True 让 OT 内部用 HRHead 估计
        ot_lambda_identity: float = 0.05,
        ot_lambda_band: float = 0.10,
        ot_lambda_src_consistency: float = 0.05,
    ):
        super().__init__()
        self.frames = frames
        self.use_ot = use_ot
        self.num_domains = num_domains

        # ======================
        # PhysNet 原始结构
        # ======================
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=64, out_channels=64,
                kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]
            ),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=64, out_channels=64,
                kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]
            ),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        # ======================
        # OT 模块（集成点：32x32 与 16x16 两个特征）
        # 说明：wrapper 支持 5D (B,C,T,H,W) 输入，会自动池化空间 -> (B,T,C) 做对齐，再广播还原
        # ======================
        if self.use_ot:
            # 3232 特征：通道 64，时间长度 T/2
            cfg_3232 = OTAlignConfig(
                feat_dim=64,
                num_prototypes=64,
                align_layers=3,
                align_kernel=3,
                use_attention=False,
                align_dropout=0.1,

                lambda_hr=1.0,
                sigma_hr=10.0,
                learnable_w=True,

                sink_epsilon=0.08,
                sink_iters=50,
                sink_debiased=True,

                lambda_band=ot_lambda_band,
                band_fps=30.0,
                band_range=(0.7, 3.0),

                lambda_identity=ot_lambda_identity,
                lambda_src_consistency=ot_lambda_src_consistency,

                use_internal_hr_head=use_internal_hr_head,

                # PhysNet 的中间特征是 (B,C,T,H,W)；这里不强制，ShapeAdapter 会自动识别
                time_dim=None,
                channel_dim=None,
            )
            self.ot_3232 = OTAlignWrapper(cfg_3232)

            # 1616 特征：通道 64，时间长度 T/4
            cfg_1616 = OTAlignConfig(**{**cfg_3232.__dict__})
            self.ot_1616 = OTAlignWrapper(cfg_1616)
        else:
            self.ot_3232 = None
            self.ot_1616 = None

    def forward(
        self,
        x: torch.Tensor,                  # (B,3,T,128,128)  (PhysNet 原始输入)
        domain_ids: torch.Tensor = None,  # (B,)  每个样本的域 id
        hr_pred: torch.Tensor = None,     # (B,)  可选：连续标签(如 HR)，用于条件代价；不传则可内部估计(需 use_internal_hr_head=True)
        return_ot_losses: bool = True,
    ):
        """
        返回：
          rPPG: (B, T_in)  这里用输入的 length（原 PhysNet 写法）
          x_visual: 原输入 x
          x_visual3232: (B,64,T/2,32,32)（若启用OT，则为对齐后的）
          x_visual1616: (B,64,T/4,16,16)（若启用OT，则为对齐后的）
          ot_losses (可选): dict
        """
        x_visual = x
        batch, channel, length, width, height = x.shape  # PhysNet 原始写法：length=时间维

        ot_losses = {}

        x = self.ConvBlock1(x)          # (B,16,T,128,128)
        x = self.MaxpoolSpa(x)          # (B,16,T,64,64)

        x = self.ConvBlock2(x)          # (B,32,T,64,64)
        x_visual6464 = self.ConvBlock3(x)  # (B,64,T,64,64)
        x = self.MaxpoolSpaTem(x_visual6464)  # (B,64,T/2,32,32)

        x = self.ConvBlock4(x)          # (B,64,T/2,32,32)
        x_visual3232 = self.ConvBlock5(x)  # (B,64,T/2,32,32)

        # ===== OT 注入点 #1：32x32 特征（对齐后再下采样）=====
        if self.use_ot and (domain_ids is not None):
            x_visual3232_aligned, loss_3232 = self.ot_3232(x_visual3232, domain_ids, hr_pred=hr_pred)
            x_visual3232 = x_visual3232_aligned
            if return_ot_losses:
                ot_losses.update({f"3232/{k}": v for k, v in loss_3232.items()})

        x = self.MaxpoolSpaTem(x_visual3232)  # (B,64,T/4,16,16)

        x = self.ConvBlock6(x)          # (B,64,T/4,16,16)
        x_visual1616 = self.ConvBlock7(x)  # (B,64,T/4,16,16)

        # ===== OT 注入点 #2：16x16 特征（对齐后再空间下采样）=====
        if self.use_ot and (domain_ids is not None):
            x_visual1616_aligned, loss_1616 = self.ot_1616(x_visual1616, domain_ids, hr_pred=hr_pred)
            x_visual1616 = x_visual1616_aligned
            if return_ot_losses:
                ot_losses.update({f"1616/{k}": v for k, v in loss_1616.items()})

        x = self.MaxpoolSpa(x_visual1616)  # (B,64,T/4,8,8)

        x = self.ConvBlock8(x)          # (B,64,T/4,8,8)
        x = self.ConvBlock9(x)          # (B,64,T/4,8,8)
        x = self.upsample(x)            # (B,64,T/2,8,8)
        x = self.upsample2(x)           # (B,64,T,8,8)

        x = self.poolspa(x)             # (B,64,frames,1,1)  frames 默认 128
        x = self.ConvBlock10(x)         # (B,1,frames,1,1)

        rPPG = x.view(-1, length)       # 与原 PhysNet 一致：用输入 length

        if return_ot_losses:
            return rPPG, x_visual, x_visual3232, x_visual1616, ot_losses
        else:
            return rPPG, x_visual, x_visual3232, x_visual1616
