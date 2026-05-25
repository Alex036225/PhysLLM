from neural_methods.model.Phys.EfficientPhys import EfficientPhys
import torch
import torch.nn as nn
import clip
from collections import OrderedDict
from neural_methods.model.Phys.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp, PhysFormer_encoder
# from neural_methods.model.Phys.PhysMamba import PhysMamba
from .abstract_base import EncoderBase, resolve_checkpoint_path
import torchvision.transforms.functional as TF
from neural_methods.model.Phys.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
import torch.nn.functional as F
from pytorch_wavelets import DWT1D, IDWT1D


def _strip_state_dict_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key[7:] if key.startswith("module.") else key] = value
    return new_state_dict


def _load_checkpoint_state(path, nested_key=None):
    checkpoint = torch.load(path, map_location="cpu")
    if nested_key and isinstance(checkpoint, dict) and nested_key in checkpoint:
        checkpoint = checkpoint[nested_key]
    elif isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    return _strip_state_dict_prefix(checkpoint)

class CLIPVideoEncoder(EncoderBase):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载预训练的 CLIP 模型和预处理器
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.float()  # otherwise the default is 'float16'
        # 冻结模型参数
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # 定义 CLIP 使用的均值和标准差
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device)

    def encode(self, x):
        """
        使用 CLIP 模型对视频进行编码，提取视频特征。

        参数:
            x: 输入的视频数据，形状为 (batch_size, num_frames, channels, height, width)
        返回:
            video_features: 形状为 (batch_size, feature_dim)
        """
        batch_size, channels, num_frames, height, width = x.size()

        # 将视频帧展开为 (batch_size * num_frames, channels, height, width)
        frames = x.view(-1, channels, height, width).to(self.device)

        # 归一化像素值到 [0, 1]
        frames = (frames - frames.min()) / (frames.max() - frames.min())

        # Resize 到 (224, 224)
        frames = torch.nn.functional.interpolate(frames, size=(224, 224), mode='bicubic', align_corners=False)

        # Normalization
        frames = self.normalize(frames)

        # 使用 CLIP 提取每一帧的特征
        with torch.no_grad():
            frame_features = self.model.encode_image(frames)  # Shape: (batch_size * num_frames, feature_dim)

        # 将帧特征重新组合为 (batch_size, num_frames, feature_dim)
        video_features = frame_features.view(batch_size, num_frames, -1)  # Shape: (batch_size, num_frames, feature_dim)

        # 对帧特征进行聚合，例如取平均值
        # video_features = video_features.mean(dim=1)  # Shape: (batch_size, feature_dim)

        return video_features

    def normalize(self, tensor):
        """
        对输入的张量进行归一化，使其符合 CLIP 模型的预处理要求。

        参数:
            tensor: 输入张量，形状为 (N, C, H, W)
        返回:
            归一化后的张量
        """
        # tensor: (N, C, H, W)
        mean = self.clip_mean.view(1, -1, 1, 1)
        std = self.clip_std.view(1, -1, 1, 1)
        return (tensor - mean) / std



class PhysFormerVideoEncoder(EncoderBase):
    def __init__(self, configs):
        
        # 从 configs 中提取模型参数
        chunk_len = configs.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        resize_h = configs.TRAIN.DATA.PREPROCESS.RESIZE.H
        resize_w = configs.TRAIN.DATA.PREPROCESS.RESIZE.W
        image_size = (chunk_len, resize_h, resize_w)

        patch_size = configs.MODEL.PHYSFORMER.PATCH_SIZE
        patches = (patch_size,) * 3

        dim = configs.MODEL.PHYSFORMER.DIM
        ff_dim = configs.MODEL.PHYSFORMER.FF_DIM
        num_heads = configs.MODEL.PHYSFORMER.NUM_HEADS
        num_layers = configs.MODEL.PHYSFORMER.NUM_LAYERS
        dropout_rate = configs.MODEL.DROP_RATE
        theta = configs.MODEL.PHYSFORMER.THETA

        # 初始化 PhysFormer 模型
        self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=image_size,
            patches=patches,
            dim=dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            theta=theta
        )
        
        ckpt_path = resolve_checkpoint_path(
            configs, "VIDEO_ENCODER", "PHYSLLM_VIDEO_ENCODER_CKPT"
        )
        self.model.load_state_dict(_load_checkpoint_state(ckpt_path))
        self.model.cuda()

        # 将模型设置为评估模式并冻结参数
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, x):
        """
        使用 PhysFormer 编码器处理视频。

        参数:
            facial_video: 形状为 (batch_size, num_frames, channels, height, width)
        返回:
            video_features: 形状为 (batch_size, seq_len)，即预测的 rPPG 信号
        """
        # 确保输入在正确的设备上
        # facial_video = facial_video.to(self.device)

        # 调整输入形状为 (batch_size, channels, num_frames, height, width)
        assert x.dim() == 5

        with torch.no_grad():
            # 设置 gra_sharp 参数
            gra_sharp = 2.0
            gra_sharp = torch.tensor(2.0, device="cuda:0")
            # 前向传播
            rPPG, Trans_features, Trans_features2, Trans_features3 = self.model(x, gra_sharp)
            rPPG_raw = rPPG
            # 归一化 rPPG 信号(时域弱平稳)
            # rPPG = rPPG - torch.mean(rPPG, dim=-1, keepdim=True)
            # rPPG = rPPG / torch.std(rPPG, dim=-1, keepdim=True)
            # 加入TimeFreqStationary
            # seq_len = rPPG.shape[1]
            # kernel_len = 5
            # wavelet = "sym3"
            # j = 2
            # model = TimeFreqStationary(seq_len, kernel_len, wavelet, j).cuda()
            # rPPG = model(rPPG)
        # 返回 rPPG 信号作为视频特征
        return rPPG, Trans_features, Trans_features2, Trans_features3



# class PhysMambaVideoEncoder(EncoderBase):
#     def __init__(self,):

#         # 初始化 PhysFormer 模型
#         self.model = PhysMamba()
#         ckpt_path = "/path/to/UBFC-rPPG_PhysMamba_DiffNormalized.pth"
#         state_dict = torch.load(ckpt_path)
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             if k.startswith("module."):
#                 new_state_dict[k[7:]] = v  # 去掉 'module.'
#             else:
#                 new_state_dict[k] = v
#         self.model.load_state_dict(new_state_dict)
#         self.model.cuda()

#         # 将模型设置为评估模式并冻结参数
#         self.model.eval()
#         for param in self.model.parameters():
#             param.requires_grad = False

#     def encode(self, x):
#         """
#         使用 PhysFormer 编码器处理视频。

#         参数:
#             facial_video: 形状为 (batch_size, num_frames, channels, height, width)
#         返回:
#             video_features: 形状为 (batch_size, seq_len)，即预测的 rPPG 信号
#         """
#         # 确保输入在正确的设备上
#         # facial_video = facial_video.to(self.device)

#         # 调整输入形状为 (batch_size, channels, num_frames, height, width)
#         assert x.dim() == 5

#         with torch.no_grad():
#             # 前向传播
#             rPPG = self.model(x)
#             # 归一化 rPPG 信号
#             # rPPG = rPPG - torch.mean(rPPG, dim=-1, keepdim=True)
#             # rPPG = rPPG / torch.std(rPPG, dim=-1, keepdim=True)

#         # 返回 rPPG 信号作为视频特征
#         return rPPG
    

class PhysNetVideoEncoder(EncoderBase):
    def __init__(self, configs):

        frames_num = configs.MODEL.PHYSNET.FRAME_NUM

        # 初始化 PhysNet 模型
        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=frames_num).to('cuda')  # [3, T, 128,128]

        ckpt_path = resolve_checkpoint_path(
            configs, "VIDEO_ENCODER", "PHYSLLM_VIDEO_ENCODER_CKPT"
        )
        self.model.load_state_dict(_load_checkpoint_state(ckpt_path))
        # self.model.cuda()
        # self.model = torch.nn.DataParallel(self.model, device_ids=list(range(1,2)))

        # 将模型设置为评估模式并冻结参数
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, x):
        """
        使用 PhysNet 编码器处理视频。
        参数:
        facial_video: 形状为 (batch_size, channels, num_frames, height, width)
        返回:
        video_features: 形状为 (batch_size, seq_len)，即预测的 rPPG 信号
        """
        # 确保输入在正确的设备上
        # facial_video = facial_video.to(self.device)

        # 调整输入形状为 (batch_size, channels, num_frames, height, width)
        assert x.dim() == 5

        with torch.no_grad():
            # 前向传播
            rPPG, x_visual6464, x_visual3232, x_visual1616 = self.model(x)
            # 标准化 rPPG 信号
            # rPPG = rPPG - torch.mean(rPPG, dim=-1, keepdim=True)
            # rPPG = rPPG / torch.std(rPPG, dim=-1, keepdim=True)

            # # 加入TimeFreqStationary
            # seq_len = rPPG.shape[1]
            # kernel_len = 40
            # wavelet = "sym3"
            # j = 2
            # model = TimeFreqStationary(seq_len, kernel_len, wavelet, j)
            # rPPG = model(rPPG)
        # 返回 rPPG 信号作为视频特征
        return rPPG,x_visual6464, x_visual3232, x_visual1616


class EfficientPhysVideoEncoder(EncoderBase):
    def __init__(self, configs):

        # 从 configs 中提取模型参数
        # chunk_len = 160  # configs.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        resize_h = configs.TRAIN.DATA.PREPROCESS.RESIZE.H
        resize_w = configs.TRAIN.DATA.PREPROCESS.RESIZE.W
        self.frame_depth = configs.MODEL.EFFICIENTPHYS.FRAME_DEPTH
        self.num_of_gpu = 1
        self.base_len = self.num_of_gpu * self.frame_depth
        # 初始化 EfficientPhys 模型
        # self.model = EfficientPhys(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to('cuda')
        self.model = EfficientPhys(frame_depth=self.frame_depth, img_size=resize_h).to('cuda')

        ckpt_path = resolve_checkpoint_path(
            configs, "VIDEO_ENCODER", "PHYSLLM_VIDEO_ENCODER_CKPT"
        )
        self.model.load_state_dict(_load_checkpoint_state(ckpt_path))
        # self.model.cuda()
        # self.model = torch.nn.DataParallel(self.model, device_ids=list(range(1,2)))

        # 将模型设置为评估模式并冻结参数
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, x):
        """
        使用 PhysFormer 编码器处理视频。

        参数:
            facial_video: 形状为 (batch_size, channels, num, height, width)
        返回:
            video_features: 形状为 (batch_size, seq_len)，即预测的 rPPG 信号
        """
        # 确保输入在正确的设备上
        # facial_video = facial_video.to(self.device)

        # 调整输入形状为 (batch_size, channels, num_frames, height, width)
        assert x.dim() == 5

        with torch.no_grad():
            # 设置 gra_sharp 参数
            x = x.transpose(1, 2).contiguous()
            N, D, C, H, W = x.shape
            data = x.view(N * D, C, H, W)
            # labels = labels.view(-1, 1)
            data = data[:(N * D) // self.base_len * self.base_len]
            # Add one more frame for EfficientPhys since it does torch.diff for the input
            last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
            data = torch.cat((data, last_frame), 0)
            # labels = labels[:(N * D) // self.base_len * self.base_len]
            rPPG, d2, d3, d7 = self.model(data)
            # print('d2 : ', d2.shape) # torch.Size([512, 32, 126, 126])
            # print('d3 : ', d3.shape) # torch.Size([512, 32, 63, 63])
            # print('d7 : ', d7.shape) # torch.Size([512, 64, 30, 30])
            # d2 = d2.view(N, D, d2.shape[1], d2.shape[2], d2.shape[3]).transpose(2, 3).contiguous()
            d2 = d2.view(N, D, d2.shape[1], d2.shape[2], d2.shape[3]).transpose(1, 2).contiguous()
            d3 = d3.view(N, D, d3.shape[1], d3.shape[2], d3.shape[3]).transpose(1, 2).contiguous()
            d7 = d7.view(N, D, d7.shape[1], d7.shape[2], d7.shape[3]).transpose(1, 2).contiguous()

            rPPG = rPPG.reshape(N, D)
            # 归一化 rPPG 信号
            # rPPG = rPPG - torch.mean(rPPG, dim=-1, keepdim=True)
            # rPPG = rPPG / torch.std(rPPG, dim=-1, keepdim=True)

        # 返回 rPPG 信号作为视频特征
        return rPPG, d2, d3, d7
        
    
# TODO: 时域频域
class TimeFreqStationary(nn.Module):
    def __init__(self, seq_len, kernel_len, wavelet="coif6", j=2): # j 取 2 
        super(TimeFreqStationary, self).__init__()
        self.seq_len = seq_len
        self.kernel_len = kernel_len
        self.epsilon = 1e-5
        # wavelet ：小波基
        # coif6: Coiflet 小波，适合对信号进行高精度分解。
        # sym3: Symlet 小波，更对称、更平滑。
        # db4: Daubechies 小波，经典小波基，适合广泛应用。

        # 时域滑动窗口处理
        self.pad = nn.ReplicationPad1d(padding=(kernel_len // 2, kernel_len // 2 - ((kernel_len + 1) % 2)))

        # 频域小波变换
        self.dwt = DWT1D(wave=wavelet, J=j)
        self.idwt = IDWT1D(wave=wavelet)  # 逆小波变换

        # 时域与频域的权重参数
        self.dwt_ratio = nn.Parameter(
            torch.clamp(torch.full((1, 1), 0.5), min=0., max=1.)  # 单通道权重
        )

    def forward(self, x):
        """
        Args:
            x: 输入时间序列，形状为 [batch_size, seq_len]

        Returns:
            融合后的平稳序列，形状与输入相同。
        """
        # --- 时域平稳化 ---

        x_window = x.unfold(-1, self.kernel_len, 1)  # 滑动窗口展开
        m, s = x_window.mean(dim=-1), x_window.std(dim=-1)  # 滑动窗口均值和标准差
        m, s = self.pad(m), self.pad(s)  # 填充以匹配输入维度
        norm_x = (x - m) / (s + self.epsilon)  # 时域平稳化序列

        # --- 频域平稳化 ---
        ac, dc_list = self.dwt(x.unsqueeze(1))  # 小波分解（增加通道维度以适配 DWT1D）
        ac = ac.squeeze(1)  # 移除通道维度
        norm_ac, (mac, sac) = self._normalize(ac)
        norm_dc, m_list, s_list = [], [], []
        for dc in dc_list:
            dc = dc.squeeze(1)  # 移除通道维度
            norm_dc_part, (mdc, sdc) = self._normalize(dc)
            norm_dc.append(norm_dc_part)
            m_list.append(mdc)
            s_list.append(sdc)

        # 将平稳化后的低频和高频分量重构
        freq_x = self.idwt([norm_ac.unsqueeze(1), [d.unsqueeze(1) for d in norm_dc]]).squeeze(1)

        # --- 加权求和 ---
        dwt_r, time_r = self.dwt_ratio, 1 - self.dwt_ratio
        combined_x = norm_x * time_r + freq_x * dwt_r

        return combined_x

    def _normalize(self, x):
        """对输入序列进行滑动窗口归一化"""
        x_window = x.unfold(-1, self.kernel_len, 1)  # 滑动窗口展开
        m, s = x_window.mean(dim=-1), x_window.std(dim=-1)  # 计算均值和标准差
        m, s = self.pad(m), self.pad(s)  # 填充
        norm_x = (x - m) / (s + self.epsilon)  # 归一化
        return norm_x, (m, s)

# 示例
# if __name__ == "__main__":
#     batch_size = 8
#     seq_len = 64
#     kernel_len = 5
#     wavelet = "sym3"
#     j = 2

#     x = torch.rand(batch_size, seq_len)  # 单通道输入
#     model = TimeFreqStationary(seq_len, kernel_len, wavelet, j)
#     output = model(x)
#     print("Output shape:", output.shape)


class CrossAttentionFusion(nn.Module):
    def __init__(self, clip_feature_dim, physformer_feature_dim, fusion_dim):
        super(CrossAttentionFusion, self).__init__()
        
        # 用于线性变换的投影层
        self.clip_projection = nn.Linear(clip_feature_dim, fusion_dim)
        self.physformer_projection = nn.Linear(physformer_feature_dim, fusion_dim)
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, batch_first=True)
        
    def forward(self, clip_features, physformer_features):
        """
        参数：
            clip_features: (batch_size, num_frames, clip_feature_dim)
            physformer_features: (batch_size, seq_len, physformer_feature_dim)
        返回：
            fused_features: (batch_size, fusion_dim)
        """
        # 投影到相同的特征空间
        clip_features = self.clip_projection(clip_features)  # (batch_size, num_frames, fusion_dim)
        physformer_features = self.physformer_projection(physformer_features)  # (batch_size, seq_len, fusion_dim)
        
        # 为交叉注意力准备 Query、Key、Value
        # Query 是 clip_features, Key 和 Value 是 physformer_features
        query = physformer_features  # (batch_size, num_frames, fusion_dim)
        key = clip_features  # (batch_size, seq_len, fusion_dim)
        value = clip_features  # (batch_size, seq_len, fusion_dim)
        
        # 使用交叉注意力融合
        attended_features, _ = self.cross_attention(query, key, value)  # (batch_size, seq_len, fusion_dim)
        
        # 对时间维度进行池化，得到最终融合的特征
        # fused_features = attended_features.mean(dim=1)  # (batch_size, fusion_dim)
        
        return attended_features

class PhysformerCLIP(nn.Module):
    def __init__(self, clip_feature_dim=512, physformer_feature_dim=128):
        super(PhysformerCLIP, self).__init__()
        
        # 初始化 CLIP 和 PhysFormer 编码器
        self.clip_encoder = CLIPVideoEncoder()
        self.physformer_encoder = PhysFormerEncoder()
        fusion_dim = self.physformer_encoder.model.dim // 2 
        # 初始化交叉注意力融合模块
        self.fusion_module = CrossAttentionFusion(clip_feature_dim, physformer_feature_dim, fusion_dim)
        self.predictor = self.physformer_encoder.model.ConvBlockLast

    def encode(self, video):
        """
        参数：
            video: (batch_size, num_frames, channels, height, width)
        返回：
            fused_features: (batch_size, fusion_dim)
        """
        # 使用 CLIP 编码器提取视频特征
        clip_features = self.clip_encoder.encode(video)  # (batch_size, num_frames, clip_feature_dim)

        # 使用 PhysFormer 编码器提取视频特征
        physformer_features = self.physformer_encoder.encode(video)  # (batch_size, seq_len, physformer_feature_dim)

        # print("clip_features", clip_features.shape)
        # print("physformer_features", physformer_features.shape)

        # 使用交叉注意力融合特征
        fused_features = self.fusion_module(clip_features, physformer_features)
        # print(fused_features.shape)
        rPPG = self.predictor(fused_features) 
        rPPG = rPPG.squeeze(1) 

        return rPPG


    
