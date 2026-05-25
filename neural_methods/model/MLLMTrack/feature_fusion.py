from neural_methods.model.MLLMTrack.MMFuser import MMFuser
import torch
from torch import nn
import torch.nn.functional as F

class FeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim, bottleneck_dim=None):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        if bottleneck_dim is not None and bottleneck_dim < output_dim:
            # 使用瓶颈结构：先投影到 bottleneck_dim，再投影到 output_dim
            self.projection = nn.Sequential(
                nn.Linear(input_dim, bottleneck_dim),
                nn.GELU(), # 可选，增加非线性
                nn.Linear(bottleneck_dim, output_dim)
            )
        else:
            # 原始结构
            self.projection = nn.Linear(input_dim, output_dim)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_map, target_len):
        # feature_map: [batch, channel, time, H, W]
        B, C, T, H, W = feature_map.shape

        # 空间池化
        x = feature_map.transpose(1, 2)  # [B, T, C, H, W]
        x = x.reshape(-1, C, H, W)  # [B*T, C, H, W]
        x = self.spatial_pool(x)  # [B*T, C, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B*T, C]
        x = x.reshape(B, T, C)  # [B, T, C]

        # 时间维度调整
        x = F.interpolate(x.transpose(1, 2), size=target_len,
                          mode='linear').transpose(1, 2)  # [B, target_len, C]
        # print("x:", x.shape)
        # 投影到目标维度
        x = self.projection(x)  # [B, target_len, output_dim]
        return x


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, feature_channels, token_dim):
        super().__init__()
        self.projections = nn.ModuleList([
            FeatureProjection(in_dim, token_dim)
            for in_dim in feature_channels
        ])

    def forward(self, features, target_len):
        # features: 包含不同尺度特征的列表
        projected_features = []
        # print(features[0].shape)
        # print(features[1].shape)
        # print(features[2].shape)
        for feat, proj in zip(features, self.projections):
            proj_feat = proj(feat, target_len)
            projected_features.append(proj_feat)

        # 拼接所有特征
        return torch.cat(projected_features, dim=1)  # [B, len*num_features, token_dim]
    
class MultiAverageFeatureFusion(nn.Module):
    def __init__(self, feature_channels, token_dim):
        super().__init__()
        self.projections = nn.ModuleList([
            FeatureProjection(in_dim, token_dim)
            for in_dim in feature_channels
        ])

    def forward(self, features, target_len):
        # features: 包含不同尺度特征的列表
        projected_features = []
        # print(features[0].shape)
        # print(features[1].shape)
        # print(features[2].shape)
        for feat, proj in zip(features, self.projections):
            proj_feat = proj(feat, target_len)
            projected_features.append(proj_feat)

        # 拼接所有特征
        fused_feature = torch.stack(projected_features, dim=0).sum(dim=0) / len(projected_features)
        return fused_feature
    
class MMFusion(nn.Module):
    def __init__(self, feature_channels, token_dim, bottleneck_dim=384): # 添加 bottleneck_dim 参数
        super().__init__()
        # 修改 FeatureProjection，传入 bottleneck_dim
        self.projections = nn.ModuleList([
            FeatureProjection(in_dim, token_dim, bottleneck_dim) # 在投影时就使用瓶颈
            for in_dim in feature_channels
        ])
        self.bottleneck_dim = bottleneck_dim
        self.token_dim = token_dim

        # 如果 bottleneck_dim < token_dim，则需要降维和升维层来包裹 MMFuser
        if bottleneck_dim < token_dim:
            self.dim_reduction = nn.Linear(token_dim, bottleneck_dim)
            self.dim_increase = nn.Linear(bottleneck_dim, token_dim)
            # MMFuser 在瓶颈维度上操作
            self.MMFuser = MMFuser(bottleneck_dim, 8) # 假设头数不变或根据 bottleneck_dim 调整
        else:
             # 如果 bottleneck_dim >= token_dim，则直接使用原始 MMFuser
             self.MMFuser = MMFuser(token_dim, 8)
             # 不需要额外的降维/升维层
             self.dim_reduction = None
             self.dim_increase = None

    def forward(self, features, target_len):
        # features: 包含不同尺度特征的列表
        projected_features = []
        # print(features[0].shape)
        # print(features[1].shape)
        # print(features[2].shape)
        for feat, proj in zip(features, self.projections):
            proj_feat = proj(feat, target_len)
            projected_features.append(proj_feat)

        # 拼接所有特征 (除了最后一个)
        X = torch.cat(projected_features[:-1], dim=1) # 修改：拼接除最后一个外的所有
        FL = projected_features[-1]

        # --- 应用 MMFuser (可能在瓶颈维度) ---
        if self.dim_reduction is not None and self.dim_increase is not None:
            # 降维
            X_reduced = self.dim_reduction(X)
            FL_reduced = self.dim_reduction(FL)
            # MMFuser 融合
            fused_feature_reduced = self.MMFuser(X_reduced, FL_reduced)
            # 升维
            fused_feature = self.dim_increase(fused_feature_reduced)
        else:
            # 直接融合
            fused_feature = self.MMFuser(X, FL)

        return fused_feature
    
class OTFusion(nn.Module):
    def __init__(self, feature_channels, token_dim, PotentialNet):
        super().__init__()
        self.projections = nn.ModuleList([
            FeatureProjection(in_dim, token_dim)
            for in_dim in feature_channels
        ])
        self.MMFuser = MMFuser(token_dim, 8)
        self.potentialNet = PotentialNet

    def forward(self, features, target_len):
        # features: 包含不同尺度特征的列表
        projected_features = []
        phi_list = []
        # print(features[0].shape)
        # print(features[1].shape)
        # print(features[2].shape)
        for feat, proj in zip(features, self.projections):
            proj_feat = proj(feat, target_len)
            proj_fusion, phi = self.potentialNet(proj_feat)
            phi_list.append(phi)
            projected_features.append(proj_fusion)

        # 拼接所有特征
        X = torch.cat(projected_features[:-2], dim=1)
        FL = projected_features[-1]
        fused_feature = self.MMFuser(X, FL)

        return fused_feature, projected_features, phi_list  # 向量，列表
    
class Projector(nn.Module):
    def __init__(self, feature_channels, token_dim):
        super().__init__()
        self.projections = nn.ModuleList([
            FeatureProjection(in_dim, token_dim)
            for in_dim in feature_channels
        ])

    def forward(self, features, target_len):
        # features: 包含不同尺度特征的列表
        projected_features = []
        phi_list = []
        for feat, proj in zip(features, self.projections):
            proj_feat = proj(feat, target_len)
            projected_features.append(proj_feat)

        return projected_features  # 向量，列表