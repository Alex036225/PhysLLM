import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple
import itertools
from torch.nn.utils import spectral_norm

class PotentialNet(nn.Module):
    def __init__(self, input_dim, feature_dim=1536):
        super().__init__()
        # 势函数
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, feature_dim),
            nn.ReLU()
        )
        # 势函数头（输出标量）
        self.phi_head = nn.Linear(feature_dim, 1)
    
    def forward(self, x):
        # 提取域不变特征
        features = self.feature_extractor(x)  # [batch_size, feature_dim]
        # 计算势函数值
        phi = self.phi_head(features)       # [batch_size, 1]
        return features, phi

def c_transform(features_source: torch.Tensor, 
                phi_values_source: torch.Tensor,
                features_target: torch.Tensor,
                cost_fn: callable) -> torch.Tensor:
    """
    计算c-transform，使用已提取的源域特征和势函数值
    
    Args:
        features_source: 源域的特征 [N, feature_dim]
        phi_values_source: 源域的势函数值 [N, 1]
        features_target: 目标域样本 [1, feature_dim]
        cost_fn: 成本函数
    """
    # 计算成本: cost(x_i, y) - phi(x_i)
    costs = cost_fn(features_source, features_target) - phi_values_source.squeeze(-1)
    return torch.min(costs)

def ot_dual_loss_multi_domain(
    domain_features: List[List[torch.Tensor]],  
    domain_phi_values: List[List[torch.Tensor]],  
    cost_fn: callable,
    reduction: str = 'mean'
) -> torch.Tensor:
    """   
    Args:
        domain_features: List of lists where each sublist contains 3 tensors of shape [N_i, len, feature_dim]
        domain_phi_values: List of lists where each sublist contains 3 tensors of shape [N_i, len, 1]
        cost_fn: Cost function
        reduction: Reduction method ('mean' or 'sum')
        
    Returns:
        loss: Computed loss tensor
    """
    losses = []
    num_domains = len(domain_features)
    
    # Assume all sublists have the same length (3 in this case)
    num_indices = len(domain_features[0])
    
    for idx in range(num_indices):
        # For each index, compute OT loss between domains
        for i, j in itertools.combinations(range(num_domains), 2):
            # Get features and phi values at current index
            features_i = domain_features[i][idx]  # Shape [N_i, len, feature_dim]
            phi_i = domain_phi_values[i][idx]     # Shape [N_i, len, 1]
            features_j = domain_features[j][idx]  # Shape [N_j, len, feature_dim]
            phi_j = domain_phi_values[j][idx]     # Shape [N_j, len, 1]
            
            # Compute mean phi for domain i
            mean_phi_i = phi_i.mean()
            
            # Compute mean c-transform for domain j
            phi_c_j_values = []
            for y in features_j:
                y_ = y.unsqueeze(0)  # Add batch dimension
                phi_c_j = c_transform(features_i, phi_i, y_, cost_fn)
                phi_c_j_values.append(phi_c_j)
            phi_c_j = torch.stack(phi_c_j_values).mean()
            
            # Dual loss component
            dual_loss = -(mean_phi_i + phi_c_j)
            losses.append(dual_loss)
    
    if not losses:
        return torch.tensor(0.0, device=domain_features[0][0].device)
    
    total_loss = torch.stack(losses)

    if reduction == 'mean':
        return total_loss.mean()
    elif reduction == 'sum':
        return total_loss.sum()
    else:
        raise ValueError("Invalid reduction. Must be 'mean' or 'sum'.")

# 成本函数：平方欧氏距离
def squared_cost(x, y):
    return torch.sum((x - y)**2, dim=-1)

# 自主学习的度量函数
class AdaptiveCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = None  # 初始化为 None，将在第一次前向传播时动态创建
    
    def forward(self, x, y):
        """
        计算自适应代价（学习马氏距离）: (x-y)^T W (x-y)
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, feature_dim)
            y (Tensor): 输入张量，形状为 (batch_size, feature_dim)

        Returns:
            cost (Tensor): 自适应代价，形状为 (batch_size,)
        """
        # 动态推断 feature_dim
        feature_dim = x.size(-1)  # 假设 x 和 y 的最后一个维度是特征维度
        # 如果 W 尚未初始化，动态创建 W
        if self.W is None:
            self.W = nn.Parameter(torch.randn(feature_dim, feature_dim)).cuda()
        # 计算差值
        diff = (x - y).cuda()
        # 使用 einsum 计算 (x-y)^T W (x-y)
        return torch.einsum('bsi,ij,bsj->bs', diff, self.W, diff)


class AdvancedPotentialNet(nn.Module):
    def __init__(self, input_dim, feature_dim=1536, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 势函数
        self.feature_extractor = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, feature_dim * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(feature_dim * 2, feature_dim * 2),
            spectral_norm(nn.Linear(feature_dim * 2, feature_dim)),
            nn.LayerNorm(feature_dim)
        )
        
        
        # 基于Transformer的势能头
        self.phi_head = TransformerPotentialHead(
            feature_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)  # [batch, feature_dim][4, 32, 1536]
        phi = self.phi_head(features)        # [batch, 1]
        return features, phi


class ResidualBlock(nn.Module):
    """残差块增强特征提取"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = spectral_norm(nn.Linear(in_dim, out_dim))
        self.linear2 = spectral_norm(nn.Linear(out_dim, out_dim))
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x + residual  # 残差连接


class TransformerPotentialHead(nn.Module):
    """基于自注意力的势能头"""
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.mlp = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, feature_dim * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(feature_dim * 2, feature_dim)),
            nn.GELU()
        )
        
        # 最终势能值输出
        self.phi_output = spectral_norm(nn.Linear(feature_dim, 1))
        
    def forward(self, x):
        # 自注意力处理（添加虚拟序列维度）
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP处理
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        # 输出势能值
        phi = self.phi_output(x.squeeze(1))
        return phi