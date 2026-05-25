import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.nn.utils import spectral_norm
from itertools import combinations

class DomainGeneralizationOT(nn.Module):
    def __init__(self, 
                 input_dim=512,           # 统一输入维度
                 latent_dim=256,          # 共享潜在空间维度
                 num_domains=3,           # 源域数量
                 temp=0.1,                # 权重温度系数
                 proj_type='sphere',      # 投影类型
                 ot_alternate_steps=2,    # 交替优化步数
                 use_spectral_norm=True):
        super().__init__()
        self.K = num_domains
        self.latent_dim = latent_dim
        self.temp = temp
        self.proj_type = proj_type
        self.ot_steps = ot_alternate_steps
        
        # 可学习的域权重
        self.domain_weights = nn.Parameter(torch.ones(num_domains)/num_domains)
        
        # 共享特征投影网络
        self.domain_projectors = nn.ModuleList([
            self._build_projection(input_dim, latent_dim, use_spectral_norm)
        ])
        
        # 对偶势函数网络
        self.potential_net = self._build_potential_net(latent_dim, use_spectral_norm)
        
        # 域间协方差记忆库
        self.register_buffer('cov_memory', torch.eye(num_domains))
        
    def _build_projection(self, in_dim, out_dim, use_sn):
        layers = [
            nn.Linear(in_dim, out_dim*2),
            nn.LayerNorm(out_dim*2),
            nn.GELU()
        ]
        if use_sn:
            layers.append(spectral_norm(nn.Linear(out_dim*2, out_dim)))
        else:
            layers.append(nn.Linear(out_dim*2, out_dim))
        return nn.Sequential(*layers)
    
    def _build_potential_net(self, latent_dim, use_sn):
        layers = []
        dims = [latent_dim, 512, 256, 1]
        for i in range(len(dims)-1):
            lin = nn.Linear(dims[i], dims[i+1])
            if use_sn and i != len(dims)-2:
                lin = spectral_norm(lin)
            layers.append(lin)
            if i != len(dims)-2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.SiLU())
        return nn.Sequential(*layers)
    
    def _domain_alignment(self, domain_features):
        """ 多域分布对齐 """
        # 1. 计算域间MMD
        mmd_loss = 0
        for i, j in combinations(range(self.K), 2):
            mmd_loss += self._compute_mmd(domain_features[i], domain_features[j])
        
        # 2. 二阶矩对齐
        cov_loss = 0
        covs = [self._covariance(f) for f in domain_features]
        for i, j in combinations(range(self.K), 2):
            cov_loss += F.mse_loss(covs[i], covs[j])
        
        return mmd_loss + 0.5*cov_loss
    
    def _compute_mmd(self, x, y):
        """ 最大均值差异 """
        x_kernel = self._gaussian_kernel(x, x)
        y_kernel = self._gaussian_kernel(y, y)
        xy_kernel = self._gaussian_kernel(x, y)
        return x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    
    def _gaussian_kernel(self, x, y, sigma=1.0):
        pairwise_dist = torch.cdist(x, y)**2
        return torch.exp(-pairwise_dist / (2*sigma**2))
    
    def _covariance(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        return torch.einsum('bdi,bdj->bij', x, x) / (x.size(1)-1)
    
    def _ot_alternating(self, source_features, projecteds):
        """ 交替优化OT对偶问题 """
        
        # 阶段1: 固定势函数，优化映射
        with torch.no_grad():
            potentials = [self.potential_net(f.mean(1)) for f in projecteds]
        
        for _ in range(self.ot_steps):
            transport_loss = 0
            for k in range(self.K):
                cost = F.mse_loss(source_features[k], projecteds[k])
                transport_loss += self.domain_weights[k] * (cost - potentials[k])
            transport_loss.backward()
            # 这里需要添加优化器步骤（实际使用时需配合优化器）
        
        # 阶段2: 固定映射，优化势函数
        with torch.no_grad():
            new_projecteds = [proj(f) for proj, f in zip(self.domain_projectors, source_features)]
        
        potential_loss = 0
        for k in range(self.K):
            cost = F.mse_loss(source_features[k], new_projecteds[k].detach())
            potential = self.potential_net(new_projecteds[k].mean(1))
            potential_loss += self.domain_weights[k] * (potential - cost)
        (-potential_loss).backward()  # 最大化目标
        # 同样需要优化器步骤
        
        return transport_loss + potential_loss
    
    def forward(self, domain_features):
        """ 
        修改后的前向传播流程：
        1. 每次调用只执行一步交替优化（投影网络或势函数）
        2. 自动更新训练状态
        3. 保留所有损失计算和特征融合逻辑
        """
        # === 1. 特征投影 ===
        projecteds = [self._project(proj(f), self.proj_type) 
                     for proj, f in zip(self.domain_projectors, domain_features)]
        
        # === 2. 交替优化控制 ===
        if self.current_ot_step % 2 == 0:
            # 阶段1：优化传输映射（固定势函数）
            with torch.no_grad():
                potentials = [self.potential_net(f.mean(1)) for f in projecteds]
            
            ot_loss = 0
            for k in range(self.K):
                cost = F.mse_loss(domain_features[k], projecteds[k])
                ot_loss += self.domain_weights[k] * (cost - potentials[k])
        else:
            # 阶段2：优化势函数（固定投影网络）
            with torch.no_grad():
                new_projecteds = [proj(f) for proj, f in zip(self.domain_projectors, domain_features)]
            
            ot_loss = 0
            for k in range(self.K):
                cost = F.mse_loss(domain_features[k], new_projecteds[k].detach())
                potential = self.potential_net(new_projecteds[k].mean(1))
                ot_loss += self.domain_weights[k] * (potential - cost)
            ot_loss = -ot_loss  # 最大化目标
        
        # === 3. 域对齐损失（保留原设计） ===
        align_loss = self._domain_alignment(projecteds)
        
        # === 4. 动态权重更新（保留原设计） ===
        with torch.no_grad():
            sim_matrix = torch.corrcoef(torch.stack([f.mean(0) for f in projecteds]))
            self.cov_memory = 0.9*self.cov_memory + 0.1*sim_matrix
            self.domain_weights.data = F.softmax(
                self.temp * self.cov_memory.mean(dim=1), dim=0)
        
        # === 5. 更新训练状态 ===
        self.current_ot_step = (self.current_ot_step + 1) % self.max_ot_steps
        self.total_steps += 1
        
        # === 6. 融合不变特征 ===
        weights = F.softmax(self.domain_weights, dim=0)
        invariant_feature = sum(w * f for w, f in zip(weights, projecteds))
        
        return invariant_feature, {
            'total_loss': ot_loss + 0.5*align_loss,
            'ot_loss': ot_loss,
            'align_loss': align_loss,
            'domain_weights': weights
        }

    def _project(self, z, proj_type):
        """ 几何约束投影 """
        if proj_type == 'sphere':
            return z / z.norm(dim=-1, keepdim=True) * sqrt(self.latent_dim)
        elif proj_type == 'chi':
            chi = torch.randn(z.size(0), 1, device=z.device).pow(2)
            return z * torch.sqrt(chi) / z.norm(dim=-1, keepdim=True)
        return z