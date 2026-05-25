# conditional_btm_sb.py
"""
Conditional BTM + Schrödinger Bridge 模块
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

该模块封装了 Conditional BTM 和 Schrödinger Bridge 的逻辑，
用于执行特征对齐。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------- 1. 数值稳定 Sinkhorn (内部使用) -------------------- #
def log_sinkhorn(cost, reg, max_iter=100, tol=1e-9):
    """内部使用的 Sinkhorn 算法实现"""
    m, n = cost.shape
    log_a = torch.full((m,), -np.log(m), device=cost.device)
    log_b = torch.full((n,), -np.log(n), device=cost.device)
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    K = -cost / reg
    for _ in range(max_iter):
        f_prev = f
        g = log_b - torch.logsumexp(K + f[:, None], dim=0)
        f = log_a - torch.logsumexp(K + g[None, :], dim=1)
        if torch.max(torch.abs(f - f_prev)) < tol:
            break
    return torch.exp(K + f[:, None] + g[None, :])

# -------------------- 2. Schrödinger Bridge 插值 (内部使用) ---------------- #
def schrodinger_bridge_interpolate(x0, x1, steps=10, sigma=0.1):
    """使用扩散桥采样模拟 Schrödinger Bridge 路径"""
    device = x0.device
    if x1.size(0) == 1 and x0.size(0) > 1:
        x1 = x1.expand(x0.size(0), -1)
    timesteps = torch.linspace(0, 1, steps, device=device)
    paths = []
    for t in timesteps:
        xt = (1 - t) * x0 + t * x1
        noise = torch.randn_like(xt) * sigma * torch.sqrt(t * (1 - t))
        paths.append(xt + noise)
    return torch.stack(paths)

# -------------------- 3. 对偶势网络 --------------------------- #
class PotentialNet(nn.Module):
    """对偶势网络"""
    def __init__(self, feat_dim, dom_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + dom_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x, dom_onehot):
        return self.net(torch.cat([x, dom_onehot], dim=1))

# -------------------- 3½. 类别重心网络 ------------------ #
class BarycenterNet(nn.Module):
    """根据类别标签生成目标重心"""
    def __init__(self, cls_dim, feat_dim, hidden=128):
        super().__init__()
        self.feat_dim = feat_dim # Store feat_dim for use in forward
        self.net = nn.Sequential(
            nn.Linear(cls_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim)
        )

    def forward(self, cls_onehot):
        # 确保输出维度正确，即使输入是 [1, cls_dim] 或 [B, cls_dim]
        return self.net(cls_onehot) # [B, feat_dim]

# -------------------- 4. Conditional BTM 模块 ---------------------- #
class ConditionalBTMAligner(nn.Module):
    """
    Conditional BTM + Schrödinger Bridge 特征对齐模块。

    该模块接收原始特征、域标签和类别标签，输出对齐后的特征。

    Args:
        feat_dim (int): 输入特征的维度。
        num_domains (int): 域的总数。
        num_classes (int): 离散化类别的总数。
        dual_lr (float, optional): 对偶优化的学习率。默认为 1e-3。
        dual_iter (int, optional): 对偶优化的迭代次数。默认为 40。
        sb_steps (int, optional): Schrödinger Bridge 插值的步数。默认为 5。
        sb_sigma (float, optional): Schrödinger Bridge 的噪声水平。默认为 0.1。
        train_alignment (bool, optional): 是否在前向传播时训练对齐器。
                                     如果为 False，则使用已训练的参数进行对齐。
                                     默认为 True。
    """
    def __init__(self, feat_dim, num_domains, num_classes,
                 dual_lr=1e-3, dual_iter=40,
                 sb_steps=5, sb_sigma=0.1,
                 train_alignment=True): # 新增参数
        super().__init__()
        self.feat_dim = feat_dim
        self.num_domains = num_domains
        self.num_classes = num_classes
        self.dual_iter = dual_iter
        self.sb_steps = sb_steps
        self.sb_sigma = sb_sigma
        self.train_alignment = train_alignment # Store the flag

        self.pnet = PotentialNet(feat_dim, num_domains)
        self.bnet = BarycenterNet(num_classes, feat_dim)
        
        # 优化器
        self.opt = optim.Adam(
            list(self.pnet.parameters()) + list(self.bnet.parameters()),
            lr=dual_lr
        )
        # 标志位，用于跟踪是否已进行过训练（当 train_alignment=False 时）
        self._is_trained = False 

    def _onehot(self, indices, n, device):
        """通用 one-hot 编码函数"""
        oh = torch.zeros(len(indices), n, device=device)
        oh[torch.arange(len(indices)), indices.long()] = 1
        return oh

    def _align_group(self, feats, dom_labels, cls_labels):
        """为一个类别组内的所有域进行对齐"""
        device = feats.device
        groups = {int(d): feats[dom_labels == d] for d in torch.unique(dom_labels)}
        
        if self.train_alignment or not self._is_trained:
            # 1) 训练对偶势网络 (同时优化 BarycenterNet)
            for _ in range(self.dual_iter):
                self.opt.zero_grad()
                # 计算对偶目标
                dual = sum(self.pnet(x, self._onehot(torch.full((len(x),), d), self.num_domains, device)).mean()
                        for d, x in groups.items())
                # 可选的 BarycenterNet 正则化
                bnet_reg_loss = 0.0
                total_loss = -dual + bnet_reg_loss
                total_loss.backward()
                self.opt.step()
            
            # 标记为已训练
            if not self._is_trained:
                self._is_trained = True

        # 2) 第一阶段：计算每个样本的去域化表示 (x - grad)
        nograd_features = torch.zeros_like(feats)
        for d, x in groups.items():
            idx = (dom_labels == d).nonzero(as_tuple=True)[0]
            x_req = x.clone().detach().requires_grad_(True)
            phi = self.pnet(x_req, self._onehot(torch.full((len(x_req),), d), self.num_domains, device)).sum()
            grad = torch.autograd.grad(phi, x_req, create_graph=False)[0]
            x_nograd = (x - grad).detach()
            nograd_features[idx] = x_nograd

        # 3) 第二阶段：类别引导对齐
        unique_cls = torch.unique(cls_labels)
        assert len(unique_cls) == 1, "Expected single class in _align_group"
        c = unique_cls.item()

        center_nograd = nograd_features.mean(dim=0, keepdim=True)
        target_c = self.bnet(self._onehot(torch.tensor([c]), self.num_classes, device))
        translation_vector = target_c - center_nograd
        final_targets = nograd_features + translation_vector

        # 4) 使用 Schrödinger Bridge 将原始特征对齐到个性化目标
        out = torch.zeros_like(feats)
        for d, x in groups.items():
            idx = (dom_labels == d).nonzero(as_tuple=True)[0]
            sample_targets = final_targets[idx]
            path = schrodinger_bridge_interpolate(
                x, sample_targets,
                steps=self.sb_steps, sigma=self.sb_sigma
            )
            out[idx] = path[-1]
        return out

    def forward(self, feats, dom_labels, cls_labels):
        """
        执行特征对齐。

        Args:
            feats (torch.Tensor): 输入特征, shape [N, feat_dim]。
            dom_labels (torch.Tensor): 域标签, shape [N], 类型为 LongTensor。
            cls_labels (torch.Tensor): 离散化的类别标签, shape [N], 类型为 LongTensor。

        Returns:
            torch.Tensor: 对齐后的特征, shape [N, feat_dim]。
        """
        # 输入验证
        assert feats.dim() == 2 and feats.size(1) == self.feat_dim, \
            f"feats must be [N, {self.feat_dim}], got {feats.shape}"
        assert dom_labels.dim() == 1 and dom_labels.size(0) == feats.size(0), \
            f"dom_labels must be [N], got {dom_labels.shape}"
        assert cls_labels.dim() == 1 and cls_labels.size(0) == feats.size(0), \
            f"cls_labels must be [N], got {cls_labels.shape}"
        assert torch.all((dom_labels >= 0) & (dom_labels < self.num_domains)), \
            "Domain labels out of range"
        assert torch.all((cls_labels >= 0) & (cls_labels < self.num_classes)), \
            "Class labels out of range"

        if self.train_alignment or not self._is_trained:
             # 如果需要训练或尚未训练，则设置为训练模式
            # （注意：这会影响模型中所有子模块的 train/eval 状态）
            # 如果只想训练内部参数而不影响外部，可以省略或谨慎使用
            # self.train() # 可选，取决于整体使用方式
            pass # 内部 _align_group 会处理训练逻辑
        else:
            # 如果不需要训练且已训练，则设置为评估模式
            # self.eval() # 可选
            pass

        out = torch.zeros_like(feats)
        # 对每个类别进行处理
        for c in torch.unique(cls_labels):
            cls_mask = (cls_labels == c)
            idx = cls_mask.nonzero(as_tuple=True)[0]
            # 对这些样本进行对齐
            out[idx] = self._align_group(feats[idx], dom_labels[idx], cls_labels[idx])
        
        # 如果模块整体处于训练模式，保持梯度；否则，可以detach（可选优化）
        # if not self.training:
        #     out = out.detach()
        return out

    def is_trained(self):
        """检查模块是否已进行过训练。"""
        return self._is_trained

# -------------------- 5. 简化封装函数 (可选) --------------------
# 如果你希望有一个更简单的函数接口，可以提供一个包装函数
def align_features(features, domain_labels, class_labels,
                   feat_dim, num_domains, num_classes,
                   dual_lr=1e-3, dual_iter=40,
                   sb_steps=5, sb_sigma=0.1,
                   train_alignment=True):
    """
    简化接口：使用 ConditionalBTMAligner 对特征进行对齐。

    Args:
        features (np.ndarray or torch.Tensor): 输入特征 [N, feat_dim]。
        domain_labels (np.ndarray or torch.Tensor): 域标签 [N]。
        class_labels (np.ndarray or torch.Tensor): 类别标签 [N]。
        ... (其他参数同 ConditionalBTMAligner)

    Returns:
        np.ndarray: 对齐后的特征 [N, feat_dim]。
    """
    # 1. 输入类型转换
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    if isinstance(domain_labels, np.ndarray):
        domain_labels = torch.from_numpy(domain_labels).long()
    if isinstance(class_labels, np.ndarray):
        class_labels = torch.from_numpy(class_labels).long()

    device = features.device if features.is_cuda else torch.device('cpu')
    features = features.to(device)
    domain_labels = domain_labels.to(device)
    class_labels = class_labels.to(device)

    # 2. 初始化模块
    aligner = ConditionalBTMAligner(
        feat_dim=feat_dim, num_domains=num_domains, num_classes=num_classes,
        dual_lr=dual_lr, dual_iter=dual_iter,
        sb_steps=sb_steps, sb_sigma=sb_sigma,
        train_alignment=train_alignment
    ).to(device)

    # 3. 执行对齐
    aligned_features = aligner(features, domain_labels, class_labels)

    # 4. 返回 numpy 数组
    return aligned_features.detach().cpu().numpy()
