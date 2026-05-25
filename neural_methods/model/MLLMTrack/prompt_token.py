import torch
from torch import nn

class AttentiveCompressor(nn.Module):
    def __init__(self, hidden_dim, target_len):
        super().__init__()
        self.target_len = target_len
        self.hidden_dim = hidden_dim

        # 可学习的查询向量
        self.query = nn.Parameter(torch.randn(1, target_len, hidden_dim))

        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        输入:
        - x: [batch_size, seq_len, hidden_dim]
        输出:
        - compressed: [batch_size, target_len, hidden_dim]
        """
        batch_size = x.size(0)

        # 扩展查询向量到批次维度
        query = self.query.expand(batch_size, -1, -1)

        # 通过注意力机制压缩序列
        compressed, _ = self.attention(query, x, x)
        compressed = self.layer_norm(compressed)

        return compressed