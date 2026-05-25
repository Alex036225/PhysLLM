import torch
import torch.nn as nn
import torch.nn.functional as F

class MMFuser(nn.Module):
    def __init__(self, dim, heads=8):
        super(MMFuser, self).__init__()

        # Cross Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)

        # Self Attention
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)


        # Learnable vectors for scaling
        self.gamma1 = nn.Parameter(torch.zeros(dim))
        self.gamma2 = nn.Parameter(torch.zeros(dim))

    def forward(self, X, query_feature):

        X = X.permute(1, 0, 2)
        query_feature = query_feature.permute(1, 0, 2)

        # Cross Attention: Query from deep features (FL) and Key/Value from shallow features (X)
        cross_attn_output, _ = self.cross_attention(query_feature, X, X)

        # Self Attention: Apply self attention on the cross attention output
        self_attn_output, _ = self.self_attention(cross_attn_output, cross_attn_output, cross_attn_output)

        # Combine cross-attention and self-attention outputs
        fused_output = cross_attn_output + self.gamma2 * self_attn_output

        # Final fusion of deep features FL and the output of attention mechanisms
        final_output = query_feature + self.gamma1 * fused_output

        return final_output.permute(1, 0, 2)
