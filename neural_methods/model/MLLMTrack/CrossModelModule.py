import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalModule(nn.Module):
    def __init__(self, num_queries=1, embed_dim=768, num_heads=4, ff_dim=768*2):
        super(CrossModalModule, self).__init__()
        
        # Learnable Queries
        self.learnable_token = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        # Self-Attention layers for ts and vf
        self.self_attention_ts = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_attention_vf = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Cross-Attention layers: learnable queries with ts and vf
        self.cross_attention_queries_ts = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attention_queries_vf = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Feed Forward Networks (FFN)
        self.ffn_ts = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ffn_vf = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization
        self.norm_ts = nn.LayerNorm(embed_dim)
        self.norm_vf = nn.LayerNorm(embed_dim)
        self.norm_queries = nn.LayerNorm(embed_dim)
    
    def forward(self, ts, vf):
        # ts: (batch, seq_len, dim)
        # vf: (batch, channel, dim)
        
        # Transpose to (seq_len, batch, dim) for MultiheadAttention
        ts = ts.transpose(0, 1)  # (seq_len, batch, dim)
        vf = vf.transpose(0, 1)  # (channel, batch, dim)
        
        # Self-attention for ts
        ts_self_attn, _ = self.self_attention_ts(ts, ts, ts)
        ts = self.norm_ts(ts + ts_self_attn)
        
        # Self-attention for vf
        vf_self_attn, _ = self.self_attention_vf(vf, vf, vf)
        vf = self.norm_vf(vf + vf_self_attn)
        
        # Cross-attention: learnable queries with ts
        queries = self.learnable_token.unsqueeze(1).expand(-1, ts.size(1), -1)  # Expand for batch size
        queries_to_ts, _ = self.cross_attention_queries_ts(queries, ts, ts)
        queries = self.norm_queries(queries + queries_to_ts)
        
        # Cross-attention: learnable queries with vf
        queries_to_vf, _ = self.cross_attention_queries_vf(queries, vf, vf)
        queries = self.norm_queries(queries + queries_to_vf)
        
        # FFN processing for ts and vf
        ts = self.norm_ts(ts + self.ffn_ts(ts))
        vf = self.norm_vf(vf + self.ffn_vf(vf))
        
        # Transpose back to original shape
        ts = ts.transpose(0, 1)  # (batch, seq_len, dim) ts: torch.Size([4, 16, 768])
        vf = vf.transpose(0, 1)  # (batch, channel, dim) vf: torch.Size([4, 96, 768])
        queries = queries.transpose(0, 1)
        return queries, ts, vf

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, E_p, E_g):
        # E_p: (batchsize, seqlen, dim)
        # E_g: (batchsize, seqlen, dim)
        batch_size = E_p.size(0)
        seq_len = E_p.size(1)
        total_loss = 0.0
        for i in range(batch_size):
            for j in range(seq_len):
                cosine_sim = torch.nn.functional.cosine_similarity(E_p[i, j], E_g[i, j], dim=0)
                total_loss += (1 - cosine_sim)
        return total_loss / (batch_size * seq_len)

