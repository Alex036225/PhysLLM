from math import sqrt
# from neural_methods.model.MLLMTrack.Qformer import QFormer
from neural_methods.model.MLLMTrack.feature_fusion import MMFusion, MultiScaleFeatureFusion, MultiAverageFeatureFusion
from neural_methods.model.PhysLLMModules.processors.video_encoder import CLIPVideoEncoder, EfficientPhysVideoEncoder, PhysFormerVideoEncoder, PhysNetVideoEncoder, PhysformerCLIP
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoConfig, AutoModelForCausalLM ,  AutoTokenizer, AutoModel
from neural_methods.model.PhysLLMModules.TimeLLM.layers.Embed import PatchEmbedding
from neural_methods.model.PhysLLMModules.TimeLLM.layers.StandardNorm import Normalize
from peft import get_peft_model, LoraConfig
from pytorch_wavelets import DWT1D, IDWT1D
from neural_methods.model.MLLMTrack.CrossModelModule import CrossModalModule, CosineSimilarityLoss

# =========================================================================================
# ======================== 新增模块 (MoE, PTT, Auxiliary) ===============================
# =========================================================================================

class Expert(nn.Module):
    """一个简单的MLP专家网络"""
    def __init__(self, input_dim, output_dim=5): # HR, RR, SpO2, BP_sys, BP_dia
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    """门控网络，决定每个专家的权重"""
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)

class MoEHead(nn.Module):
    """混合专家预测头"""
    def __init__(self, input_dim, output_dim=5, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        gating_weights = self.gating(x)  # (batch_size, num_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # (batch_size, num_experts, output_dim)
        
        # 加权平均
        # gating_weights.unsqueeze(-1): (batch_size, num_experts, 1)
        # final_output: (batch_size, output_dim)
        final_output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return final_output

class AuxiliaryFreqHead(nn.Module):
    """辅助任务头，用于预测频域谱"""
    def __init__(self, input_dim, freq_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, freq_dim)
        )
    def forward(self, x):
        return self.net(x)

class PseudoPTTModule(nn.Module):
    """伪PTT模块，从两个视觉特征图中估计PTT"""
    def __init__(self, feature_channels, token_dim):
        super().__init__()
        # 假设我们使用两个尺度的特征图, e.g., 64x64 and 16x16
        self.fusion = MMFusion(feature_channels, token_dim)
        self.ptt_estimator = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 输出单个PTT值
        )
    def forward(self, features, target_len):
        fused_features = self.fusion(features, target_len) # (B, T, Dim)
        # 使用最后一个时间步的特征来估计PTT
        ptt_feature = fused_features[:, -1, :] 
        pseudo_ptt_value = self.ptt_estimator(ptt_feature)
        return pseudo_ptt_value

# =========================================================================================
# =========================================================================================

class NewPhysLLM(nn.Module):
    def __init__(self, configs,):
        super(NewPhysLLM, self).__init__()
        
        self.video_encoder_type = configs.MODEL.VIDEO_ENC
        self.video_encoder = self.get_video_encoder(self.video_encoder_type)
        self.d_llm = 1536 # deepseek
        self.patch_len = 16
        self.stride = 8
        llm_layers = 32
        dropout = 0.1
        self.rPPG_based_encoder = ["PhysMamba", "PhysFormer", "PhysNet", "PhysFormerCLIP", "EfficientPhys"]
        self.enc_type = configs.MODEL.VIDEO_ENC
        enc_in = 1

        if configs.MODEL.LLM == 'DeepSeek':
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            self.deepseek_config = AutoConfig.from_pretrained(model_name)
            self.deepseek_config.num_hidden_layers = 32
            self.deepseek_config.output_attentions = True
            self.deepseek_config.output_hidden_states = True
            try:
                self.llm_model = AutoModel.from_pretrained(
                    model_name, trust_remote_code=True, local_files_only=False,
                    torch_dtype=torch.float32, config=self.deepseek_config,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True, local_files_only=False
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    model_name, trust_remote_code=True, local_files_only=False, config=self.deepseek_config
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True, local_files_only=False
                )
        else:
            raise Exception('LLM model must be DeepSeek for this architecture')
        
        self.lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none")
        self.llm_LoRA = get_peft_model(self.llm_model, self.lora_config)
        self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
        
        self.dropout = nn.Dropout(dropout)
        self.patch_embedding = PatchEmbedding(self.d_llm, 8, 4, dropout)

        self.normalize_layers = Normalize(enc_in, affine=False)
        self.tsmoudle = TimeSeriesMoudle(configs.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)

        # --- 新的模块实例化 ---
        self.target_sequence_length = 32  # 特征融合的目标长度
        
        # 主rPPG信号的特征融合模块
        self.feature_fusion_main = MMFusion(
            feature_channels=[64, 64, 64], token_dim=self.d_llm
        )
        # 用于PTT的特征融合及估算模块
        self.ptt_module = PseudoPTTModule(
            feature_channels=[64, 16], token_dim=self.d_llm # 假设使用不同尺度的特征
        )

        # 主预测头 (MoE)
        # 输入维度 = LLM输出维度 + PTT特征维度(1)
        self.moe_head = MoEHead(input_dim=self.d_llm + 1, output_dim=5, num_experts=8)

        # 辅助任务头 (频域)
        # 输入维度 = LLM输出维度
        # 输出维度 = 频域谱长度, e.g., chunk_len/2 + 1 for rfft
        self.aux_freq_head = AuxiliaryFreqHead(input_dim=self.d_llm, freq_dim=configs.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH // 2 + 1)
        
        self.configs = configs

    def forward(self, data, prompt):
        # 1. Video Encoder: 提取rPPG信号和多尺度视觉特征
        if self.configs.MODEL.VIDEO_ENC == "PhysNet":
            x_enc, x_visual64, x_visual32, x_visual16  = self.video_encoder.encode(data)
        elif self.configs.MODEL.VIDEO_ENC == "EfficientPhys":
            x_enc, x_visual16, x_visual32, x_visual64 = self.video_encoder.encode(data)
        elif self.configs.MODEL.VIDEO_ENC == "PhysFormer":
            x_enc, x_visual64, x_visual32, x_visual16 = self.video_encoder.encode(data)
        else:
            raise NotImplementedError(f"Encoder {self.configs.MODEL.VIDEO_ENC} not supported for this architecture")
        
        x_enc = self.tsmoudle(x_enc)

        # 2. 特征处理与融合
        # 2.1 主rPPG路径的视觉特征融合
        x_fusion_main = self.feature_fusion_main(
            features=[x_visual64, x_visual32, x_visual16],
            target_len=self.target_sequence_length
        )

        # 2.2 PTT路径的视觉特征融合与估算
        # 使用不同尺度的特征来模拟不同面部区域
        pseudo_ptt_value = self.ptt_module(
            features=[x_visual64, x_visual16], # 使用差异大的特征图
            target_len=self.target_sequence_length
        )

        # 3. LLM输入准备
        x_enc = x_enc.unsqueeze(-1)
        x_enc = self.normalize_layers(x_enc, 'norm')
        x_enc, n_vars = self.patch_embedding(x_enc)  # (B, Patch_Num, d_llm)

        # 拼接rPPG patch特征和主视觉融合特征
        llama_enc_out = torch.cat([x_enc, x_fusion_main], dim=1)

        # 4. LLM前向传播
        llm_output = self.llm_LoRA(inputs_embeds=llama_enc_out).last_hidden_state  # (B, total_seq_len, d_llm)

        # 5. 特征聚合
        # 使用最后一个时间步的隐藏状态作为聚合特征
        aggregated_feature = llm_output[:, -1, :] # (B, d_llm)

        # 6. 主任务预测 (MoE Head)
        # 6.1 拼接PTT特征
        final_feature_for_pred = torch.cat([aggregated_feature, pseudo_ptt_value], dim=1) # (B, d_llm + 1)
        # 6.2 通过MoE头得到最终预测值
        main_prediction = self.moe_head(final_feature_for_pred) # (B, 5) -> (HR, RR, SpO2, Sys, Dia)

        # 7. 辅助任务预测 (Freq Head)
        # 使用不含PTT的聚合特征
        aux_prediction = self.aux_freq_head(aggregated_feature) # (B, freq_dim)
        
        # 8. 返回主任务和辅助任务的预测结果
        return main_prediction, aux_prediction


    def get_video_encoder(self, video_encoder_type):
        if video_encoder_type == "PhysFormer":
            return PhysFormerVideoEncoder()
        elif video_encoder_type == "PhysNet":
            return PhysNetVideoEncoder()
        elif video_encoder_type == "EfficientPhys":
            return EfficientPhysVideoEncoder()
        else:
            raise NotImplementedError(f"Video encoder '{video_encoder_type}' is not implemented.")

class TimeSeriesMoudle(nn.Module):
    def __init__(self, seq_len, kernel_len=10, wavelet="coif6", j=3):
        super(TimeSeriesMoudle, self).__init__()
        self.seq_len = seq_len
        self.dwt = DWT1D(wave=wavelet, J=j)
        self.idwt = IDWT1D(wave=wavelet)
        self.dwt_ratio = nn.Parameter(torch.full((1, 1), 0.5))

    def forward(self, x):
        norm_x,_,_ = self._norm(x)
        norm_x = self._ewma_filter(norm_x)
        ac, dc_list = self.dwt(x.unsqueeze(1))
        ac = ac.squeeze(1)
        norm_ac, _, _ = self._norm(ac)
        norm_ac = self._ewma_filter(norm_ac)
        norm_dc = []
        for dc in dc_list:
            dc = dc.squeeze(1)
            norm_dc_part, _, _ = self._norm(dc)
            norm_dc_part = self._ewma_filter(norm_dc_part)
            norm_dc.append(norm_dc_part)
        freq_x = self.idwt([norm_ac.unsqueeze(1), [d.unsqueeze(1) for d in norm_dc]]).squeeze(1)
        dwt_r, time_r = self.dwt_ratio, 1 - self.dwt_ratio
        combined_x = norm_x * time_r + freq_x * dwt_r
        return combined_x

    def _norm(self, x):
        sample_mean = torch.mean(x, dim=1, keepdim=True)
        sample_std = torch.std(x, dim=1, keepdim=True)
        x = (x - sample_mean) / (sample_std + 1e-8)
        return x, sample_mean, sample_std

    def _ewma_filter(self, y, alpha=0.8):
        z = torch.zeros_like(y)
        if y.size(0) > 0:
            z[0] = y[0]
            for i in range(1, len(y)):
                z[i] = alpha * y[i] + (1 - alpha) * z[i-1]
        return z

# 以下模块定义保持不变或已在上面集成，为简洁省略
# CATBlock1, CATBlock, z_score_normalize
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x