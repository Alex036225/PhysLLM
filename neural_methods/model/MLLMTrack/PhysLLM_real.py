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


class PhysLLM_real(nn.Module):
    """
    PhysLLM model for Text Generation (QA).
    Modifications:
    1. Uses Learnable Query Tokens instead of compressed word embeddings.
    2. Removes explicit statistics (min/max) from text prompts.
    3. Corrects input ordering: [Visual Features, Text Prompt].
    """
    def __init__(self, configs,):
        super(PhysLLM_real, self).__init__()
        self.video_encoder_type = configs.MODEL.VIDEO_ENC
        self.video_encoder = self.get_video_encoder(self.video_encoder_type)
        self.pred_len = configs.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.d_ff = 128
        self.top_k = 5
        self.patch_len = 16
        self.stride = 8
        llm_layers = 32
        dropout = 0.1
        seq_len = 512
        
        # --- Encoder Settings ---
        self.rPPG_based_encoder = ["PhysMamba", "PhysFormer", "PhysNet", "PhysFormerCLIP", "EfficientPhys"]
        self.feature_based_encoder = ["clip"]
        self.enc_type = configs.MODEL.VIDEO_ENC
        enc_in = 1 if configs.MODEL.VIDEO_ENC in self.rPPG_based_encoder else 512
        
        # --- LLM Initialization (保持原有逻辑) ---
        self.is_sundial = False
        if configs.MODEL.LLM == 'LLAMA':
            self.d_llm = 4096
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = llm_layers
            self.llm_model = AutoModelForCausalLM.from_pretrained(               
                'huggyllama/llama-7b', trust_remote_code=True, config=self.llama_config)
            self.tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b', trust_remote_code=True)
            
        elif configs.MODEL.LLM == 'GPT2':
            self.d_llm = 768
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = llm_layers
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                'openai-community/gpt2', trust_remote_code=True, config=self.gpt2_config)
            self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2', trust_remote_code=True)
            
        elif configs.MODEL.LLM == 'DeepSeek':
            self.d_llm = 1536
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            self.deepseek_config = AutoConfig.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, config=self.deepseek_config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        else:
            raise Exception('LLM model is not defined or supported for this version.')

        # --- LoRA Setup ---
        target_modules = None 
        # DeepSeek/Qwen usually needs specific target modules, but generic works for many
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM", 
        )
        self.llm_LoRA = get_peft_model(self.llm_model, self.lora_config)

        # --- Tokenizer Adjustment ---
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
            if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'

        self.dropout = nn.Dropout(dropout)
        self.patch_embedding = PatchEmbedding(self.d_llm, 8, 4, dropout) # Note: checks input dim
        
        # Handle word embeddings for Prompt (Text)
        if hasattr(self, 'is_sundial') and self.is_sundial:
             # ... (Sundial embedding handling same as before)
             self.word_embeddings = nn.Parameter(torch.randn(50257, self.d_llm)) # Fallback
        else:
            self.word_embeddings = self.llm_model.get_input_embeddings().weight

        # --- [CRITICAL CHANGE 1] Learnable Queries instead of Mapping Layer ---
        # 移除了 self.mapping_layer
        # 定义对齐模块的维度
        self.reduced_dim = 512 
        self.num_query_tokens = 32 
        
        # 初始化 Learnable Queries (Q-Former style queries)
        # 这些向量将作为 CATBlock 的 Query，去“查询”视觉特征
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_query_tokens, self.reduced_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.patch_nums = int((seq_len - self.patch_len) / self.stride + 2)
        
        if self.enc_type in self.rPPG_based_encoder:
            self.input_type = 'rPPG_sequence'
        elif self.enc_type in self.feature_based_encoder:
            self.input_type = 'video_feature'
        else:
            raise ValueError(f"Invalid encoder type: {self.enc_type}")
        
        self.normalize_layers = Normalize(enc_in, affine=False)
        self.tsmoudle = TimeSeriesMoudle(self.d_ff)
        
        # Feature Fusion
        self.feature_fusion = MMFusion(
            feature_channels=[64, 64, 64],
            token_dim=self.d_llm,
            bottleneck_dim=self.reduced_dim
        )
        
        self.target_sequence_length = 32
        self.configs = configs
        self.loss_function = CosineSimilarityLoss() # Can be changed to CrossEntropy for pure text gen training
        
        # Dimension Adapters
        self.dim_reduction = nn.Linear(self.d_llm, self.reduced_dim)
        self.dim_increase = nn.Linear(self.reduced_dim, self.d_llm)
        
        # Attention Blocks
        # CATBlock1: Cross Attention between [Queries] and [Visual Features]
        self.catblock = CATBlock1(self.reduced_dim, 4, self.reduced_dim * 2)
        
        # Text Generation Parameters
        self.max_new_tokens = 256
        self.temperature = 0.7
        self.do_sample = True

    def forward(self, data, prompt, generate_text=True, max_new_tokens=None, temperature=None, do_sample=None):
        # 1. Video Encoding (Same as before)
        if self.configs.MODEL.VIDEO_ENC == "PhysNet":
            x_enc, x_visual6464, x_visual3232, x_visual1616  = self.video_encoder.encode(data)
            x_enc = self.tsmoudle(x_enc)
            x_fusion = self.feature_fusion(
                features=[x_visual6464, x_visual3232, x_visual1616],
                target_len=self.target_sequence_length
            )
        elif self.configs.MODEL.VIDEO_ENC == "EfficientPhys":
            x_enc, x_visual1616, x_visual3232, x_visual6464 = self.video_encoder.encode(data)
            x_enc = self.tsmoudle(x_enc)
            x_fusion = self.feature_fusion(
                features=[x_visual1616, x_visual3232, x_visual6464],
                target_len=self.target_sequence_length
            )
        elif self.configs.MODEL.VIDEO_ENC == "PhysFormer":
            x_enc, Trans_features, Trans_features2, Trans_features3 = self.video_encoder.encode(data)
            x_enc = self.tsmoudle(x_enc)
            x_fusion = self.feature_fusion(
                features=[Trans_features, Trans_features2, Trans_features3],
                target_len=self.target_sequence_length
            )        
        else:
            raise "please specify encoder fusion dim"
        
        if self.input_type == 'rPPG_sequence':
            if generate_text:
                return self.process_and_generate_text(x_enc, prompt, x_fusion, prompt_use=True,
                                                      max_new_tokens=max_new_tokens, 
                                                      temperature=temperature, 
                                                      do_sample=do_sample)
            else:
                return self.process_rppg_sequence(x_enc, prompt, x_fusion, prompt_use=True)

    def get_prompt_embeddings(self, prompts, device):
        """Helper to get text embeddings safely across models"""
        prompt_tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(device)
        
        if hasattr(self, 'is_sundial') and self.is_sundial:
            try:
                if hasattr(self.llm_LoRA, 'get_input_embeddings'):
                    return self.llm_LoRA.get_input_embeddings()(prompt_tokens)
                else:
                    return F.embedding(prompt_tokens, self.word_embeddings)
            except:
                return F.embedding(prompt_tokens, self.word_embeddings)
        else:
            return self.llm_LoRA.get_input_embeddings()(prompt_tokens)

    def process_and_generate_text(self, x_enc, prompt, x_fusion, prompt_use=False, 
                                  max_new_tokens=None, temperature=None, do_sample=None):
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        do_sample = do_sample if do_sample is not None else self.do_sample
        
        # --- Preprocessing Visual Features ---
        x_enc = x_enc.unsqueeze(-1)
        B, T, N = x_enc.size()
        x_enc = self.normalize_layers(x_enc, 'norm')
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1) # [B, T, 1]
        
        # --- [CRITICAL CHANGE 2] Constructing Clean Prompts ---
        # Removed min/max/lags calculation
        prompts = []
        # General system instruction for the LLM
        system_prompt = "Analyze the physiological signals and answer the following question:"
        
        for b in range(B):
            user_q = prompt[b] if prompt is not None and len(prompt) > b else "Describe the signal."
            # Simple prompt structure: Instruction -> User Input
            prompt_text = f"{system_prompt}\nUser: {user_q}\nAssistant:"
            prompts.append(prompt_text)
            
        # --- Visual Encoding (Patch Embedding) ---
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous() # Back to [B, 1, T] ? 
        x_enc = x_enc.permute(0, 2, 1).contiguous() # [B, T, 1]
        x_enc, n_vars = self.patch_embedding(x_enc)  # [B, 32, d_llm] (Note: Check patch_embedding output dim)
        
        # --- [CRITICAL CHANGE 3] Visual Feature Alignment ---
        # 1. Expand Query Tokens to batch size
        queries = self.query_tokens.expand(B, -1, -1) # [B, 32, 512]
        
        # 2. Reduce Visual Features dimension to match Queries (if needed)
        x_enc_reduced = self.dim_reduction(x_enc)       # [B, 32, 512]
        x_fusion_reduced = self.dim_reduction(x_fusion) # [B, 32, 512]
        
        # 3. Cross Attention: Queries attend to Visual Features
        # The queries extract information from x_enc and x_fusion
        ts_we_reduced = self.catblock(queries, x_enc_reduced)     # Querying Raw Signal Features
        xf_we_reduced = self.catblock(queries, x_fusion_reduced)  # Querying Fused/Video Features
        
        # 4. Project back to LLM dimension
        # These are now "Visual Prompt Embeddings"
        ts_we = self.dim_increase(ts_we_reduced) # [B, 32, d_llm]
        xf_we = self.dim_increase(xf_we_reduced) # [B, 32, d_llm]
        
        # --- Get Text Embeddings ---
        text_embeddings = self.get_prompt_embeddings(prompts, x_enc.device) # [B, text_len, d_llm]
        
        # --- [CRITICAL CHANGE 4] Concatenation Order ---
        # Standard Multimodal Causal Order: [Visual Features, Text Prompt]
        # We assume the user wants the model to see the signal, then read the question.
        # Note: Depending on training, you might want just xf_we or both. Using both here.
        
        # visual_context = torch.cat([ts_we, xf_we], dim=1) # [B, 64, d_llm]
        # inputs_embeds = torch.cat([visual_context, text_embeddings], dim=1) 
        
        inputs_embeds = torch.cat([ts_we, xf_we, text_embeddings], dim=1)
        
        # --- Generate ---
        generated_texts = []
        for batch_idx in range(B):
            sample_embeds = inputs_embeds[batch_idx:batch_idx+1]
            try:
                with torch.no_grad():
                    generated_ids = self.llm_LoRA.generate(
                        inputs_embeds=sample_embeds,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                # Decode
                # Note: We need to handle cutting off the input prompt if generate returns full seq
                # But here we passed inputs_embeds, generate() output usually starts from 1st generated token
                # or full sequence depending on version. Usually full sequence.
                # However, we don't have input_ids length because we used embeds.
                # A robust way is to rely on the fact that generate() with inputs_embeds returns generated tokens.
                # Let's decode everything and user can parse.
                text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                generated_texts.append(text)
            except Exception as e:
                generated_texts.append(f"Error: {str(e)}")
                
        return generated_texts

    def process_rppg_sequence(self, x_enc, prompt, x_fusion, prompt_use=False):
        """
        Modified to return hidden states for training, matching the new architecture.
        """
        # ... (Preprocessing consistent with above) ...
        x_enc = x_enc.unsqueeze(-1)
        B, T, N = x_enc.size()
        x_enc = self.normalize_layers(x_enc, 'norm')
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # Simple Prompt Construction for Training
        prompts = []
        system_prompt = "Analyze the physiological signals and answer the following question:"
        for b in range(B):
            user_q = prompt[b] if prompt is not None and len(prompt) > b else "Describe the signal."
            prompt_text = f"{system_prompt}\nUser: {user_q}\nAssistant:"
            prompts.append(prompt_text)
            
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc, n_vars = self.patch_embedding(x_enc)
        
        # --- Visual Alignment (Same as above) ---
        queries = self.query_tokens.expand(B, -1, -1)
        
        x_enc_reduced = self.dim_reduction(x_enc)
        x_fusion_reduced = self.dim_reduction(x_fusion)
        
        ts_we_reduced = self.catblock(queries, x_enc_reduced)
        xf_we_reduced = self.catblock(queries, x_fusion_reduced)
        
        ts_we = self.dim_increase(ts_we_reduced)
        xf_we = self.dim_increase(xf_we_reduced)
        
        # --- Text Embeddings ---
        text_embeddings = self.get_prompt_embeddings(prompts, x_enc.device)
        
        # --- Concatenation (Visual -> Text) ---
        inputs_embeds = torch.cat([ts_we, xf_we, text_embeddings], dim=1)
        
        # --- Forward Pass (Get Hidden States) ---
        # Note: For Causal LM training, you usually provide labels shifted by 1.
        # Here we just return the output states/logits.
        outputs = self.llm_LoRA(inputs_embeds=inputs_embeds, output_hidden_states=True)
        
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        return outputs[0] # Tuple fallback
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

class TimeSeriesMoudle(nn.Module):
    def __init__(self, seq_len, kernel_len=10, wavelet="coif6", j=3):
        super(TimeSeriesMoudle, self).__init__()
        self.seq_len = seq_len
        self.kernel_len = kernel_len
        self.epsilon = 1e-5
        self.pad = nn.ReplicationPad1d(padding=(kernel_len // 2, kernel_len // 2 - ((kernel_len + 1) % 2)))
        self.dwt = DWT1D(wave=wavelet, J=j)
        self.idwt = IDWT1D(wave=wavelet)
        self.dwt_ratio = nn.Parameter(
            torch.clamp(torch.full((1, 1), 0.5), min=0., max=1.)
        )
    def forward(self, x):
        norm_x,_,_ = self._norm(x)
        norm_x = self._ewma_filter(norm_x)
        ac, dc_list = self.dwt(x.unsqueeze(1))
        ac = ac.squeeze(1)
        norm_ac, mac, sac = self._norm(ac)
        norm_ac = self._ewma_filter(norm_ac)
        norm_dc, m_list, s_list = [], [], []
        for dc in dc_list:
            dc = dc.squeeze(1)
            norm_dc_part, mdc, sdc = self._norm(dc)
            norm_dc_part = self._ewma_filter(norm_dc_part)
            norm_dc.append(norm_dc_part)
            m_list.append(mdc)
            s_list.append(sdc)
        freq_x = self.idwt([norm_ac.unsqueeze(1), [d.unsqueeze(1) for d in norm_dc]]).squeeze(1)
        dwt_r, time_r = self.dwt_ratio, 1 - self.dwt_ratio
        combined_x = norm_x * time_r + freq_x * dwt_r
        return combined_x
    def _norm(self, x):
        sample_mean = torch.mean(x, dim=(1), keepdim=True)
        sample_std = torch.std(x, dim=(1), keepdim=True)
        x = (x - sample_mean) / sample_std
        return x, sample_mean, sample_std
    def _ewma_filter(self, y, alpha=0.8):
        z = torch.zeros_like(y)
        z[0] = y[0]
        for i in range(1, len(y)):
            z[i] = alpha * y[i] + (1 - alpha) * z[i-1]
        return z

class CATBlock1(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super(CATBlock1, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, d_model)
        )
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
    def forward(self, query, key_value, mask=None):
        query = query.transpose(0, 1)
        key_value = key_value.transpose(0, 1)
        attn_output, _ = self.self_attn(key_value, key_value, key_value, key_padding_mask=mask)
        query = self.layer_norm1(query + attn_output)
        cross_attn_output, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=mask)
        query = self.layer_norm2(query + cross_attn_output)
        ffn_output = self.ffn(query)
        output = self.layer_norm3(query + ffn_output)
        output = output.transpose(0, 1)
        return output

class CATBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super(CATBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
    def forward(self, query, key_value, mask=None):
        query = query.transpose(0, 1)
        key_value = key_value.transpose(0, 1)
        attn_output, _ = self.self_attn(key_value, key_value, key_value, key_padding_mask=mask)
        key_value = self.layer_norm1(key_value + attn_output)
        cross_attn_output, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=mask)
        key_value = self.layer_norm2(key_value + cross_attn_output)
        ffn_output = self.ffn(key_value)
        output = self.layer_norm3(key_value + ffn_output)
        output = output.transpose(0, 1)
        return output

def z_score_normalize(tensor):
    mean = tensor.mean(dim=(1, 2), keepdim=True)
    std = tensor.std(dim=(1, 2), keepdim=True, unbiased=False)
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

if __name__ == "__main__":
    import yaml
    import torch
    from thop import profile, clever_format
    from easydict import EasyDict
    # --- 从 physllm.yaml 加载配置 ---
    with open("configs/physllm_only_test_zpu_example.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    configs = EasyDict(cfg_dict)
    # 创建并切换到评估模式
    model = PhysLLM_real(configs)
    model.eval()
    # 构造假输入，根据模型实际输入需求调整形状
    dummy_data = torch.randn(1, 3, 128, 128, 128)
    dummy_prompt = ["Test prompt"]
    # 测试文本生成
    with torch.no_grad():
        generated_texts = model(dummy_data, dummy_prompt, generate_text=True)
        print("Generated texts:")
        for i, text in enumerate(generated_texts):
            print(f"Sample {i}: {text}")
