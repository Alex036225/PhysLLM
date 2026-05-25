from math import sqrt
from neural_methods.model.MLLMTrack.feature_fusion import MultiScaleFeatureFusion
from neural_methods.model.PhysLLMModules.processors.video_encoder import CLIPVideoEncoder, PhysFormerVideoEncoder, PhysMambaVideoEncoder, PhysNetVideoEncoder, PhysformerCLIP
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from neural_methods.model.PhysLLMModules.TimeLLM.layers.Embed import PatchEmbedding
import transformers
from neural_methods.model.PhysLLMModules.TimeLLM.layers.StandardNorm import Normalize
from peft import get_peft_model, LoraConfig
from pytorch_wavelets import DWT1D, IDWT1D
from neural_methods.model.MLLMTrack.CrossModelModule import CrossModalModule, CosineSimilarityLoss

class NewPhysLLM(nn.Module):
    def __init__(self, configs,):
        super(NewPhysLLM, self).__init__()
        
        self.video_encoder_type = configs.MODEL.VIDEO_ENC
        self.video_encoder = self.get_video_encoder(self.video_encoder_type)  # "rPPG" or "VideoEmb"
        self.pred_len = configs.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH  # configs.pred_len
        # self.seq_len = configs.seq_len
        self.d_ff = 128  # configs.d_ff
        self.top_k = 5
        # TODO:BERT改为768
        # self.d_llm = 4096  # configs.llm_dim
        self.d_llm = 768  # configs.llm_dim
        self.patch_len = 16  # configs.patch_len
        self.stride = 8  # configs.stride
        llm_layers = 32
        dropout = 0.1
        n_heads = 8
        seq_len = 512
        self.rPPG_based_encoder = ["PhysMamba", "PhysFormer", "PhysNet", "PhysFormerCLIP"]
        self.feature_based_encoder = ["clip"]
        self.enc_type = configs.MODEL.VIDEO_ENC
        enc_in = 1 if configs.MODEL.VIDEO_ENC in self.rPPG_based_encoder else 512  # 512 is the feature dimension in clip
        d_model = 32 if configs.MODEL.VIDEO_ENC in self.rPPG_based_encoder else 512  # 512 is the feature dimension in clip

        if configs.MODEL.LLM == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(               
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
            except EnvironmentError: 
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.MODEL.LLM == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.MODEL.LLM == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )

            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')
        
        # # TODO:加入LoRA
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ2SEQ_LM",
        )
        self.llm_LoRA = get_peft_model(self.llm_model, self.lora_config)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        self.dropout = nn.Dropout(dropout)

        # self.patch_embedding = PatchEmbedding(
        #     d_model, self.patch_len, self.stride, dropout)
        self.patch_embedding = PatchEmbedding(
            768, self.patch_len, self.stride, dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight

        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(d_model, n_heads, self.d_ff, self.d_llm)  # (32, 8, 128, 768)

        self.patch_nums = int((seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(enc_in, self.head_nf, self.pred_len, head_dropout=dropout)

        if self.enc_type in self.rPPG_based_encoder:
            self.input_type = 'rPPG_sequence'
        elif self.enc_type in self.feature_based_encoder:
            self.input_type = 'video_feature'
            feature_dim = ...  # 您需要根据实际特征维度设置
        else:
            raise ValueError(f"Invalid encoder type: {self.enc_type}")
        self.normalize_layers = Normalize(enc_in, affine=False)
        # TODO
        self.tsencoder = TimeFreqStationary(self.d_ff, 5)
        self.tsmoudle = TimeSeriesMoudle(self.d_ff, 12)
        # todo: 定义了一个多尺度融合特征
        self.feature_fusion = MultiScaleFeatureFusion(
            feature_channels=[64, 64, 64],  # 对应三个特征图的通道数
            token_dim=self.d_llm # 768
        )
        # # target_len应该与LLM的token序列长度匹配
        self.target_sequence_length = 32  # 举例,需要根据实际情况设置
        self.configs = configs
        self.projector = nn.Linear(756, 756)
        # todo:定义一个可学习的token
        self.learnable_token = nn.Parameter(torch.randn(1, 1, self.d_llm))
        self.crossmodule = CrossModalModule(num_queries=16, embed_dim=768, num_heads=4, ff_dim=768*2)
        self.loss_function = CosineSimilarityLoss()



    def forward(self, data, prompt):
        if self.configs.MODEL.VIDEO_ENC == "PhysNet":
            x_enc, x_visual6464, x_visual3232, x_visual1616  = self.video_encoder.encode(data)
            x_fusion = self.feature_fusion(
                features=[x_visual6464, x_visual3232, x_visual1616],
                target_len=self.target_sequence_length
            )

            x_enc = self.tsmoudle(x_enc)
            # TODO:2025.1.12
            # x_fusion = x_fusion.permute(1, 0, 2)
            # multihead_attention = nn.MultiheadAttention(x_fusion.shape[-1], 4).cuda()
            # x_fusion_attention, attention_weights = multihead_attention(x_fusion, x_fusion, x_fusion)
            # x_fusion_attention = x_fusion_attention.permute(1, 0, 2)
        else:
            raise "please specify encoder fusion dim"
        
        if self.input_type == 'rPPG_sequence':
            dec_out, loss = self.process_rppg_sequence(x_enc, prompt,x_fusion)
            # original_rppg = x_enc
            modified_rppg = dec_out.squeeze(-1)

        elif self.input_type == 'video_feature':
            dec_out = self.process_video_feature(x_enc, prompt)
            original_rppg = torch.tensor([0], device=x_enc.device)
            modified_rppg = dec_out.squeeze(-1)
        else:
            raise ValueError(f"Invalid input type: {self.input_type}")
        
        return modified_rppg, loss  # [:, -self.pred_len:, :]


        
    def process_rppg_sequence(self, x_enc, prompt, x_fusion):
        """
        处理 rPPG 序列信号，根据 prompt 进行微调或调整。
        """

        x_enc = x_enc.unsqueeze(-1)
        B, T, N = x_enc.size()
        x_enc = self.normalize_layers(x_enc, 'norm')
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 生成任务描述
        Task = "Remote Photoplethysmography (rPPG) signals exhibit domain differences across different datasets, primarily manifested in factors such as the subject's skin color, ethnicity, gender, age, and environmental lighting conditions."
        # Task = "Please predict the Remote Photoplethysmography (rPPG) signals."
        # 构建完整的 prompt
        prompts = []
        for b in range(B):
            min_values = torch.min(x_enc, dim=1)[0]
            max_values = torch.max(x_enc, dim=1)[0]
            medians = torch.median(x_enc, dim=1).values
            lags = self.calcute_lags(x_enc)
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            trends = x_enc.diff(dim=1).sum(dim=1)

            Statistic = f"Input statistics: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}"

            prompt_text = (
                # f"Dataset description: This is a dataset about rPPG signal."
                f"Task description: {Task}"
                f"Face description:{prompt[b]}"
                f"{Statistic}"

            )
 
            prompts.append(prompt_text)
            
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc, n_vars = self.patch_embedding(x_enc) # [4, 16, 32]
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # x_enc = self.reprogramming_layer(x_enc, source_embeddings, source_embeddings) # [4, 16, 768]
        prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(0)
        # TODO
        queries, ts, vf = self.crossmodule(x_enc, x_fusion)
        loss = self.loss_function(ts, vf)

        # 2. 嵌入权重编码
        prompt_embeddings = self.llm_LoRA.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)  
        # 拼接 
        # llama_enc_out = torch.cat([prompt_embeddings, x_enc, x_fusion], dim=1)  # [B, total_seq_len, d_llm] 
        llama_enc_out = torch.cat([prompt_embeddings, queries, ts, vf], dim=1)  # [B, total_seq_len, d_llm] 
        # Learnable token
        llama_enc_out = torch.cat([llama_enc_out, self.learnable_token.expand(llama_enc_out.shape[0], -1, -1)], dim=1)
        # 通过 LLM 模型
        dec_out = self.llm_LoRA(inputs_embeds=llama_enc_out).last_hidden_state  # [B, total_seq_len, d_llm]
        # 切片
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))  
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')
        adjusted_rppg = dec_out
        
        return adjusted_rppg, loss # 返回形状为 [B, T, 1]
    


    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def get_video_encoder(self, video_encoder_type):
        if video_encoder_type == "clip":
            return CLIPVideoEncoder()
        elif video_encoder_type == "PhysFormer":
            return PhysFormerVideoEncoder()
        elif video_encoder_type == "PhysMamba":
            return PhysMambaVideoEncoder()
        elif video_encoder_type == "PhysNet":
            return PhysNetVideoEncoder()
        elif video_encoder_type == "PhysFormerCLIP":
            return PhysformerCLIP()
        else:
            raise NotImplementedError(f"Video encoder '{video_encoder_type}' is not implemented.")



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

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

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
    
class TimeSeriesMoudle(nn.Module):
    def __init__(self, seq_len, kernel_len, wavelet="coif6", j=3): # j 取 2 
        super(TimeSeriesMoudle, self).__init__()
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
        # 自适应时域与频域的权重参数
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
            norm_dc_part, (mdc, sdc) = self._normalize(dc) # 低频分量的均值和标准差
            norm_dc.append(norm_dc_part)
            m_list.append(mdc)
            s_list.append(sdc) # m_list, s_list: 高频分量的均值和标准差


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
    



    




