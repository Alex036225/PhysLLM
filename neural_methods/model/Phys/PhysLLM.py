from math import sqrt
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from ..PhysLLMModules.TimeLLM.layers.Embed import PatchEmbedding
import transformers
from ..PhysLLMModules.TimeLLM.layers.StandardNorm import Normalize
from peft import get_peft_model, LoraConfig
import argparse

transformers.logging.set_verbosity_error()


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


class PhysLLM(nn.Module):

    def __init__(self, configs,):
        super(PhysLLM, self).__init__()
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
        self.rPPG_based_encoder = ["PhysMamba", "PhysFormer", "PhysNet"]
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
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
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
        for name, param in self.llm_LoRA.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameter: {name}")



        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False
        # TODO:LoRA改为TRUE
        for name, param in self.llm_LoRA.named_parameters():
            if "lora" in name:
                param.requires_grad = True


        self.dropout = nn.Dropout(dropout)

        self.patch_embedding = PatchEmbedding(
            d_model, self.patch_len, self.stride, dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(d_model, n_heads, self.d_ff, self.d_llm)  # (32, 8, 128, 4096)

        self.patch_nums = int((seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        # torch.Size([2, 1, 128, 64])
        self.output_projection = FlattenHead(enc_in, self.head_nf, self.pred_len, head_dropout=dropout)
        # else:
            # raise NotImplementedError
        # 根据输入类型定义处理层
        if self.enc_type in self.rPPG_based_encoder:
            # 对于 rPPG 序列信号，使用 ReprogrammingLayer
            self.input_type = 'rPPG_sequence'
            # self.reprogramming_layer = ReprogrammingLayer(
            #     d_model=1,  # 输入维度为 1
            #     n_heads=n_heads,
            #     d_keys=None,
            #     d_llm=self.d_llm
            # )
            # self.output_projection = nn.Linear(self.d_llm, 1)  # 输出维度为 1
        elif self.enc_type in self.feature_based_encoder:
            # 对于视频特征，使用映射层
            self.input_type = 'video_feature'
            feature_dim = ...  # 您需要根据实际特征维度设置
            # self.feature_mapping = nn.Linear(feature_dim, self.d_llm)
            # self.output_projection = nn.Linear(self.d_llm, self.pred_len)  # 输出预测的 rPPG 序列
        else:
            raise ValueError(f"Invalid encoder type: {self.enc_type}")

        self.normalize_layers = Normalize(enc_in, affine=False)

    def forward(self, data):
        x_enc = data['video_features']
        prompt = data['prompt']
        # dec_out = self.forecast(x_enc, prompt)
        if self.input_type == 'rPPG_sequence':
            dec_out = self.process_rppg_sequence(x_enc, prompt)
            original_rppg = x_enc
            modified_rppg = dec_out.squeeze(-1)
        elif self.input_type == 'video_feature':
            dec_out = self.process_video_feature(x_enc, prompt)
            original_rppg = torch.tensor([0], device=x_enc.device)
            modified_rppg = dec_out.squeeze(-1)
        else:
            raise ValueError(f"Invalid input type: {self.input_type}")
        
        return modified_rppg, original_rppg  # [:, -self.pred_len:, :]
    
    def process_rppg_sequence(self, x_enc, prompt):
        """
        处理 rPPG 序列信号，根据 prompt 进行微调或调整。
        """
        x_enc = x_enc.unsqueeze(-1)
        B, T, N = x_enc.size()
        x_enc = self.normalize_layers(x_enc, 'norm')

        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 生成任务描述
        Task = "Remote Photoplethysmography (rPPG) signals exhibit domain differences across different datasets, primarily manifested in factors such as the subject's skin color, ethnicity, gender, age, and environmental lighting conditions. Please utilize the powerful zero-shot generalization capability of the LLM to fine-tune the acquired rPPG signals based on the aforementioned different conditions to improve the accuracy of the signals."
        
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

            Statistic = f"Input statistics: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, the trend of input is {'upward' if trends[b] > 0 else 'downward'}, top 5 lags are : {lags_values_str}<|<end_prompt>|>"

            prompt_text = (
                f"<|start_prompt|>Dataset description: This is a dataset about rPPG signal."
                f"{prompt[b]}"
                f"Task description: {Task}"
                f"{Statistic}<|end_prompt|>"
            )
            prompts.append(prompt_text)
        
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # 拼接 prompt_embeddings 和 enc_out
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)  # [B, total_seq_len, d_llm]
        
        # 通过 LLM 模型
        dec_out = self.llm_LoRA(inputs_embeds=llama_enc_out).last_hidden_state  # [B, total_seq_len, d_llm]
        
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        adjusted_rppg = self.normalize_layers(dec_out, 'denorm')
        
        return adjusted_rppg  # 返回形状为 [B, T, 1]


    def process_video_feature(self, x_enc, prompt):
        """
        处理视频编码器的特征，根据 prompt 将其解码为 rPPG 信号。
        """
        B, T, N = x_enc.size()
        
        # 生成任务描述
        Task = "Remote Photoplethysmography (rPPG) signals exhibit domain differences across different datasets, primarily manifested in factors such as the subject's skin color, ethnicity, gender, age, and environmental lighting conditions. Please utilize the powerful zero-shot generalization capability of the LLM to predict the corresponding rPPG signals based on the aforementioned different conditions and the provided video features."
        
        # 构建完整的 prompt
        prompts = []
        for b in range(B):
            prompt_text = (
                f"<|start_prompt|>Dataset description: This is a dataset about rPPG signal."
                f"{prompt[b]}\n"
                f"Task description:{Task}\n"
            )
            prompts.append(prompt_text)
        
        # 对 prompt 进行编码
        prompt_tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(x_enc.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tokens)
        
        # 对特征进行处理
        # 这里假设您已经在传入 `data` 前处理好特征，可以直接使用
        # 如果需要进一步压缩特征，可以在这里添加映射层
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # x_enc_flat = x_enc.view(B, T, -1)  # 展平特征
        enc_out = self.reprogramming_layer(x_enc, source_embeddings, source_embeddings)
        # enc_out = self.feature_mapping(x_enc_flat)  # [B, T, d_llm]
        
        # 拼接 prompt_embeddings 和 enc_out
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)  # [B, total_seq_len, d_llm]
        
        # 通过 LLM 模型
        # TODO：eval
        # self.llm_model.eval()
        dec_out = self.llm_LoRA(inputs_embeds=llama_enc_out).last_hidden_state  # [B, total_seq_len, d_llm]
        
        # 提取对应于输入特征的输出
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(
            dec_out, (-1, 1, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        
        # 使用输出投影层，得到预测的 rPPG 信号
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        adjusted_rppg = dec_out.permute(0, 2, 1).contiguous()

        # adjusted_rppg = self.normalize_layers(dec_out, 'denorm')
        
        return adjusted_rppg  # 返回形状为 [B, T, pred_len]


    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


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
