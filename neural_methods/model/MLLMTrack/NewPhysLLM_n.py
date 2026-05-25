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

class NewPhysLLM(nn.Module):
    def __init__(self, configs,):
        super(NewPhysLLM, self).__init__()
        self.video_encoder_type = configs.MODEL.VIDEO_ENC
        self.video_encoder = self.get_video_encoder(self.video_encoder_type)  # "rPPG" or "VideoEmb"
        self.pred_len = configs.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH  # configs.pred_len
        # self.seq_len = configs.seq_len
        self.d_ff = 128  # configs.d_ff
        # TODO:BERT改为768
        # self.d_llm = 4096  # configs.llm_dim
        # self.d_llm = 768  # bert
        # self.d_llm = 1536 # deepseek
        # d_llm will be set based on the selected LLM model
        self.patch_len = 16  # configs.patch_len
        self.stride = 8  # configs.stride
        llm_layers = 32
        dropout = 0.1
        n_heads = 8
        seq_len = 512
        self.rPPG_based_encoder = ["PhysMamba", "PhysFormer", "PhysNet", "PhysFormerCLIP", "EfficientPhys"]
        self.feature_based_encoder = ["clip"]
        self.enc_type = configs.MODEL.VIDEO_ENC
        enc_in = 1 if configs.MODEL.VIDEO_ENC in self.rPPG_based_encoder else 512  # 512 is the feature dimension in clip
        d_model = 32 if configs.MODEL.VIDEO_ENC in self.rPPG_based_encoder else 512  # 512 is the feature dimension in clip
        if configs.MODEL.LLM == 'LLAMA':
            self.is_sundial = False
            self.d_llm = 4096  # llama
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
            self.is_sundial = False
            self.d_llm = 768  # gpt2
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
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
            self.is_sundial = False
            self.d_llm = 768  # bert
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
        elif configs.MODEL.LLM == 'DeepSeek':
            self.is_sundial = False
            self.d_llm = 1536  # deepseek
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            self.deepseek_config = AutoConfig.from_pretrained(model_name)
            self.deepseek_config.num_hidden_layers = 32
            self.deepseek_config.output_attentions = True
            self.deepseek_config.output_hidden_states = True
            try:
                self.llm_model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only= False,
                    torch_dtype=torch.float32,
                    config=self.deepseek_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.deepseek_config,
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.MODEL.LLM == 'Sundial':
            model_name = "thuml/sundial-base-128m"
            self.is_sundial = True  # Flag to identify Sundial model
            try:
                self.sundial_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                # Get hidden dimension from config
                if hasattr(self.sundial_config, 'hidden_size'):
                    self.d_llm = self.sundial_config.hidden_size
                elif hasattr(self.sundial_config, 'd_model'):
                    self.d_llm = self.sundial_config.d_model
                elif hasattr(self.sundial_config, 'n_embd'):  # Some models use n_embd
                    self.d_llm = self.sundial_config.n_embd
                else:
                    # Default for sundial-base-128m, typically 512 or 768
                    # Try to infer from model after loading
                    self.d_llm = 512
                    print(f"Warning: Could not determine hidden dimension for Sundial from config, using default: {self.d_llm}")
                
                # Try to set output settings if supported
                # Note: Sundial has a bug with output_attentions=True, so we disable it
                # We only need hidden_states, not attention weights
                if hasattr(self.sundial_config, 'output_attentions'):
                    self.sundial_config.output_attentions = False  # Disable due to Sundial bug
                if hasattr(self.sundial_config, 'output_hidden_states'):
                    self.sundial_config.output_hidden_states = True
                
                try:
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.sundial_config,
                    )
                    # Try to get actual hidden dimension from loaded model
                    if hasattr(self.llm_model, 'config'):
                        if hasattr(self.llm_model.config, 'hidden_size'):
                            self.d_llm = self.llm_model.config.hidden_size
                        elif hasattr(self.llm_model.config, 'd_model'):
                            self.d_llm = self.llm_model.config.d_model
                        elif hasattr(self.llm_model.config, 'n_embd'):
                            self.d_llm = self.llm_model.config.n_embd
                    print(f"Sundial model loaded. Hidden dimension: {self.d_llm}")
                except EnvironmentError:
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.sundial_config,
                    )
                    # Update d_llm from loaded model
                    if hasattr(self.llm_model, 'config'):
                        if hasattr(self.llm_model.config, 'hidden_size'):
                            self.d_llm = self.llm_model.config.hidden_size
                        elif hasattr(self.llm_model.config, 'd_model'):
                            self.d_llm = self.llm_model.config.d_model
                        elif hasattr(self.llm_model.config, 'n_embd'):
                            self.d_llm = self.llm_model.config.n_embd
                    print(f"Sundial model loaded. Hidden dimension: {self.d_llm}")
                
                # Sundial is a time series model, may not need traditional tokenizer
                # AutoTokenizer doesn't recognize SundialConfig, so we use a fallback tokenizer
                print("Sundial model doesn't have a standard tokenizer. Using GPT2 tokenizer as fallback...")
                try:
                    # Try to load GPT2 tokenizer as fallback
                    self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
                    if not self.tokenizer.pad_token:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    print("GPT2 tokenizer loaded successfully as fallback for Sundial")
                except Exception as tokenizer_error:
                    print(f"Warning: Failed to load GPT2 tokenizer: {tokenizer_error}")
                    # Last resort: create a minimal tokenizer
                    # This is a very basic fallback - may need adjustment based on actual usage
                    from transformers import PreTrainedTokenizer
                    class DummyTokenizer:
                        def __init__(self):
                            self.pad_token = '[PAD]'
                            self.eos_token = '[EOS]'
                            self.vocab_size = 50257
                        def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=2048, **kwargs):
                            # Return a dummy tokenized output
                            import torch
                            if isinstance(text, list):
                                batch_size = len(text)
                            else:
                                batch_size = 1
                            # Return dummy input_ids with shape [batch_size, max_length]
                            return type('obj', (object,), {
                                'input_ids': torch.zeros(batch_size, min(max_length, 512), dtype=torch.long)
                            })()
                    self.tokenizer = DummyTokenizer()
                    print("Using dummy tokenizer for Sundial (minimal functionality)")
            except Exception as e:
                print(f"Error loading Sundial model: {e}")
                raise Exception(f'Failed to load Sundial model: {e}')
        else:
            raise Exception('LLM model is not defined')
        # # TODO:加入LoRA
        # Check if Sundial model supports LoRA (may need special handling)
        if hasattr(self, 'is_sundial') and self.is_sundial:
            # Try to apply LoRA, but handle potential incompatibility
            try:
                self.lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",  # Sundial is causal LM for time series
                )
                self.llm_LoRA = get_peft_model(self.llm_model, self.lora_config)
                print("LoRA successfully applied to Sundial model")
            except Exception as e:
                print(f"Warning: LoRA may not be fully compatible with Sundial: {e}")
                print("Using Sundial model without LoRA modifications")
                self.llm_LoRA = self.llm_model
        else:
            self.lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ2SEQ_LM",
            )
            self.llm_LoRA = get_peft_model(self.llm_model, self.lora_config)
        # # 冻结所有模型参数
        # for param in self.llm_model.parameters():
        #     param.requires_grad = False
        # # 这样 self.llm_LoRA 就和 self.llm_model 相同了，因为它没有进行任何 LoRA 改造
        # self.llm_LoRA = self.llm_model
        # Handle tokenizer pad token
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            if hasattr(self.tokenizer, 'add_special_tokens'):
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        self.dropout = nn.Dropout(dropout)
        # self.patch_embedding = PatchEmbedding(
        #     d_model, self.patch_len, self.stride, dropout)
        # self.patch_embedding = PatchEmbedding(
        #     768, self.patch_len, self.stride, dropout)
        self.patch_embedding = PatchEmbedding(
            self.d_llm, 8, 4, dropout)
        # Handle word embeddings - Sundial may have different structure
        if hasattr(self, 'is_sundial') and self.is_sundial:
            try:
                # Try to get input embeddings from Sundial
                if hasattr(self.llm_model, 'get_input_embeddings'):
                    input_embeddings = self.llm_model.get_input_embeddings()
                    if hasattr(input_embeddings, 'weight'):
                        self.word_embeddings = input_embeddings.weight
                        self.vocab_size = self.word_embeddings.shape[0]
                    else:
                        # Fallback: create dummy embeddings
                        print("Warning: Sundial embeddings structure differs, using fallback")
                        self.vocab_size = 50257  # GPT2 vocab size as fallback
                        self.word_embeddings = nn.Parameter(torch.randn(self.vocab_size, self.d_llm))
                else:
                    # Create learnable embeddings if not available
                    print("Warning: Sundial does not have get_input_embeddings, creating learnable embeddings")
                    self.vocab_size = 50257  # GPT2 vocab size as fallback
                    self.word_embeddings = nn.Parameter(torch.randn(self.vocab_size, self.d_llm))
            except Exception as e:
                print(f"Warning: Error getting Sundial embeddings: {e}, using fallback")
                self.vocab_size = 50257
                self.word_embeddings = nn.Parameter(torch.randn(self.vocab_size, self.d_llm))
        else:
            # For other LLMs (LLAMA, GPT2, BERT, DeepSeek), get embeddings from the model
            self.word_embeddings = self.llm_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 32
        # 可学习的source embeddings，专门为rPPG任务设计
        self.source_embeddings = nn.Parameter(torch.randn(self.num_tokens, self.d_llm) * 0.02)
        # self.reprogramming_layer = ReprogrammingLayer(d_model, n_heads, self.d_ff, self.d_llm)  # (32, 8, 128, 768)
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
        # TODO: TimeSeriesMoudle处理时间序列平稳化
        self.tsmoudle = TimeSeriesMoudle(seq_len=self.pred_len)
        # todo: 定义了一个多尺度融合特征
        # self.feature_fusion = MultiScaleFeatureFusion(
        #     feature_channels=[64, 64, 64],  # 对应三个特征图的通道数
        #     token_dim=self.d_llm # 768
        # )
        # self.feature_fusion = MultiAverageFeatureFusion(
        #     feature_channels=[64, 64, 64],  # 对应三个特征图的通道数
        #     token_dim=self.d_llm # 768
        # )
        bottleneck_dim_for_fusion = 768  # 从512提升到768，增强特征融合能力
        self.feature_fusion = MMFusion(
            feature_channels=[64, 64, 64],
            token_dim=self.d_llm,
            bottleneck_dim=bottleneck_dim_for_fusion
        )
        # self.feature_fusion = MMFusion(
        #     feature_channels=[64, 64, 64],  # 对应三个特征图的通道数[64, 64, 64], [32, 64, 64], [96, 96, 96]
        #     token_dim=self.d_llm # 768
        # )
        # # target_len应该与LLM的token序列长度匹配
        self.target_sequence_length = 32  # 举例,需要根据实际情况设置
        self.configs = configs
        # todo:定义一个可学习的token
        self.learnable_token = nn.Parameter(torch.randn(1, 1, self.d_llm))
        # 降维和升维层 - 减少信息损失
        self.reduced_dim = 1024  # 从512改为1024，减少信息损失
        self.dim_reduction = nn.Linear(self.d_llm, self.reduced_dim)
        self.dim_increase = nn.Linear(self.reduced_dim, self.d_llm)
        # 使用更大的维度和更多头数的CATBlock1以提高表达能力
        self.catblock = CATBlock1(self.reduced_dim, 8, self.reduced_dim * 2)  # 8头注意力
    def set_training_stage(self, stage='stage1'):
        """
        设置训练阶段
        Args:
            stage: 'stage1' - 冻结LLM，只训练encoder和fusion模块
                   'stage2' - 解冻LLM（LoRA），端到端训练
        """
        def set_requires_grad(module, requires_grad):
            """安全地设置模块的requires_grad"""
            if hasattr(module, 'parameters'):
                for param in module.parameters():
                    param.requires_grad = requires_grad
            elif hasattr(module, 'model') and hasattr(module.model, 'parameters'):
                # 处理包装了model的情况（如video_encoder）
                for param in module.model.parameters():
                    param.requires_grad = requires_grad
        
        if stage == 'stage1':
            print("=" * 50)
            print("Stage 1: 冻结LLM，训练Video Encoder和Feature Fusion")
            print("=" * 50)
            # 冻结LLM的所有参数
            for param in self.llm_LoRA.parameters():
                param.requires_grad = False
            
            # 确保其他模块可训练
            set_requires_grad(self.video_encoder, True)
            set_requires_grad(self.feature_fusion, True)
            set_requires_grad(self.tsmoudle, True)
            set_requires_grad(self.patch_embedding, True)
            set_requires_grad(self.catblock, True)
            set_requires_grad(self.output_projection, True)
            set_requires_grad(self.dim_reduction, True)
            set_requires_grad(self.dim_increase, True)
            
            # 可学习参数
            self.source_embeddings.requires_grad = True
            self.learnable_token.requires_grad = True
            
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
            
        elif stage == 'stage2':
            print("=" * 50)
            print("Stage 2: 解冻LLM (LoRA)，端到端微调")
            print("=" * 50)
            # 解冻LLM的LoRA参数
            for name, param in self.llm_LoRA.named_parameters():
                # 只解冻LoRA相关参数
                if 'lora' in name.lower():
                    param.requires_grad = True
                    print(f"  解冻: {name}")
                else:
                    param.requires_grad = False
            
            # 其他模块也保持可训练
            set_requires_grad(self.video_encoder, True)
            set_requires_grad(self.feature_fusion, True)
            set_requires_grad(self.tsmoudle, True)
            set_requires_grad(self.patch_embedding, True)
            set_requires_grad(self.catblock, True)
            set_requires_grad(self.output_projection, True)
            set_requires_grad(self.dim_reduction, True)
            set_requires_grad(self.dim_increase, True)
            
            self.source_embeddings.requires_grad = True
            self.learnable_token.requires_grad = True
            
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        else:
            raise ValueError(f"Unknown stage: {stage}. Must be 'stage1' or 'stage2'")
    
    def forward(self, data):
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
            raise ValueError("please specify encoder fusion dim")
        
        if self.input_type == 'rPPG_sequence':
            dec_out = self.process_rppg_sequence(x_enc, x_fusion)
            modified_rppg = dec_out.squeeze(-1)
        return modified_rppg
    def process_rppg_sequence(self, x_enc, x_fusion):
        """
        处理 rPPG 序列信号。
        """
        x_enc = x_enc.unsqueeze(-1)
        B, T, N = x_enc.size()
        x_enc = self.normalize_layers(x_enc, 'norm')
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc, n_vars = self.patch_embedding(x_enc)  # [B, 32, d_llm]
        
        # 使用可学习的source embeddings（专门为rPPG任务训练）
        source_embeddings = self.source_embeddings.unsqueeze(0).repeat(B, 1, 1)  # [B, num_tokens, d_llm]
        
        # 对输入进行降维
        source_embeddings_reduced = self.dim_reduction(source_embeddings)
        x_enc_reduced = self.dim_reduction(x_enc)
        x_fusion_reduced = self.dim_reduction(x_fusion)
        
        # 通过优化后的CATBlock1
        ts_we_reduced = self.catblock(source_embeddings_reduced, x_enc_reduced)
        xf_we_reduced = self.catblock(source_embeddings_reduced, x_fusion_reduced)
        
        # 升维回原始维度，并添加残差连接以保留信息
        # 使用缩放因子平衡残差连接的贡献
        ts_we = self.dim_increase(ts_we_reduced) + 0.5 * x_enc
        xf_we = self.dim_increase(xf_we_reduced) + 0.5 * x_fusion
        
        # 拼接特征
        llama_enc_out = torch.cat([ts_we, xf_we], dim=1)  # [B, total_seq_len, d_llm]
        
        # 通过 LLM 模型
        if hasattr(self, 'is_sundial') and self.is_sundial:
            try:
                llm_output = self.llm_LoRA(
                    inputs_embeds=llama_enc_out,
                    output_attentions=False,
                    output_hidden_states=True
                )
                if hasattr(llm_output, 'last_hidden_state'):
                    dec_out = llm_output.last_hidden_state
                elif hasattr(llm_output, 'hidden_states') and len(llm_output.hidden_states) > 0:
                    dec_out = llm_output.hidden_states[-1]
                elif isinstance(llm_output, tuple) and len(llm_output) > 0:
                    dec_out = llm_output[0]
                else:
                    raise ValueError("Could not extract hidden states from Sundial output")
            except Exception as e:
                print(f"Warning: Error with inputs_embeds for Sundial: {e}")
                print("Attempting alternative forward method...")
                try:
                    llm_output = self.llm_LoRA(inputs_embeds=llama_enc_out)
                    if hasattr(llm_output, 'last_hidden_state'):
                        dec_out = llm_output.last_hidden_state
                    elif isinstance(llm_output, tuple):
                        dec_out = llm_output[0]
                    else:
                        raise ValueError("Could not extract hidden states")
                except Exception as e2:
                    print(f"Fallback method also failed: {e2}")
                    raise Exception(f"Sundial model forward pass failed. Original error: {e}, Fallback error: {e2}")
        else:
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
        return adjusted_rppg  # 返回形状为 [B, T, 1]
    
    def get_video_encoder(self, video_encoder_type):
        if video_encoder_type == "clip":
            return CLIPVideoEncoder()
        elif video_encoder_type == "PhysFormer":
            return PhysFormerVideoEncoder()
        elif video_encoder_type == "PhysNet":
            return PhysNetVideoEncoder()
        elif video_encoder_type == "PhysFormerCLIP":
            return PhysformerCLIP()
        elif video_encoder_type == "EfficientPhys":
            return EfficientPhysVideoEncoder()
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
class TimeSeriesMoudle(nn.Module):
    def __init__(self, seq_len, kernel_len=10, wavelet="db4", j=2): # j=2减少分解层数，db4更适合心率信号
        super(TimeSeriesMoudle, self).__init__()
        self.seq_len = seq_len
        self.kernel_len = kernel_len
        self.epsilon = 1e-5
        # wavelet ：小波基
        # db4: Daubechies 小波，适合心率等生理信号
        # j=2: 减少分解层数，保留更多高频信息（心率变化）
        # 时域滑动窗口处理
        self.pad = nn.ReplicationPad1d(padding=(kernel_len // 2, kernel_len // 2 - ((kernel_len + 1) % 2)))
        # 频域小波变换
        self.dwt = DWT1D(wave=wavelet, J=j)
        self.idwt = IDWT1D(wave=wavelet)  # 逆小波变换
        # 自适应时域与频域的权重参数 - 初始更偏向频域（心率是频域特征）
        self.dwt_ratio = nn.Parameter(
            torch.clamp(torch.full((1, 1), 0.7), min=0., max=1.)  # 0.7偏向频域
        )
    def forward(self, x):
        """
        Args:
            x: 输入时间序列，形状为 [batch_size, seq_len]
        Returns:
            融合后的平稳序列，形状与输入相同。
        """
        # --- 时域平稳化 ---
        norm_x,_,_ = self._norm(x)
        norm_x = self._ewma_filter(norm_x, alpha=0.3)  # 降低平滑度，保留更多动态变化
        
        # --- 频域平稳化 ---
        ac, dc_list = self.dwt(x.unsqueeze(1))  # 小波分解（增加通道维度以适配 DWT1D）
        ac = ac.squeeze(1)  # 移除通道维度
        norm_ac, mac, sac = self._norm(ac)
        norm_ac = self._ewma_filter(norm_ac, alpha=0.3)  # 降低平滑度
        
        norm_dc, m_list, s_list = [], [], []
        for dc in dc_list:
            dc = dc.squeeze(1)  # 移除通道维度
            norm_dc_part, mdc, sdc = self._norm(dc)  # 低频分量的均值和标准差
            norm_dc_part = self._ewma_filter(norm_dc_part, alpha=0.3)  # 降低平滑度
            norm_dc.append(norm_dc_part)
            m_list.append(mdc)
            s_list.append(sdc)
        # 将平稳化后的低频和高频分量重构
        freq_x = self.idwt([norm_ac.unsqueeze(1), [d.unsqueeze(1) for d in norm_dc]]).squeeze(1)
        # --- 加权求和 ---
        dwt_r, time_r = self.dwt_ratio, 1 - self.dwt_ratio
        combined_x = norm_x * time_r + freq_x * dwt_r
        return combined_x
    def _norm(self, x):
        sample_mean = torch.mean(x, dim=(1), keepdim=True)  # 每个样本的均值
        sample_std = torch.std(x, dim=(1), keepdim=True)    # 每个样本的标准差
        x = (x - sample_mean) / (sample_std + self.epsilon)
        return x, sample_mean, sample_std
    
    def _ewma_filter(self, y, alpha=0.3):
        """
        可微分的EWMA滤波实现，支持batch处理
        Args:
            y: 输入张量 [batch_size, seq_len]
            alpha: 平滑系数 (0.3表示更依赖当前值，减少过度平滑)
        Returns:
            滤波后的张量 [batch_size, seq_len]
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = y.shape
        z = torch.zeros_like(y)
        z[:, 0] = y[:, 0]
        
        # 使用累积计算，保持可微分性
        for i in range(1, seq_len):
            z[:, i] = alpha * y[:, i] + (1 - alpha) * z[:, i-1]
        
        if squeeze_output:
            z = z.squeeze(0)
        
        return z
class CATBlock1(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super(CATBlock1, self).__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加 Dropout 以增强泛化能力
            nn.Linear(ff_dim, d_model)
        )
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
    def forward(self, query, key_value, mask=None):
        # Transpose query and key_value to (seq_length, batch_size, embedding_dim)
        query = query.transpose(0, 1)
        key_value = key_value.transpose(0, 1)
        # Self-attention
        attn_output, _ = self.self_attn(key_value, key_value, key_value, key_padding_mask=mask)
        query = self.layer_norm1(query + attn_output)
        # Cross-attention
        cross_attn_output, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=mask)
        query = self.layer_norm2(query + cross_attn_output)
        # Feed-forward network
        ffn_output = self.ffn(query)
        output = self.layer_norm3(query + ffn_output)
        # Transpose output back to (batch_size, seq_length, embedding_dim)
        output = output.transpose(0, 1)
        return output
class CATBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super(CATBlock, self).__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads)
        # Feed-forward network
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
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
    def forward(self, query, key_value, mask=None):
        # Transpose query and key_value to (seq_length, batch_size, embedding_dim)
        query = query.transpose(0, 1)
        key_value = key_value.transpose(0, 1)
        # Self-attention
        attn_output, _ = self.self_attn(key_value, key_value, key_value, key_padding_mask=mask)
        key_value = self.layer_norm1(key_value + attn_output)
        # Cross-attention
        cross_attn_output, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=mask)
        key_value = self.layer_norm2(key_value + cross_attn_output)
        # Feed-forward network
        ffn_output = self.ffn(key_value)
        output = self.layer_norm3(key_value + ffn_output)
        # Transpose output back to (batch_size, seq_length, embedding_dim)
        output = output.transpose(0, 1)
        return output
def z_score_normalize(tensor):
    mean = tensor.mean(dim=(1, 2), keepdim=True)  # 按样本标准化
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
    model = NewPhysLLM(configs)
    model.eval()
    # 构造假输入，根据模型实际输入需求调整形状
    # 例如 PhysNet 编码器接受 [batch_size, 3, T, H, W]
    dummy_data = torch.randn(1, 3, 128, 128, 128)
    # 计算 MACs 和参数量
    macs, params = profile(model, inputs=(dummy_data,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}")
    print(f"Parameters: {params}")
