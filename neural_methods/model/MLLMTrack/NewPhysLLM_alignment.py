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

class NNLM(nn.Module):
    """
    Neural Network Language Model (NNLM) - Bengio et al. (2003)
    Original paper: "A Neural Probabilistic Language Model"
    
    This implementation adapts NNLM for time series processing in PhysLLM.
    Instead of predicting next word, it processes time series embeddings.
    """
    def __init__(self, embedding_dim, hidden_dim, num_layers=1, context_window=3, dropout=0.1):
        super(NNLM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_window = context_window
        
        # Input dimension: context_window * embedding_dim (concatenated embeddings)
        input_dim = context_window * embedding_dim
        
        # Build feed-forward network layers
        layers = []
        current_dim = input_dim
        
        # First hidden layer
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
        
        # Additional hidden layers if num_layers > 1
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
        
        self.feedforward = nn.Sequential(*layers)
        
        # Output projection (for compatibility with PhysLLM architecture)
        # Outputs embeddings of the same dimension as input
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, inputs_embeds, output_attentions=False, output_hidden_states=True):
        """
        Forward pass of NNLM.
        
        Args:
            inputs_embeds: [batch_size, seq_len, embedding_dim]
            output_attentions: Not used in NNLM (no attention mechanism)
            output_hidden_states: If True, return intermediate hidden states
        
        Returns:
            ModelOutput-like object with last_hidden_state attribute
        """
        batch_size, seq_len, embedding_dim = inputs_embeds.shape
        
        # Process sequence with sliding window
        # For each position, use context_window previous embeddings
        outputs = []
        hidden_states = []
        
        for i in range(seq_len):
            # Get context window (previous context_window tokens)
            start_idx = max(0, i - self.context_window + 1)
            context_embeds = inputs_embeds[:, start_idx:i+1, :]  # [B, context_len, emb_dim]
            
            # Pad if necessary (at the beginning of sequence)
            if context_embeds.shape[1] < self.context_window:
                padding = torch.zeros(
                    batch_size, 
                    self.context_window - context_embeds.shape[1], 
                    embedding_dim,
                    device=context_embeds.device,
                    dtype=context_embeds.dtype
                )
                context_embeds = torch.cat([padding, context_embeds], dim=1)
            
            # Flatten context window: [B, context_window * embedding_dim]
            context_flat = context_embeds.reshape(batch_size, -1)
            
            # Feed-forward network
            hidden = self.feedforward(context_flat)  # [B, hidden_dim]
            
            if output_hidden_states:
                hidden_states.append(hidden)
            
            # Project to output dimension
            output = self.output_projection(hidden)  # [B, embedding_dim]
            outputs.append(output)
        
        # Stack outputs: [B, seq_len, embedding_dim]
        last_hidden_state = torch.stack(outputs, dim=1)
        
        # Create output object compatible with transformers ModelOutput
        class ModelOutput:
            def __init__(self, last_hidden_state, hidden_states=None):
                self.last_hidden_state = last_hidden_state
                self.hidden_states = hidden_states if hidden_states else [last_hidden_state]
        
        return ModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states if output_hidden_states else None
        )

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
        elif configs.MODEL.LLM == 'NNLM':
            # Neural Network Language Model - Bengio et al. (2003)
            self.is_sundial = False
            # NNLM parameters
            # embedding_dim will be set based on d_llm (we'll set d_llm first)
            # Default hidden dimension for NNLM
            nnlm_hidden_dim = 512  # Can be adjusted
            nnlm_num_layers = 2  # Number of hidden layers
            nnlm_context_window = 3  # Context window size (n-gram order)
            
            # Set d_llm to match typical embedding dimensions
            # NNLM typically uses smaller dimensions than modern LLMs
            self.d_llm = 768  # Standard embedding dimension for compatibility
            
            print(f"Initializing NNLM (Bengio 2003) with embedding_dim={self.d_llm}, "
                  f"hidden_dim={nnlm_hidden_dim}, num_layers={nnlm_num_layers}, "
                  f"context_window={nnlm_context_window}")
            
            # Initialize NNLM model
            self.llm_model = NNLM(
                embedding_dim=self.d_llm,
                hidden_dim=nnlm_hidden_dim,
                num_layers=nnlm_num_layers,
                context_window=nnlm_context_window,
                dropout=dropout
            )
            
            # NNLM doesn't use tokenizer (it works directly with embeddings)
            # Create a dummy tokenizer for compatibility
            class DummyTokenizer:
                def __init__(self):
                    self.pad_token = '[PAD]'
                    self.eos_token = '[EOS]'
                    self.vocab_size = 10000  # Dummy vocab size
                def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=2048, **kwargs):
                    import torch
                    if isinstance(text, list):
                        batch_size = len(text)
                    else:
                        batch_size = 1
                    # Return dummy input_ids
                    return type('obj', (object,), {
                        'input_ids': torch.zeros(batch_size, min(max_length, 512), dtype=torch.long)
                    })()
            
            self.tokenizer = DummyTokenizer()
            print("NNLM model initialized successfully (no tokenizer needed)")
            
            # Create dummy word embeddings for compatibility with existing code
            # NNLM doesn't use pre-trained word embeddings, but we need this for the mapping layer
            self.vocab_size = 10000  # Dummy vocab size
            self.word_embeddings = nn.Parameter(torch.randn(self.vocab_size, self.d_llm))
            print(f"Created dummy word embeddings with vocab_size={self.vocab_size}")
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
        elif configs.MODEL.LLM == 'NNLM':
            # NNLM is a simple feed-forward network, doesn't need LoRA
            # Just use the model directly
            self.llm_LoRA = self.llm_model
            print("NNLM model: LoRA not needed (using model directly)")
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
        # Handle tokenizer pad token (skip for NNLM as it uses dummy tokenizer)
        if configs.MODEL.LLM != 'NNLM':
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
        # Handle word embeddings - Sundial and NNLM may have different structure
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
        elif configs.MODEL.LLM == 'NNLM':
            # NNLM doesn't have get_input_embeddings, word_embeddings already created in NNLM initialization
            # self.word_embeddings and self.vocab_size are already set in the NNLM initialization block
            # No need to do anything here, they're already initialized
            pass
        else:
            # For other LLMs (LLAMA, GPT2, BERT, DeepSeek), get embeddings from the model
            self.word_embeddings = self.llm_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 32
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
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
        # TODO
        self.tsmoudle = TimeSeriesMoudle(self.d_ff)
        # todo: 定义了一个多尺度融合特征
        # self.feature_fusion = MultiScaleFeatureFusion(
        #     feature_channels=[64, 64, 64],  # 对应三个特征图的通道数
        #     token_dim=self.d_llm # 768
        # )
        # self.feature_fusion = MultiAverageFeatureFusion(
        #     feature_channels=[64, 64, 64],  # 对应三个特征图的通道数
        #     token_dim=self.d_llm # 768
        # )
        bottleneck_dim_for_fusion = 512 # <<<--- 调整这个值来控制参数量
        self.feature_fusion = MMFusion(
            feature_channels=[64, 64, 64],
            token_dim=self.d_llm,
            bottleneck_dim=bottleneck_dim_for_fusion # <<<--- 传入瓶颈维度
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
        self.loss_function = CosineSimilarityLoss()
        # 降维和升维层
        self.reduced_dim = 512  # 修改为 512
        self.dim_reduction = nn.Linear(self.d_llm, self.reduced_dim)
        self.dim_increase = nn.Linear(self.reduced_dim, self.d_llm)
        # 使用更小的维度和更少的头数的CATBlock1
        self.catblock = CATBlock1(self.reduced_dim, 4, self.reduced_dim * 2)  # 修改为 512*2 中间维度
    def forward(self, data, prompt):
        if self.configs.MODEL.VIDEO_ENC == "PhysNet":
            x_enc, x_visual6464, x_visual3232, x_visual1616  = self.video_encoder.encode(data)
            x_enc = self.tsmoudle(x_enc)
            # 按样本标准化
            # sample_mean = torch.mean(x_enc, dim=(1), keepdim=True)  # 每个样本的均值
            # sample_std = torch.std(x_enc, dim=(1), keepdim=True)    # 每个样本的标准差
            # x_enc = (x_enc - sample_mean) / sample_std
            # x_visual6464 = z_score_normalize(x_visual6464)
            # x_visual3232 = z_score_normalize(x_visual3232)
            # x_visual1616 = z_score_normalize(x_visual1616)
            x_fusion = self.feature_fusion(
                features=[x_visual6464, x_visual3232, x_visual1616],
                target_len=self.target_sequence_length
            )
        elif self.configs.MODEL.VIDEO_ENC == "EfficientPhys":
            x_enc, x_visual1616, x_visual3232, x_visual6464 = self.video_encoder.encode(data)
            x_enc = self.tsmoudle(x_enc)
            x_fusion = self.feature_fusion(   # 32, 64, 64
                features=[x_visual1616, x_visual3232, x_visual6464],
                target_len=self.target_sequence_length
            )
        elif self.configs.MODEL.VIDEO_ENC == "PhysFormer":
            x_enc, Trans_features, Trans_features2, Trans_features3 = self.video_encoder.encode(data)
            x_enc = self.tsmoudle(x_enc)
            x_fusion = self.feature_fusion(   # 64, 64, 64
                features=[Trans_features, Trans_features2, Trans_features3],
                target_len=self.target_sequence_length
            )        
        else:
            raise "please specify encoder fusion dim"
        if self.input_type == 'rPPG_sequence':
            dec_out= self.process_rppg_sequence(x_enc, prompt,x_fusion, prompt_use=False)
            # original_rppg = x_enc
            modified_rppg = dec_out.squeeze(-1)
        return modified_rppg  # [:, -self.pred_len:, :]
    def process_rppg_sequence(self, x_enc, prompt, x_fusion, prompt_use=False):
        """
        处理 rPPG 序列信号，根据 prompt 进行微调或调整。
        """
        x_enc = x_enc.unsqueeze(-1)
        B, T, N = x_enc.size()
        x_enc = self.normalize_layers(x_enc, 'norm')
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        # 生成任务描述
        if prompt_use == True:
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
                    f"Dataset description: This is a dataset about rPPG signal."
                    f"Task description: {Task}"
                    f"Face description:{prompt[b]}"
                    f"{Statistic}"
                )
                prompts.append(prompt_text)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc, n_vars = self.patch_embedding(x_enc) # [4, 32, 768]
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        if prompt_use == True:
            prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(x_enc.device)
            # 2. 嵌入权重编码
            # Handle different LLM types' embedding structure
            if hasattr(self, 'is_sundial') and self.is_sundial:
                try:
                    if hasattr(self.llm_LoRA, 'get_input_embeddings'):
                        prompt_embeddings = self.llm_LoRA.get_input_embeddings()(prompt.to(x_enc.device))
                    else:
                        # Fallback: use word_embeddings directly
                        prompt_embeddings = F.embedding(prompt.to(x_enc.device), self.word_embeddings)
                except Exception as e:
                    print(f"Warning: Error getting prompt embeddings for Sundial: {e}, using fallback")
                    prompt_embeddings = F.embedding(prompt.to(x_enc.device), self.word_embeddings)
            elif self.configs.MODEL.LLM == 'NNLM':
                # NNLM doesn't have get_input_embeddings, use word_embeddings directly
                prompt_embeddings = F.embedding(prompt.to(x_enc.device), self.word_embeddings)
            else:
                prompt_embeddings = self.llm_LoRA.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)  
        # 3. CAT (x_enc, x_fusion, self.word_embedding) with dimension reduction
        source_embeddings = source_embeddings.repeat(x_enc.shape[0], 1, 1)
        # 对输入进行降维
        source_embeddings_reduced = self.dim_reduction(source_embeddings)
        x_enc_reduced = self.dim_reduction(x_enc)
        x_fusion_reduced = self.dim_reduction(x_fusion)
        # 通过优化后的CATBlock1
        ts_we_reduced = self.catblock(source_embeddings_reduced, x_enc_reduced)
        xf_we_reduced = self.catblock(source_embeddings_reduced, x_fusion_reduced)
        # 升维回原始维度
        ts_we = self.dim_increase(ts_we_reduced)
        xf_we = self.dim_increase(xf_we_reduced)
        # ts_we = x_enc
        #qfeature = self.qformer(ts_we, xf_we)
        # 拼接 
        if prompt_use == True:
            llama_enc_out = torch.cat( [prompt_embeddings, ts_we, xf_we], dim=1)  # [B, total_seq_len, d_llm] 
        else:
            llama_enc_out = torch.cat( [ts_we, xf_we], dim=1)  # [B, total_seq_len, d_llm]
        # llama_enc_out = torch.cat( [prompt_embeddings, x_enc, x_fusion], dim=1)  # [B, total_seq_len, d_llm] 
        # llama_enc_out = torch.cat( [prompt_embeddings, x_fusion], dim=1)  # [B, total_seq_len, d_llm]
        # llama_enc_out = torch.cat( [x_enc, x_fusion], dim=1)  # [B, total_seq_len, d_llm] 
        # 通过 LLM 模型
        # Handle Sundial's potentially different forward signature
        if hasattr(self, 'is_sundial') and self.is_sundial:
            try:
                # Sundial: explicitly disable output_attentions to avoid bug
                # Pass output_attentions=False in forward call to override config
                llm_output = self.llm_LoRA(
                    inputs_embeds=llama_enc_out,
                    output_attentions=False,  # Explicitly disable to avoid Sundial bug
                    output_hidden_states=True  # We need hidden states
                )
                if hasattr(llm_output, 'last_hidden_state'):
                    dec_out = llm_output.last_hidden_state
                elif hasattr(llm_output, 'hidden_states') and len(llm_output.hidden_states) > 0:
                    dec_out = llm_output.hidden_states[-1]  # Use last hidden state
                elif isinstance(llm_output, tuple) and len(llm_output) > 0:
                    dec_out = llm_output[0]  # First element is usually hidden states
                else:
                    raise ValueError("Could not extract hidden states from Sundial output")
            except Exception as e:
                print(f"Warning: Error with inputs_embeds for Sundial: {e}")
                print("Attempting alternative forward method...")
                # Fallback: try without explicit output_attentions parameter
                try:
                    # Try with minimal parameters
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
        elif self.configs.MODEL.LLM == 'NNLM':
            # NNLM forward pass
            llm_output = self.llm_LoRA(
                inputs_embeds=llama_enc_out,
                output_attentions=False,  # NNLM doesn't have attention
                output_hidden_states=True
            )
            # NNLM returns ModelOutput-like object with last_hidden_state
            if hasattr(llm_output, 'last_hidden_state'):
                dec_out = llm_output.last_hidden_state
            elif isinstance(llm_output, tuple) and len(llm_output) > 0:
                dec_out = llm_output[0]
            else:
                raise ValueError("Could not extract hidden states from NNLM output")
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
        return adjusted_rppg # 返回形状为 [B, T, 1]
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
    def __init__(self, seq_len, kernel_len=10, wavelet="coif6", j=3): # j 取 2 
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
        # x_window = x.unfold(-1, self.kernel_len, 1)  # 滑动窗口展开
        # m, s = x_window.mean(dim=-1), x_window.std(dim=-1)  # 滑动窗口均值和标准差
        # m, s = self.pad(m), self.pad(s)  # 填充以匹配输入维度
        # norm_x = (x - m) / (s + self.epsilon)  # 时域平稳化序列
        norm_x,_,_ = self._norm(x)
        norm_x = self._ewma_filter(norm_x)
        # --- 频域平稳化 ---
        ac, dc_list = self.dwt(x.unsqueeze(1))  # 小波分解（增加通道维度以适配 DWT1D）
        ac = ac.squeeze(1)  # 移除通道维度
        norm_ac, mac, sac = self._norm(ac)
        norm_ac = self._ewma_filter(norm_ac)
        norm_dc, m_list, s_list = [], [], []
        for dc in dc_list:
            dc = dc.squeeze(1)  # 移除通道维度
            norm_dc_part, mdc, sdc = self._norm(dc) # 低频分量的均值和标准差
            norm_dc_part = self._ewma_filter(norm_dc_part)
            norm_dc.append(norm_dc_part)
            m_list.append(mdc)
            s_list.append(sdc) # m_list, s_list: 高频分量的均值和标准差
        # 将平稳化后的低频和高频分量重构
        freq_x = self.idwt([norm_ac.unsqueeze(1), [d.unsqueeze(1) for d in norm_dc]]).squeeze(1)
        # --- 加权求和 ---
        dwt_r, time_r = self.dwt_ratio, 1 - self.dwt_ratio
        combined_x = norm_x * time_r + freq_x * dwt_r
        return combined_x
    def _norm(self, x):
        sample_mean = torch.mean(x, dim=(1), keepdim=True)  # 每个样本的均值
        sample_std = torch.std(x, dim=(1), keepdim=True)    # 每个样本的标准差
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
    # 例如 PhysNet 编码器接受 [batch_size, seq_len, 1]
    dummy_data = torch.randn(1, 3, 128, 128, 128)
    dummy_prompt = ["Test prompt"]
    # 计算 MACs 和参数量
    macs, params = profile(model, inputs=(dummy_data, ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}")
    print(f"Parameters: {params}")
