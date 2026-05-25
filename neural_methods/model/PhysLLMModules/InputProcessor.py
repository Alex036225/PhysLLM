import torch
import torch.nn as nn
from .processors.env_encoder import CLIPEnvEncoder
from .processors.video_encoder import (
    CLIPVideoEncoder,
    EfficientPhysVideoEncoder,
    PhysFormerVideoEncoder,
    PhysNetVideoEncoder,
)
from .processors.face_encoder import FaceXFormerEncoder

class Description_Model():
    def __init__(self, description):
        self.description = description

    def encode(self, x):
        description = []
        for i in range(x.size(0)):
            description.append(self.description)
        return description


def freeze_and_setDevice(models, device='cpu'):
    for model in models:
        if isinstance(model, nn.Module):
            for param in model.parameters():
                param.requires_grad = False



class RPPGInputProcessor(nn.Module):
    """
    RPPGInputProcessor类用于处理多模态输入:
    - 从 face_model 提取面部特征
    - 从 env_model 提取场景特征
    - 将提取的特征整合到一个单一的输入数据结构中，以便 TimeLLM 进行处理
    """
    def __init__(self, configs, face_model_type, env_model_type, video_encoder_type):
        """
        初始化RPPGInputProcessor
        
        参数:
        face_model: 一个模型（或函数）用于提取面部特征，如 faceXFormer
        env_model: 一个模型（或函数）用于生成场景描述文本，如一个视频描述模型
        tokenizer: 用于将文本转换为token ID的tokenizer，应该与TimeLLM使用的LLM对应
        device: 模型和数据所在的设备
        """
        super(RPPGInputProcessor, self).__init__()
        self.configs = configs
        self.face_model = self.get_face_model(face_model_type)
        self.env_model = self.get_env_model(env_model_type)
        self.video_encoder = self.get_video_encoder(video_encoder_type)  # "rPPG" or "VideoEmb"

        # freeze_and_setDevice([self.face_model, self.env_model, self.video_encoder], configs.DEVICE)

        self.task_description = "The rPPG signal reflects physiological parameters like heart rate, which are inferred from minute color variations in the skin due to blood flow."

    
    def forward(self, facial_video, task_description=None):
        """
        将输入的面部视频、任务描述和环境信息整合成TimeLLM可接受的输入格式
        
        参数:
        facial_video: 输入视频数据（张量或其他格式）
        task_description: 描述当前任务的文本（例如"Predict the rPPG signal"）
        
        返回:
        processed_data: {'prompt':..., 'rppg':..., 'videoEmb':...}
        """
        B, C, T, H, W = facial_video.shape
        facial_description = self.face_model.encode(facial_video)
        
        # Step 2: 生成场景描述
        scene_description = self.env_model.encode(facial_video)
        
        # Step 3: 创建Prompt
        if task_description is None:
            task_description = self.task_description
        prompt = self.generate_prompt(task_description, scene_description, facial_description, B)
        
        # Step 4: 使用视频编码器提取视频特征
        """
        这里的返回值可以是一个feature，也可以直接返回一个时序预测的值，然后让 LLM 根据 prompt 进行 zero-shot 来微调
        """
        video_features = self.video_encoder.encode(facial_video)
        
        # Step 7: 返回处理后的数据
        assert len(prompt) == B

        processed_data = {
           'prompt': prompt,
           'video_features': video_features.float(),
        }
        
        return processed_data
        
    def generate_prompt(self, task_description, scene_description, facial_attributes, B):
        """
        生成LLM使用的Prompt文本
        
        参数:
        task_description: 任务描述，例如 'Focus on detecting rPPG signal'
        scene_description: 环境描述文本
        facial_attributes: 面部属性文本
        
        返回:
        prompt: 一个字符串，包含所有信息
        """
        prompt = []
        for i in range(B):
            prompt_ = f"Task: {task_description};Scene: {scene_description[i]};Facial attributes: {facial_attributes[i]}"
            prompt.append(prompt_)
        return prompt
    

    def get_video_encoder(self, video_encoder_type):
        if video_encoder_type == "clip":
            return CLIPVideoEncoder()
        elif video_encoder_type == "PhysFormer":
            return PhysFormerVideoEncoder(self.configs)
        elif video_encoder_type == "PhysNet":
            return PhysNetVideoEncoder(self.configs)
        elif video_encoder_type == "EfficientPhys":
            return EfficientPhysVideoEncoder(self.configs)
        elif video_encoder_type == "PhysMamba":
            raise NotImplementedError("PhysMamba is not included in this public release.")
        else:
            raise NotImplementedError(f"Video encoder '{video_encoder_type}' is not implemented.")
        
    def get_face_model(self, face_model_type):
        if face_model_type == "FaceXFormer":
            face_model = FaceXFormerEncoder(self.configs)
        else:
            raise Exception('Face model is not defined')


        return face_model


    def get_env_model(self, env_model_type):
        if env_model_type is not None:
            if env_model_type == "clip":
                env_model = CLIPEnvEncoder()
            else:
                raise Exception('Face model is not defined')
        else:
            env_model = Description_Model("This is a facial-based video")

        return env_model


    def extract_rppg_features(self, facial_video):
        """
        提取rPPG相关的视频特征
        
        参数:
        facial_video: 输入面部视频张量
        
        返回:rPPG的时序预测
        """
        # TODO: 实现视频到rPPG特征的提取逻辑
        # 这可能涉及到spatio-temporal模型、ROI提取或其他rPPG特征提取方法
        
        # 占位
        return torch.zeros_like(facial_video)

    def extract_video_embedding(self, facial_video):
        """
        提取rPPG相关的视频特征
        
        参数:
        facial_video: 输入面部视频张量
        
        返回:video_features
        """
        
        # 占位
        return torch.zeros_like(facial_video)


    def project_video_features(self, video_features):
        """
        将视频特征投影到LLM嵌入空间
        
        参数:
        video_features: 来自extract_video_embedding的特征张量
        
        返回:
        video_embeddings: 与LLM嵌入维度匹配的视频嵌入张量
        """
        # TODO: 实现投影逻辑，例如使用线性层
        # video_embeddings = self.video_projection_layer(video_features)
        # return video_embeddings
        
        # 占位
        return torch.zeros_like(video_features)
