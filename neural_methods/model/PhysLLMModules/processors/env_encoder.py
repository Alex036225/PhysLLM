import torch
import clip
from PIL import Image
from abc import ABC, abstractmethod
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from .abstract_base import EncoderBase

class CLIPEnvEncoder(EncoderBase):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载预训练的 CLIP 模型和预处理器
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False

        # Environment Text Prompt
        self.lighting_descriptions = [
            "The scene is very bright.",
            "The scene is well-lit.",
            "The scene has normal lighting.",
            "The scene is dimly lit.",
            "The scene is dark.",
            "The scene is underexposed.",
            "The scene is overexposed.",
            "The scene is in shadow.",
            "The scene is in sunlight.",
            "The scene is at night."
        ]
        # 对文本描述进行编码
        self.text_tokens = clip.tokenize(self.lighting_descriptions).to(self.device)

    def encode(self, x):
        """
        使用 CLIP 模型获取场景光照描述。

        参数:
            x: 输入的视频数据，形状为 (batch_size, channels, num_frames, height, width)
        返回:
            scene_descriptions: 包含每个视频样本的场景光照描述的列表
        """
        B, C, T, H, W = x.size()
        mid_frame = T // 2
        scene_descriptions = []

        # 处理每个视频样本
        for i in range(B):
            # 提取中间帧作为代表帧
            frame = x[i, :, mid_frame, :, :].cpu()
            # 将张量转换为 PIL 图像
            frame = TF.to_pil_image(frame)
            # 使用预处理器处理图像
            image_input = self.preprocess(frame).unsqueeze(0).to(self.device)

            # 计算图像和文本的特征向量
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(self.text_tokens)

                # 计算相似度
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarities = (image_features @ text_features.T).squeeze(0)

            # 找到相似度最高的描述
            best_match_idx = similarities.argmax().item()
            best_description = self.lighting_descriptions[best_match_idx]
            scene_descriptions.append(best_description)

        return scene_descriptions
