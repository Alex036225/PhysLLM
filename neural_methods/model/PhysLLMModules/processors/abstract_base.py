from abc import ABC, abstractmethod
from pathlib import Path
import os

class EncoderBase(ABC):
    @abstractmethod
    def encode(self, x):
        """
        抽象方法，定义编码器的接口。
        参数:
            facial_video: 输入的视频数据
        返回:
            video_features: 视频特征
        """
        pass


def resolve_checkpoint_path(configs, field_name, env_var):
    """Resolve a required checkpoint path from config first, then env vars."""
    checkpoint_cfg = getattr(getattr(configs.MODEL, "CHECKPOINTS", None), field_name, "")
    checkpoint_path = checkpoint_cfg or os.getenv(env_var, "")
    if not checkpoint_path:
        raise ValueError(
            f"Missing checkpoint path for {field_name}. "
            f"Set MODEL.CHECKPOINTS.{field_name} or {env_var}."
        )

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint for {field_name} does not exist: {path}"
        )
    return str(path)
