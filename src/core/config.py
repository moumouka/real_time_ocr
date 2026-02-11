"""
配置管理模块
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """系统配置数据类"""
    # 通用设置
    use_yolo_first: bool = True
    fallback_enabled: bool = True
    confidence_threshold: float = 0.4
    nms_iou_threshold: float = 0.5
    use_gpu: bool = False
    languages: List[str] = field(default_factory=lambda: ['ch_sim', 'en'])
    min_text_area: int = 100
    max_text_area: int = 100000
    debug_mode: bool = False
    output_dir: str = "outputs"

    # 实时模式设置
    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    show_fps: bool = True
    show_confidence: bool = True
    show_detection_source: bool = True
    realtime_processing: bool = True
    process_interval: float = 0.1
    max_queue_size: int = 10

    # 显示设置
    box_color_yolo: Tuple[int, int, int] = (0, 255, 0)
    box_color_traditional: Tuple[int, int, int] = (255, 0, 0)
    box_thickness: int = 2
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_background: Tuple[int, int, int] = (0, 0, 0)
    text_size: float = 0.5
    text_thickness: int = 1

    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.output_dir, exist_ok=True)


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = SystemConfig()

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, config_path: str):
        """从YAML文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            for key, value in yaml_config.items():
                if hasattr(self.config, key):
                    # 特殊处理颜色元组
                    if 'color' in key.lower() and isinstance(value, list):
                        value = tuple(value)
                    setattr(self.config, key, value)

            logger.info(f"配置已从 {config_path} 加载")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}, 使用默认配置")

    def save_config(self, config_path: str = None):
        """保存配置到YAML文件"""
        save_path = config_path or self.config_path
        if save_path:
            config_dict = self.config.__dict__.copy()
            # 转换颜色元组为列表以便YAML保存
            for key, value in config_dict.items():
                if isinstance(value, tuple):
                    config_dict[key] = list(value)

            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"配置已保存到 {save_path}")

    def get_config(self) -> SystemConfig:
        """获取配置对象"""
        return self.config
    