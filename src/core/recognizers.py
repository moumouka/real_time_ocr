"""
文本识别器模块
"""
import re
import numpy as np
from typing import List, Optional
import logging
from dataclasses import dataclass
from .config import SystemConfig

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR结果数据类"""
    text: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    detection_source: str
    recognition_source: str = "EasyOCR"
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()


class SingletonMeta(type):
    """单例元类"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class EasyOCRRecognizer(metaclass=SingletonMeta):
    """EasyOCR识别器"""

    def __init__(self, config: SystemConfig = None):
        if config is None:
            from .config import SystemConfig
            config = SystemConfig()

        self.config = config
        self.reader = None
        self._init_reader()

    def _init_reader(self):
        """初始化EasyOCR阅读器"""
        try:
            import easyocr

            gpu = self.config.use_gpu
            logger.info(f"初始化EasyOCR, 使用GPU: {gpu}, 语言: {self.config.languages}")

            # 检查CUDA是否可用
            if gpu:
                try:
                    import torch
                    gpu = torch.cuda.is_available()
                    if not gpu:
                        logger.warning("CUDA不可用，将使用CPU")
                except:
                    gpu = False

            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=gpu,
                model_storage_directory='./models',
                download_enabled=True,
                recog_network='standard'
            )
            logger.info("EasyOCR初始化成功")

        except Exception as e:
            logger.error(f"EasyOCR初始化失败: {e}")
            raise

    def recognize(self, image: np.ndarray, boxes: List[List[int]]) -> List[OCRResult]:
        """识别文本区域"""
        if self.reader is None:
            self._init_reader()

        results = []
        for i, box in enumerate(boxes):
            try:
                # 裁剪图像区域
                x1 = max(0, box[0])
                y1 = max(0, box[1])
                x2 = min(image.shape[1], box[2])
                y2 = min(image.shape[0], box[3])

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # 使用EasyOCR进行识别
                ocr_result = self.reader.readtext(
                    crop,
                    detail=1,
                    paragraph=False,
                    batch_size=4
                )

                if ocr_result:
                    # 合并多个检测结果
                    all_texts = []
                    total_conf = 0
                    for item in ocr_result:
                        text = item[1]
                        conf = item[2]
                        all_texts.append(text)
                        total_conf += conf

                    avg_conf = total_conf / len(ocr_result)
                    combined_text = " ".join(all_texts)
                    processed_text = self._postprocess_text(combined_text)

                    results.append(OCRResult(
                        text=processed_text,
                        bbox=box,
                        confidence=float(avg_conf),
                        detection_source="Pending"
                    ))
                else:
                    # 如果没有识别到文本，添加空结果
                    results.append(OCRResult(
                        text="",
                        bbox=box,
                        confidence=0.0,
                        detection_source="Pending"
                    ))

            except Exception as e:
                logger.error(f"识别区域 {box} 时出错: {e}")
                continue

        return results

    @staticmethod
    def _postprocess_text(text: str) -> str:
        """文本后处理"""
        if not text:
            return text

        # 去除特殊字符但保留中英文和常见标点
        text = re.sub(r'[^\w\u4e00-\u9fff\s.,，。!?！？:："\'\-()（）]', '', text)

        # 合并多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    