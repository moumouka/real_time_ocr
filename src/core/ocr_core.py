"""
OCR核心引擎模块
"""
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass

from .config import SystemConfig, ConfigManager
from .detectors import TraditionalTextDetector, YOLOTextDetector
from .recognizers import EasyOCRRecognizer, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """性能统计"""
    total_frames: int = 0
    total_images: int = 0
    yolo_success: int = 0
    traditional_fallback: int = 0
    traditional_forced: int = 0
    total_time: float = 0.0
    avg_fps: float = 0.0


class OCRCore:
    """OCR核心引擎"""

    def __init__(self, config: SystemConfig = None, config_path: str = None):
        # 加载配置
        if config is None:
            config_manager = ConfigManager(config_path)
            self.config = config_manager.get_config()
        else:
            self.config = config

        # 初始化子模块（延迟加载）
        self.detector_yolo = None
        self.detector_traditional = None
        self.recognizer = None

        # 性能统计
        self.stats = PerformanceStats()

        logger.info("OCR核心引擎初始化完成")

    def _load_yolo_detector(self):
        """延迟加载YOLO检测器"""
        if self.detector_yolo is None:
            try:
                self.detector_yolo = YOLOTextDetector(self.config)
            except Exception as e:
                logger.error(f"无法加载YOLO检测器: {e}")
                self.detector_yolo = None

    def _load_traditional_detector(self):
        """延迟加载传统检测器"""
        if self.detector_traditional is None:
            self.detector_traditional = TraditionalTextDetector(self.config)

    def _load_recognizer(self):
        """延迟加载识别器"""
        if self.recognizer is None:
            self.recognizer = EasyOCRRecognizer(self.config)

    def detect_text(self, image: np.ndarray) -> Tuple[List[List[int]], str]:
        """文本检测（不包含识别）"""
        detection_method = "Unknown"
        boxes = []

        if self.config.use_yolo_first:
            self._load_yolo_detector()

            if self.detector_yolo and self.detector_yolo.model is not None:
                boxes, avg_conf = self.detector_yolo.detect(image)

                if len(boxes) == 0 or avg_conf < self.config.confidence_threshold:
                    if self.config.fallback_enabled:
                        self._load_traditional_detector()
                        boxes = self.detector_traditional.detect(image)
                        detection_method = "Traditional_Fallback"
                        self.stats.traditional_fallback += 1
                    else:
                        detection_method = "YOLO_LowQuality"
                        self.stats.yolo_success += 1
                else:
                    detection_method = "YOLO_Success"
                    self.stats.yolo_success += 1
            else:
                self._load_traditional_detector()
                boxes = self.detector_traditional.detect(image)
                detection_method = "Traditional_Forced"
                self.stats.traditional_forced += 1
        else:
            self._load_traditional_detector()
            boxes = self.detector_traditional.detect(image)
            detection_method = "Traditional_Forced"
            self.stats.traditional_forced += 1

        return boxes, detection_method

    def recognize_text(self, image: np.ndarray, boxes: List[List[int]],
                       detection_method: str) -> List[OCRResult]:
        """文本识别"""
        if len(boxes) == 0:
            return []

        self._load_recognizer()
        recognition_results = self.recognizer.recognize(image, boxes)

        # 设置检测来源
        for result in recognition_results:
            result.detection_source = detection_method

        return recognition_results

    def process_image(self, image: np.ndarray, confidence_threshold: float = None) -> List[OCRResult]:
        """处理单张图片 - 支持置信度过滤"""
        start_time = time.time()
        self.stats.total_images += 1

        # 使用传入的阈值或配置的阈值
        if confidence_threshold is None:
            confidence_threshold = self.config.confidence_threshold

        # 文本检测
        boxes, detection_method = self.detect_text(image)

        if len(boxes) == 0:
            return []

        # 文本识别
        results = self.recognize_text(image, boxes, detection_method)

        # 过滤低置信度的结果
        filtered_results = [r for r in results if r.confidence >= confidence_threshold]

        # 更新统计
        processing_time = time.time() - start_time
        self.stats.total_time += processing_time

        logger.info(f"图片处理完成，耗时: {processing_time:.2f}秒，"
                    f"识别到 {len(results)} 个文本，"
                    f"过滤后 {len(filtered_results)} 个文本 (阈值: {confidence_threshold:.2f})")

        return filtered_results

    def draw_results(self, frame: np.ndarray, results: List[OCRResult]) -> np.ndarray:
        """在图像上绘制识别结果"""
        import cv2

        display_frame = frame.copy()

        for result in results:
            box = result.bbox
            text = result.text
            source = result.detection_source
            conf = result.confidence

            # 根据检测方法选择颜色
            if "YOLO" in source:
                color = self.config.box_color_yolo
            else:
                color = self.config.box_color_traditional

            # 绘制边界框
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]),
                          color, self.config.box_thickness)

            # 准备标签文本
            label_parts = []
            if text:
                label_parts.append(text[:20])  # 最多显示20个字符

            if self.config.show_confidence:
                label_parts.append(f"{conf:.2f}")

            if self.config.show_detection_source:
                label_parts.append(source)

            label = " | ".join(label_parts)

            if label:
                # 计算文本大小
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.config.text_size,
                    self.config.text_thickness
                )

                # 标签位置
                label_x = box[0]
                label_y = box[1] - 10
                if label_y < 20:
                    label_y = box[3] + label_height + 10

                # 绘制文本背景
                cv2.rectangle(display_frame,
                              (label_x, label_y - label_height - baseline),
                              (label_x + label_width, label_y),
                              self.config.text_background, -1)

                # 绘制文本
                cv2.putText(display_frame, label,
                            (label_x, label_y - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, self.config.text_size,
                            self.config.text_color, self.config.text_thickness)

        return display_frame

    def get_statistics(self) -> PerformanceStats:
        """获取统计信息"""
        return self.stats
