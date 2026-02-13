"""
文本检测器模块
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from .config import SystemConfig

logger = logging.getLogger(__name__)


class BaseDetector:
    """检测器基类"""

    def __init__(self, config: SystemConfig):
        self.config = config

    def detect(self, image: np.ndarray) -> List[List[int]]:
        """检测文本区域"""
        raise NotImplementedError


class TraditionalTextDetector(BaseDetector):
    """传统文本检测器"""

    def __init__(self, config: SystemConfig):
        super().__init__(config)

        try:
            self.mser = cv2.MSER_create(
                delta=5,
                min_area=self.config.min_text_area,
                max_area=self.config.max_text_area
            )
        except TypeError:
            try:
                self.mser = cv2.MSER_create(
                    _delta=5,
                    _min_area=self.config.min_text_area,
                    _max_area=self.config.max_text_area
                )
            except:
                logger.warning("MSER创建失败，将仅使用轮廓检测")
                self.mser = None

        # 形态学操作核
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect(self, image: np.ndarray) -> List[List[int]]:
        """使用传统方法检测文本区域"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 使用多种检测方法
        boxes1 = self._detect_adaptive_threshold(gray)
        boxes2 = self._detect_mser(gray) if self.mser else []

        # 合并结果
        all_boxes = boxes1 + boxes2

        # 非极大值抑制
        final_boxes = self._non_max_suppression(all_boxes)

        # 过滤无效区域
        filtered_boxes = self._filter_boxes(final_boxes, image.shape)

        logger.debug(f"传统方法检测到 {len(filtered_boxes)} 个文本区域")
        return filtered_boxes

    def _detect_adaptive_threshold(self, gray: np.ndarray) -> List[List[int]]:
        """自适应阈值方法"""
        try:
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                     self.kernel_medium, iterations=2)

            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.config.min_text_area or area > self.config.max_text_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / max(h, 1)

                if (20 < w < 1000 and 10 < h < 500 and
                        0.1 < aspect_ratio < 10):
                    boxes.append([x, y, x + w, y + h])

            return boxes
        except Exception as e:
            logger.error(f"自适应阈值检测失败: {e}")
            return []

    def _detect_mser(self, gray: np.ndarray) -> List[List[int]]:
        """MSER检测方法"""
        try:
            regions, _ = self.mser.detectRegions(gray)

            boxes = []
            for region in regions:
                if len(region) < 5:
                    continue

                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                area = w * h

                if area < self.config.min_text_area or area > self.config.max_text_area:
                    continue

                aspect_ratio = w / max(h, 1)
                if 0.2 < aspect_ratio < 5:
                    boxes.append([x, y, x + w, y + h])

            return boxes
        except Exception as e:
            logger.error(f"MSER检测失败: {e}")
            return []

    @staticmethod
    def _non_max_suppression(boxes: List[List[int]], iou_threshold: float = 0.5) -> List[List[int]]:
        """非极大值抑制"""
        if len(boxes) == 0:
            return []

        boxes_array = np.array(boxes, dtype=np.float32)

        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        indices = np.argsort(y1)
        keep = []

        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            rest_indices = indices[1:]

            xx1 = np.maximum(x1[current], x1[rest_indices])
            yy1 = np.maximum(y1[current], y1[rest_indices])
            xx2 = np.minimum(x2[current], x2[rest_indices])
            yy2 = np.minimum(y2[current], y2[rest_indices])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h

            union = areas[current] + areas[rest_indices] - intersection
            iou = intersection / union

            indices = indices[1:][iou < iou_threshold]

        return boxes_array[keep].astype(np.int32).tolist()

    @staticmethod
    def _filter_boxes(boxes: List[List[int]], image_shape: Tuple) -> List[List[int]]:
        """过滤无效框"""
        height, width = image_shape[:2]
        filtered = []

        for box in boxes:
            x1, y1, x2, y2 = box

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 > x1 and y2 > y1 and (x2 - x1) > 5 and (y2 - y1) > 5:
                w = x2 - x1
                h = y2 - y1
                aspect_ratio = w / max(h, 1)

                if 0.1 < aspect_ratio < 10:
                    filtered.append([int(x1), int(y1), int(x2), int(y2)])

        return filtered


class YOLOTextDetector(BaseDetector):
    """YOLO文本检测器"""

    def __init__(self, config: SystemConfig, model_path: str = 'models/yolo26n.pt'):
        super().__init__(config)
        self.model_path = model_path
        self.model = None
        self._init_model()

    def _init_model(self):
        """初始化YOLO模型"""
        try:
            from ultralytics import YOLO

            logger.info(f"加载YOLO模型: {self.model_path}")
            self.model = YOLO(self.model_path)

            # 设置设备
            device = 'cuda' if self.config.use_gpu else 'cpu'
            try:
                import torch
                if self.config.use_gpu and not torch.cuda.is_available():
                    logger.warning("CUDA不可用，将使用CPU")
                    device = 'cpu'
            except:
                device = 'cpu'

            self.model.to(device)
            logger.info(f"YOLO模型加载成功，使用设备: {device}")

        except Exception as e:
            logger.error(f"YOLO模型加载失败: {e}")
            self.model = None

    def detect(self, image: np.ndarray, conf_threshold: float = None) -> Tuple[List[List[int]], float]:
        """
        使用YOLO检测文本区域
        返回: (边界框列表, 平均置信度)
        """
        if self.model is None:
            return [], 0.0

        if conf_threshold is None:
            conf_threshold = self.config.confidence_threshold

        try:
            results = self.model(image, conf=conf_threshold, verbose=False)

            boxes = []
            confidences = []

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf.cpu().numpy()[0])

                        boxes.append(xyxy.tolist())
                        confidences.append(conf)

            avg_conf = np.mean(confidences) if confidences else 0.0
            logger.debug(f"YOLO检测到 {len(boxes)} 个区域，平均置信度: {avg_conf:.3f}")

            return boxes, avg_conf

        except Exception as e:
            logger.error(f"YOLO检测失败: {e}")
            return [], 0.0
