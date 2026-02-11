import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import time
import os
import re
import yaml
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR结果数据类"""
    text: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    detection_source: str
    recognition_source: str = "EasyOCR"
    language: str = "zh_en"


@dataclass
class SystemConfig:
    """系统配置数据类"""
    use_yolo_first: bool = True
    fallback_enabled: bool = True
    confidence_threshold: float = 0.4
    nms_iou_threshold: float = 0.5
    use_gpu: bool = True
    languages: List[str] = None
    min_text_area: int = 100
    max_text_area: int = 100000
    debug_mode: bool = False
    output_dir: str = "ocr_results"

    def __post_init__(self):
        if self.languages is None:
            self.languages = ['ch_sim', 'en']
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
                    setattr(self.config, key, value)

            logger.info(f"配置已从 {config_path} 加载")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}, 使用默认配置")

    def save_config(self, config_path: str = None):
        """保存配置到YAML文件"""
        save_path = config_path or self.config_path
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config.__dict__, f, default_flow_style=False)
            logger.info(f"配置已保存到 {save_path}")


class SingletonMeta(type):
    """单例元类"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class TextRecognizerEasyOCR(metaclass=SingletonMeta):
    """
    EasyOCR识别器 (单例模式)
    """

    def __init__(self, config: SystemConfig = None):
        if config is None:
            config = SystemConfig()

        self.config = config
        self.reader = None
        self._init_reader()

    def _init_reader(self):
        """初始化EasyOCR阅读器"""
        try:
            # 设置GPU/CPU
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
        """
        识别文本区域
        输入: 原图 和 边界框列表
        输出: OCRResult列表
        """
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
                    logger.warning(f"跳过无效区域: {box}")
                    continue

                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    logger.warning(f"裁剪区域为空: {box}")
                    continue

                if self.config.debug_mode:
                    debug_path = os.path.join(self.config.output_dir, f"debug_crop_{x1}_{y1}.jpg")
                    cv2.imwrite(debug_path, crop)

                # 使用EasyOCR进行识别
                ocr_result = self.reader.readtext(
                    crop,
                    detail=1,
                    paragraph=False,
                    batch_size=4,
                    width_ths=0.7,
                    height_ths=0.7,
                    ycenter_ths=0.5
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

                    avg_conf = total_conf / len(ocr_result) if ocr_result else 0.0
                    combined_text = " ".join(all_texts)
                    processed_text = self._postprocess_text(combined_text)

                    results.append(OCRResult(
                        text=processed_text,
                        bbox=box,
                        confidence=float(avg_conf),
                        detection_source="Pending",  # 将在主流程中设置
                        recognition_source="EasyOCR"
                    ))
                else:
                    # 如果没有识别到文本，添加空结果
                    results.append(OCRResult(
                        text="",
                        bbox=box,
                        confidence=0.0,
                        detection_source="Pending",
                        recognition_source="EasyOCR"
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

        # 修复常见OCR错误
        replacements = {
            'O': '0',
            'l': '1',
            'I': '1',
            '|': '1',
            'Z': '2',
            'S': '5',
            'B': '8'
        }

        for wrong, correct in replacements.items():
            # 只在特定上下文中替换
            if re.match(r'^\d+$', text.replace(wrong, '')):
                text = text.replace(wrong, correct)

        return text


class TextDetectorTraditional:
    """
    传统文本检测器 (改进版)
    """

    def __init__(self, config: SystemConfig = None):
        if config is None:
            config = SystemConfig()

        self.config = config

        try:
            # 创建MSER检测器
            self.mser = cv2.MSER_create(
                delta=5,  # 修正参数名
                min_area=self.config.min_text_area,
                max_area=self.config.max_text_area
            )
        except TypeError as e:
            # 如果参数名不对，尝试不同版本的参数
            logger.warning(f"MSER参数错误: {e}, 尝试其他参数名")
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
        """
        使用传统方法检测文本区域
        返回: 边界框列表 [[x1, y1, x2, y2], ...]
        """
        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 方法1: 自适应阈值 + 形态学
        boxes1 = self._detect_adaptive_threshold(gray)

        # 方法2: MSER检测（如果可用）
        boxes2 = []
        if self.mser is not None:
            boxes2 = self._detect_mser(gray)

        # 方法3: Canny边缘检测
        boxes3 = self._detect_canny(gray)

        # 合并所有方法的结果
        all_boxes = boxes1 + boxes2 + boxes3

        # 非极大值抑制
        final_boxes = self._non_max_suppression(all_boxes)

        # 过滤无效区域
        filtered_boxes = self._filter_boxes(final_boxes, image.shape)

        logger.info(f"传统方法检测到 {len(filtered_boxes)} 个文本区域")
        return filtered_boxes

    def _detect_adaptive_threshold(self, gray: np.ndarray) -> List[List[int]]:
        """自适应阈值方法"""
        try:
            # 使用自适应阈值
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # 形态学操作增强文本区域
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel_medium, iterations=2)

            # 查找轮廓
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.config.min_text_area or area > self.config.max_text_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / max(h, 1)

                # 过滤条件
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
            if self.mser is None:
                return []

            regions, _ = self.mser.detectRegions(gray)

            boxes = []
            for region in regions:
                if len(region) < 5:  # 需要足够多的点
                    continue

                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                area = w * h

                if area < self.config.min_text_area or area > self.config.max_text_area:
                    continue

                aspect_ratio = w / max(h, 1)
                if 0.2 < aspect_ratio < 5:  # 文本通常不是极端的长宽比
                    boxes.append([x, y, x + w, y + h])

            return boxes
        except Exception as e:
            logger.error(f"MSER检测失败: {e}")
            return []

    def _detect_canny(self, gray: np.ndarray) -> List[List[int]]:
        """Canny边缘检测方法"""
        try:
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)

            # 膨胀连接边缘
            dilated = cv2.dilate(edges, self.kernel_small, iterations=2)

            # 查找轮廓
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.config.min_text_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / max(h, 1)

                # 文本区域的典型特征
                if (w > 10 and h > 10 and 0.2 < aspect_ratio < 10 and
                        w * h < self.config.max_text_area):
                    boxes.append([x, y, x + w, y + h])

            return boxes
        except Exception as e:
            logger.error(f"Canny检测失败: {e}")
            return []

    @staticmethod
    def _non_max_suppression(boxes: List[List[int]], iou_threshold: float = 0.5) -> List[List[int]]:
        """非极大值抑制"""
        if len(boxes) == 0:
            return []

        boxes_array = np.array(boxes, dtype=np.float32)

        # 计算每个框的面积
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 按y1坐标排序（从上到下）
        indices = np.argsort(y1)
        keep = []

        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            rest_indices = indices[1:]

            # 计算当前框与其他框的交集
            xx1 = np.maximum(x1[current], x1[rest_indices])
            yy1 = np.maximum(y1[current], y1[rest_indices])
            xx2 = np.minimum(x2[current], x2[rest_indices])
            yy2 = np.minimum(y2[current], y2[rest_indices])

            # 计算交集区域的宽和高
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h

            # 计算IoU
            union = areas[current] + areas[rest_indices] - intersection
            iou = intersection / union

            # 保留IoU小于阈值的框
            indices = indices[1:][iou < iou_threshold]

        return boxes_array[keep].astype(np.int32).tolist()

    @staticmethod
    def _filter_boxes(boxes: List[List[int]], image_shape: Tuple) -> List[List[int]]:
        """过滤无效框"""
        height, width = image_shape[:2]
        filtered = []

        for box in boxes:
            x1, y1, x2, y2 = box

            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # 确保有效区域
            if x2 > x1 and y2 > y1 and (x2 - x1) > 5 and (y2 - y1) > 5:
                # 检查宽高比
                w = x2 - x1
                h = y2 - y1
                aspect_ratio = w / max(h, 1)

                # 合理的文本宽高比范围
                if 0.1 < aspect_ratio < 10:
                    filtered.append([int(x1), int(y1), int(x2), int(y2)])

        return filtered


class TextDetectorYOLO:
    """
    YOLO文本检测器
    """

    def __init__(self, model_path: str = 'yolov8s.pt', config: SystemConfig = None):
        if config is None:
            config = SystemConfig()

        self.config = config
        self.model_path = model_path
        self.model = None
        self._init_model()

    def _init_model(self):
        """初始化YOLO模型"""
        try:
            logger.info(f"加载YOLO模型: {self.model_path}")

            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                logger.warning(f"模型文件不存在: {self.model_path}")
                # 使用YOLOv8的自动下载功能
                logger.info("尝试下载YOLOv8模型...")
                try:
                    self.model = YOLO(self.model_path)  # 这会自动下载
                except:
                    # 如果指定的模型下载失败，尝试其他模型
                    logger.info("尝试下载yolov8n.pt...")
                    self.model = YOLO('yolov8n.pt')
            else:
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
            logger.warning("YOLO模型未加载，尝试重新初始化")
            self._init_model()
            if self.model is None:
                return [], 0.0

        if conf_threshold is None:
            conf_threshold = self.config.confidence_threshold

        try:
            # 运行推理
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

            logger.info(f"YOLO检测到 {len(boxes)} 个区域，平均置信度: {avg_conf:.3f}")
            return boxes, avg_conf

        except Exception as e:
            logger.error(f"YOLO检测失败: {e}")
            return [], 0.0


class ModularOCR:
    """
    主系统: 协调各模块，实现自动降级
    """

    def __init__(self, config: SystemConfig = None, config_path: str = None):
        # 加载配置
        if config is None:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
        else:
            self.config = config

        # 初始化子模块（延迟加载）
        self.detector_yolo = None
        self.detector_traditional = None
        self.recognizer = None

        # 性能统计
        self.stats = {
            'total_images': 0,
            'yolo_success': 0,
            'traditional_fallback': 0,
            'traditional_forced': 0,
            'total_time': 0.0
        }

        logger.info(f"OCR系统初始化完成，配置: {self.config}")

    def _load_yolo_detector(self):
        """延迟加载YOLO检测器"""
        if self.detector_yolo is None:
            try:
                self.detector_yolo = TextDetectorYOLO(config=self.config)
            except Exception as e:
                logger.error(f"无法加载YOLO检测器: {e}")
                self.detector_yolo = None

    def _load_traditional_detector(self):
        """延迟加载传统检测器"""
        if self.detector_traditional is None:
            self.detector_traditional = TextDetectorTraditional(config=self.config)

    def _load_recognizer(self):
        """延迟加载识别器"""
        if self.recognizer is None:
            self.recognizer = TextRecognizerEasyOCR(config=self.config)

    def run(self, image_input: Union[str, np.ndarray]) -> List[OCRResult]:
        """
        运行OCR流程
        输入: 图像路径或numpy数组
        输出: OCRResult列表
        """
        start_time = time.time()
        self.stats['total_images'] += 1

        # 读取图像
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图像不存在: {image_input}")

            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"无法读取图像: {image_input}")
            image_path = image_input
        else:
            image = image_input.copy()
            image_path = "memory_image"

        logger.info(f"开始处理图像，尺寸: {image.shape}")

        # === 第一阶段：文本检测 ===
        detection_method = "Unknown"
        boxes = []

        if self.config.use_yolo_first:
            self._load_yolo_detector()

            if self.detector_yolo and self.detector_yolo.model is not None:
                logger.info("使用YOLO进行文本检测...")
                boxes, avg_conf = self.detector_yolo.detect(image)

                # 判断是否需要降级
                if len(boxes) == 0 or avg_conf < self.config.confidence_threshold:
                    if self.config.fallback_enabled:
                        logger.warning(f"YOLO检测效果不佳 (置信度: {avg_conf:.3f})，启用传统方法...")
                        self._load_traditional_detector()
                        boxes = self.detector_traditional.detect(image)
                        detection_method = "Traditional_Fallback"
                        self.stats['traditional_fallback'] += 1
                    else:
                        logger.info("降级已禁用，使用YOLO结果...")
                        detection_method = "YOLO_LowQuality"
                        self.stats['yolo_success'] += 1
                else:
                    detection_method = "YOLO_Success"
                    self.stats['yolo_success'] += 1
            else:
                # YOLO加载失败，强制使用传统方法
                logger.warning("YOLO不可用，强制使用传统方法...")
                self._load_traditional_detector()
                boxes = self.detector_traditional.detect(image)
                detection_method = "Traditional_Forced"
                self.stats['traditional_forced'] += 1
        else:
            # 强制使用传统方法
            logger.info("强制使用传统方法进行检测...")
            self._load_traditional_detector()
            boxes = self.detector_traditional.detect(image)
            detection_method = "Traditional_Forced"
            self.stats['traditional_forced'] += 1

        if len(boxes) == 0:
            logger.warning("未检测到文本区域")
            return []

        logger.info(f"检测完成，找到 {len(boxes)} 个文本区域，方法: {detection_method}")

        # === 第二阶段：文本识别 ===
        logger.info("开始文本识别...")
        self._load_recognizer()
        recognition_results = self.recognizer.recognize(image, boxes)

        # 设置检测来源
        for result in recognition_results:
            result.detection_source = detection_method

        # 计算处理时间
        processing_time = time.time() - start_time
        self.stats['total_time'] += processing_time

        logger.info(f"处理完成，耗时: {processing_time:.2f}秒，识别到 {len(recognition_results)} 个文本")

        # 保存结果（如果是文件输入）
        if isinstance(image_input, str):
            self.save_single_result(image_input, recognition_results)

        return recognition_results

    def save_single_result(self, image_path: str, results: List[OCRResult]):
        """保存单个图像的结果"""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        # 保存文本结果
        self.save_text_results(image_path, results)

        # 保存可视化结果
        if self.config.debug_mode:
            self.visualize_results(image_path, results)

    def batch_process(self, image_paths: List[str],
                      output_dir: str = None) -> Dict[str, List[OCRResult]]:
        """
        批量处理图像
        """
        if output_dir is None:
            output_dir = self.config.output_dir

        os.makedirs(output_dir, exist_ok=True)

        all_results = {}
        for img_path in image_paths:
            try:
                logger.info(f"处理图像: {img_path}")
                results = self.run(img_path)
                all_results[img_path] = results

            except Exception as e:
                logger.error(f"处理 {img_path} 时出错: {e}")
                all_results[img_path] = []

        # 保存统计信息
        self.save_statistics(output_dir)

        return all_results

    def visualize_results(self, image_path: str, results: List[OCRResult],
                          output_dir: str = None) -> str:
        """
        可视化结果
        返回: 可视化图像路径
        """
        if output_dir is None:
            output_dir = self.config.output_dir

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像用于可视化: {image_path}")
            return ""

        vis_image = image.copy()

        # 定义颜色映射
        color_map = {
            "YOLO_Success": (0, 255, 0),  # 绿色
            "YOLO_LowQuality": (0, 165, 255),  # 橙色
            "Traditional_Fallback": (255, 0, 0),  # 蓝色
            "Traditional_Forced": (128, 0, 128)  # 紫色
        }

        for i, result in enumerate(results):
            box = result.bbox
            text = result.text[:20]  # 只显示前20个字符
            source = result.detection_source
            conf = result.confidence

            # 获取颜色
            color = color_map.get(source, (255, 255, 255))

            # 绘制边界框
            cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), color, 2)

            # 绘制标签背景
            label = f"{i}: {text} ({conf:.2f})" if text else f"{i}: 空"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 确保标签在图像内
            label_x = box[0]
            label_y = box[1] - 10
            if label_y < 20:
                label_y = box[1] + label_height + 10

            cv2.rectangle(vis_image,
                          (label_x, label_y - label_height - baseline),
                          (label_x + label_width, label_y),
                          color, -1)

            # 绘制标签文本
            cv2.putText(vis_image, label,
                        (label_x, label_y - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存可视化图像
        img_name = Path(image_path).stem
        vis_path = os.path.join(output_dir, f"{img_name}_visualized.jpg")
        cv2.imwrite(vis_path, vis_image)

        logger.info(f"可视化结果已保存: {vis_path}")
        return vis_path

    def save_text_results(self, image_path: str, results: List[OCRResult],
                          output_dir: str = None) -> str:
        """
        保存文本结果到文件
        返回: 文件路径
        """
        if output_dir is None:
            output_dir = self.config.output_dir

        img_name = Path(image_path).stem
        txt_path = os.path.join(output_dir, f"{img_name}_results.txt")
        json_path = os.path.join(output_dir, f"{img_name}_results.json")

        # 保存为文本文件
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"图像: {image_path}\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"检测到 {len(results)} 个文本区域\n")
            f.write("=" * 50 + "\n\n")

            for i, result in enumerate(results):
                f.write(f"[区域 {i + 1}]\n")
                f.write(f"文本: {result.text}\n")
                f.write(f"边界框: {result.bbox}\n")
                f.write(f"置信度: {result.confidence:.3f}\n")
                f.write(f"检测方法: {result.detection_source}\n")
                f.write(f"识别方法: {result.recognition_source}\n")
                f.write("-" * 30 + "\n")

        # 保存为JSON文件
        results_dict = {
            'image_path': image_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': [
                {
                    'text': r.text,
                    'bbox': r.bbox,
                    'confidence': float(r.confidence),
                    'detection_source': r.detection_source,
                    'recognition_source': r.recognition_source
                }
                for r in results
            ]
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"文本结果已保存: {txt_path}, {json_path}")
        return txt_path

    def save_statistics(self, output_dir: str = None):
        """保存统计信息"""
        if output_dir is None:
            output_dir = self.config.output_dir

        stats_path = os.path.join(output_dir, "statistics.txt")

        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("OCR系统统计信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理图像总数: {self.stats['total_images']}\n")
            f.write(f"YOLO成功次数: {self.stats['yolo_success']}\n")
            f.write(f"传统方法降级次数: {self.stats['traditional_fallback']}\n")
            f.write(f"传统方法强制次数: {self.stats['traditional_forced']}\n")

            if self.stats['total_images'] > 0:
                yolo_success_rate = self.stats['yolo_success'] / self.stats['total_images'] * 100
                fallback_rate = self.stats['traditional_fallback'] / self.stats['total_images'] * 100
                avg_time = self.stats['total_time'] / self.stats['total_images']

                f.write(f"YOLO成功率: {yolo_success_rate:.1f}%\n")
                f.write(f"降级率: {fallback_rate:.1f}%\n")
                f.write(f"平均处理时间: {avg_time:.2f}秒/图像\n")
                f.write(f"总处理时间: {self.stats['total_time']:.2f}秒\n")

        logger.info(f"统计信息已保存: {stats_path}")

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return self.stats.copy()


def create_sample_config():
    """创建示例配置文件"""
    config = {
        'use_yolo_first': True,
        'fallback_enabled': True,
        'confidence_threshold': 0.4,
        'nms_iou_threshold': 0.5,
        'use_gpu': False,  # 默认使用CPU，避免CUDA问题
        'languages': ['ch_sim', 'en'],
        'min_text_area': 100,
        'max_text_area': 100000,
        'debug_mode': False,
        'output_dir': 'ocr_results'
    }

    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print("示例配置文件已创建: config.yaml")


# =================== 简化版运行示例 ===================
if __name__ == "__main__":
    print("=" * 60)
    print("模块化OCR系统 v2.0")
    print("=" * 60)

    # 创建示例配置文件（首次运行时）
    if not os.path.exists("config.yaml"):
        create_sample_config()
        print("\n请编辑 config.yaml 文件调整配置，然后重新运行程序")
        exit(0)

    # 创建输出目录
    os.makedirs("ocr_results", exist_ok=True)

    # 测试图片列表
    test_images = [
        "document_test_result.jpg",
        "example_document.jpg",
        "test_image.png",
        "demo_image.jpg",
        "sample.jpg"
    ]

    # 仅保留实际存在的图片
    existing_images = []
    for img in test_images:
        if os.path.exists(img):
            existing_images.append(img)

    if not existing_images:
        # 创建演示图像
        print("\n未找到测试图像，创建演示图像...")
        demo_image = np.ones((400, 600, 3), dtype=np.uint8) * 240

        # 添加一些文本
        cv2.putText(demo_image, "Hello World!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(demo_image, "你好，世界！", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(demo_image, "OCR Test 123456", (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        cv2.imwrite("demo_image.jpg", demo_image)
        print("演示图像已创建: demo_image.jpg")
        existing_images = ["demo_image.jpg"]

    try:
        # 创建OCR系统实例
        print(f"\n初始化OCR系统...")
        ocr_system = ModularOCR(config_path="config.yaml")

        print(f"\n找到 {len(existing_images)} 张测试图像")

        # 单张图片处理演示
        test_image = existing_images[0]
        print(f"\n处理单张图片: {test_image}")

        start_time = time.time()
        results = ocr_system.run(test_image)
        processing_time = time.time() - start_time

        # 显示结果
        print(f"\n=== 识别结果 ({len(results)} 个文本区域) ===")
        print(f"处理时间: {processing_time:.2f}秒")
        print("-" * 60)

        for i, result in enumerate(results):
            if result.text.strip():  # 只显示非空结果
                print(f"[{i + 1}] 文本: {result.text}")
                print(f"    置信度: {result.confidence:.3f}")
                print(f"    检测方法: {result.detection_source}")
                print(f"    边界框: {result.bbox}")
                print("-" * 40)

        # 批量处理（如果有多张图片）
        if len(existing_images) > 1:
            print(f"\n批量处理 {len(existing_images)} 张图片...")
            print("批量处理中，请稍候...")

            all_results = ocr_system.batch_process(existing_images)

            # 显示批量处理统计
            success_count = sum(1 for img, res in all_results.items() if len(res) > 0)
            print(f"\n批量处理完成:")
            print(f"  成功处理: {success_count}/{len(existing_images)} 张图片")
            print(f"  结果保存在: {ocr_system.config.output_dir}")

        # 显示统计信息
        stats = ocr_system.get_performance_stats()
        print(f"\n=== 性能统计 ===")
        print(f"处理图像总数: {stats['total_images']}")

        if stats['total_images'] > 0:
            yolo_rate = stats['yolo_success'] / stats['total_images'] * 100
            fallback_rate = stats['traditional_fallback'] / stats['total_images'] * 100
            avg_time = stats['total_time'] / stats['total_images']

            print(f"YOLO成功率: {yolo_rate:.1f}%")
            print(f"降级率: {fallback_rate:.1f}%")
            print(f"平均处理时间: {avg_time:.2f}秒/图像")
            print(f"总处理时间: {stats['total_time']:.2f}秒")

        print("\n处理完成！")

    except Exception as e:
        print(f"\n处理出错: {e}")
        import traceback

        traceback.print_exc()

        print("\n常见问题解决:")
        print("1. 确保已安装所有依赖: pip install opencv-python numpy easyocr ultralytics pyyaml")
        print("2. 如果CUDA不可用，请在config.yaml中将use_gpu改为false")
        print("3. 确保有足够的磁盘空间下载模型")


# =================== 快速使用示例 ===================
def quick_ocr_demo():
    """
    快速使用示例
    """
    # 方法1: 最简单的方式
    ocr = ModularOCR()
    results = ocr.run("your_image.jpg")

    # 方法2: 使用自定义配置
    config = SystemConfig(
        use_yolo_first=True,
        confidence_threshold=0.3,
        use_gpu=False  # 如果没有GPU
    )
    ocr = ModularOCR(config=config)

    # 方法3: 处理numpy数组
    image = cv2.imread("your_image.jpg")
    results = ocr.run(image)

    # 方法4: 批量处理
    image_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
    all_results = ocr.batch_process(image_list)

    return results


# =================== 最小依赖版本 ===================
class MinimalOCR:
    """
    最小依赖版本，仅使用OpenCV和EasyOCR
    """

    def __init__(self, lang=['ch_sim', 'en'], use_gpu=False):
        self.reader = easyocr.Reader(lang, gpu=use_gpu)

    def detect_and_recognize(self, image_path):
        """简化的检测和识别"""
        image = cv2.imread(image_path)

        # 直接使用EasyOCR的检测和识别
        results = self.reader.readtext(image)

        # 格式化结果
        formatted = []
        for (bbox, text, conf) in results:
            formatted.append({
                'text': text,
                'bbox': bbox,
                'confidence': conf
            })

        return formatted


if __name__ == "__main__":
    # 如果上面的主程序有问题，可以尝试最小版本
    # minimal_ocr = MinimalOCR()
    # results = minimal_ocr.detect_and_recognize("test.jpg")
    pass