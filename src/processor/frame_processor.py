"""
帧处理器
"""

import cv2
import time
import threading
from src.config import Config


class FrameProcessor:
    def __init__(self, text_detector, text_recognizer):
        self.detector = text_detector
        self.recognizer = text_recognizer
        self.detection_interval = Config.DETECTION_INTERVAL
        self.frame_count = 0
        self.current_results = []
        self.lock = threading.Lock()
        self.scale = 1.0

    def preprocess_frame(self, frame):
        """预处理帧"""
        height, width = frame.shape[:2]
        target_width = Config.TARGET_WIDTH
        self.scale = target_width / width
        target_height = int(height * self.scale)

        resized = cv2.resize(frame, (target_width, target_height))
        return resized

    def scale_bbox_to_original(self, bbox):
        """将边界框坐标缩放回原始尺寸"""
        return [int(coord / self.scale) for coord in bbox]

    def process(self, frame):
        """处理单帧"""
        # 预处理
        processed_frame = self.preprocess_frame(frame)

        # 每隔N帧进行检测
        self.frame_count += 1
        if self.frame_count % self.detection_interval == 0:
            # 检测文本区域
            detections = self.detector.detect(processed_frame)

            # 识别文本
            results = []
            for detection in detections:
                # 调整坐标回原始尺寸
                bbox_original = self.scale_bbox_to_original(detection['bbox'])

                # 识别文本
                text, confidence = self.recognizer.recognize(frame, bbox_original)

                if text:
                    results.append({
                        'bbox': bbox_original,
                        'detection_confidence': detection['confidence'],
                        'text': text,
                        'ocr_confidence': confidence
                    })

            # 更新结果（线程安全）
            with self.lock:
                self.current_results = results

        return processed_frame

    def get_results(self):
        """获取当前检测结果"""
        with self.lock:
            return self.current_results.copy()

    def set_detection_interval(self, interval):
        """设置检测间隔"""
        self.detection_interval = interval
        print(f"检测间隔设置为: 每{interval}帧检测一次")

    def reset_frame_count(self):
        """重置帧计数"""
        self.frame_count = 0