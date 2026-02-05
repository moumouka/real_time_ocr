"""
文本检测器（基于YOLO）
"""

import torch
from ultralytics import YOLO
from src.config import Config


class TextDetector:
    def __init__(self, model_path=Config.YOLO_MODEL):
        print("加载YOLO模型...")
        self.model = YOLO(model_path)
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            print(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
            self.model.to('cuda')
        else:
            print("使用CPU")

    def detect(self, frame, confidence_threshold=Config.DETECTION_CONFIDENCE):
        """检测文本区域"""
        try:
            # YOLO推理
            results = self.model(frame, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()

                        if confidence > confidence_threshold:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence)
                            })

            return detections
        except Exception as e:
            print(f"文本检测错误: {e}")
            return []
