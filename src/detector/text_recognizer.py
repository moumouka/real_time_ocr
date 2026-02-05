"""
文本识别器（基于EasyOCR）
"""

import cv2
import easyocr
from src.config import Config


class TextRecognizer:
    def __init__(self, languages=Config.OCR_LANGUAGES):
        print("加载OCR模型...")
        self.reader = easyocr.Reader(languages)
        self.padding = Config.OCR_PADDING
        self.confidence_threshold = Config.OCR_CONFIDENCE

    def preprocess_text_region(self, text_region):
        """预处理文本区域"""
        if text_region.size == 0:
            return None

        # 转换为灰度图
        if len(text_region.shape) == 3:
            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = text_region

        # 可选：应用图像增强
        # gray = cv2.equalizeHist(gray)
        # gray = cv2.GaussianBlur(gray, (3, 3), 0)

        return gray

    def recognize(self, frame, bbox):
        """识别文本区域中的文字"""
        x1, y1, x2, y2 = bbox

        # 添加padding
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(frame.shape[1], x2 + self.padding)
        y2 = min(frame.shape[0], y2 + self.padding)

        # 裁剪文本区域
        text_region = frame[y1:y2, x1:x2]

        if text_region.size == 0:
            return "", 0.0

        try:
            # 预处理
            processed_region = self.preprocess_text_region(text_region)
            if processed_region is None:
                return "", 0.0

            # OCR识别
            results = self.reader.readtext(
                processed_region,
                detail=1,
                paragraph=False,
                batch_size=1  # 实时模式下使用小batch
            )

            if results:
                # 提取置信度最高的结果
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1]
                confidence = best_result[2]

                if confidence > self.confidence_threshold:
                    return text, confidence

        except Exception as e:
            print(f"OCR识别错误: {e}")

        return "", 0.0
