"""
文本识别器（基于EasyOCR）- 增强版
"""

import cv2
import numpy as np
import easyocr
from src.config import Config


class EnhancedTextRecognizer:
    def __init__(self, languages=Config.OCR_LANGUAGES):
        print("加载OCR模型...")
        self.reader = easyocr.Reader(
            languages,
            gpu=True,  # 使用GPU加速
            model_storage_directory='./models',  # 自定义模型路径
            download_enabled=True  # 允许自动下载模型
        )
        self.padding = Config.OCR_PADDING
        self.confidence_threshold = Config.OCR_CONFIDENCE
        self.min_text_height = 20  # 最小文本高度（像素）
        self.max_text_height = 300  # 最大文本高度（像素）

    def adaptive_padding(self, bbox, frame_shape):
        """根据文本区域大小自适应padding"""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1

        # 根据文本大小动态调整padding
        if height < 30:
            padding = max(3, int(height * 0.3))  # 小文本需要更多padding
        elif height > 100:
            padding = min(10, int(height * 0.1))  # 大文本减少padding
        else:
            padding = self.padding

        return max(0, x1 - padding), max(0, y1 - padding), \
            min(frame_shape[1], x2 + padding), min(frame_shape[0], y2 + padding)

    def preprocess_text_region(self, text_region):
        """增强的文本区域预处理"""
        if text_region.size == 0:
            return None

        # 转换为灰度图
        if len(text_region.shape) == 3:
            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = text_region

        # 1. 尺寸检查 - 如果太小，先放大
        h, w = gray.shape
        if h < self.min_text_height or w < self.min_text_height:
            scale_factor = max(self.min_text_height / h, self.min_text_height / w, 2.0)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 2. 降噪
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # 3. 对比度增强 - CLAHE（限制对比度自适应直方图均衡化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 4. 锐化
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # 5. 二值化 - 自适应阈值
        binary = cv2.adaptiveThreshold(sharpened, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # 6. 形态学操作 - 去除小噪点，连接断裂的文字
        kernel_size = max(1, min(h, w) // 50)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        return cleaned

    def estimate_text_orientation(self, text_region):
        """估计文本方向（针对倾斜文本）"""
        if len(text_region.shape) == 3:
            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = text_region

        # Sobel算子检测边缘
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 计算角度
        angles = np.arctan2(np.abs(sobel_y), np.abs(sobel_x)) * 180 / np.pi
        avg_angle = np.mean(angles)

        # 如果倾斜角度大于阈值，返回需要旋转的角度
        if 75 < avg_angle < 105:  # 接近垂直
            return 90
        elif avg_angle > 45:  # 明显倾斜
            return avg_angle - 45
        return 0

    def rotate_text_region(self, text_region, angle):
        """旋转文本区域到水平"""
        if angle == 0:
            return text_region

        h, w = text_region.shape[:2]
        center = (w // 2, h // 2)

        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算旋转后的边界，避免裁剪
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # 调整旋转矩阵
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # 执行旋转
        rotated = cv2.warpAffine(text_region, M, (new_w, new_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def recognize_with_postprocessing(self, text, confidence):
        """识别后处理 - 提高文本质量"""
        if not text:
            return text, confidence

        # 1. 去除多余空格
        text = ' '.join(text.split())

        # 2. 常见OCR错误修正（针对中英文）
        correction_map = {
            'O': '0', 'o': '0', 'I': '1', 'l': '1', 'Z': '2',
            'S': '5', 'B': '8', '０': '0', '１': '1', '２': '2',
            '３': '3', '４': '4', '５': '5', '６': '6', '７': '7',
            '８': '8', '９': '9'
        }

        corrected_text = ''
        for char in text:
            corrected_text += correction_map.get(char, char)

        # 3. 长度过滤 - 过滤掉太短的文本（可能是噪声）
        if len(corrected_text) < 2:
            return '', 0.0

        # 根据修正程度调整置信度
        if corrected_text != text:
            confidence *= 0.95  # 修正后置信度稍微降低

        return corrected_text, confidence

    def recognize_multiple_orientations(self, processed_region):
        """尝试多个方向的识别（针对不确定方向的文本）"""
        best_text = ""
        best_confidence = 0.0

        # 尝试多个角度
        for angle in [0, 90, 180, 270]:
            rotated = self.rotate_text_region(processed_region, angle)

            results = self.reader.readtext(
                rotated,
                detail=1,
                paragraph=False,
                batch_size=1,
                allowlist=None,  # 不限制字符集
                # width_ths=0.5,  # 放宽合并阈值
                # add_margin=0.1   # 增加边距
            )

            if results:
                for result in results:
                    text, bbox, confidence = result[1], result[0], result[2]
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_text = text

        return best_text, best_confidence

    def recognize(self, frame, bbox):
        """增强的文本识别方法"""
        # 自适应padding
        x1, y1, x2, y2 = self.adaptive_padding(bbox, frame.shape)

        # 裁剪文本区域
        text_region = frame[y1:y2, x1:x2]

        if text_region.size == 0:
            return "", 0.0

        try:
            # 1. 估计文本方向
            orientation = self.estimate_text_orientation(text_region)

            # 2. 预处理
            processed_region = self.preprocess_text_region(text_region)
            if processed_region is None:
                return "", 0.0

            # 3. 如果有明显倾斜，先旋转
            if abs(orientation) > 10:
                processed_region = self.rotate_text_region(processed_region, -orientation)

            # 4. OCR识别 - 尝试两种方法
            text, confidence = "", 0.0

            # 方法1: 正常识别
            results = self.reader.readtext(
                processed_region,
                detail=1,
                paragraph=True,  # 使用段落模式（对于多行文本更好）
                batch_size=1,
                allowlist=None,
                decoder='beamsearch',  # 使用beam search解码器
                beamWidth=5  # beam宽度
            )

            if results:
                # 取置信度最高的结果
                best_result = max(results, key=lambda x: x[2])
                text, confidence = best_result[1], best_result[2]

            # 方法2: 如果方法1失败或置信度低，尝试多方向识别
            if confidence < self.confidence_threshold * 0.8:
                alt_text, alt_confidence = self.recognize_multiple_orientations(processed_region)
                if alt_confidence > confidence:
                    text, confidence = alt_text, alt_confidence

            # 5. 后处理
            if confidence > self.confidence_threshold:
                text, confidence = self.recognize_with_postprocessing(text, confidence)
                return text, confidence

        except Exception as e:
            print(f"OCR识别错误: {e}")

        return "", 0.0

    def recognize_batch(self, frame, bboxes):
        """批量识别多个文本区域（提高效率）"""
        results = []
        for bbox in bboxes:
            text, confidence = self.recognize(frame, bbox)
            if text:
                results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence
                })
        return results