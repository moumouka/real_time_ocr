"""
可视化工具模块
"""
import cv2
import numpy as np
from typing import List, Tuple

from core.recognizers import OCRResult


def draw_ocr_results(image: np.ndarray, results: List[OCRResult],
                     config) -> np.ndarray:
    """绘制OCR结果到图像"""
    display_frame = image.copy()

    for result in results:
        box = result.bbox
        text = result.text
        source = result.detection_source
        conf = result.confidence

        # 根据检测方法选择颜色
        if "YOLO" in source:
            color = config.box_color_yolo
        else:
            color = config.box_color_traditional

        # 绘制边界框
        cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]),
                      color, config.box_thickness)

        # 准备标签文本
        label_parts = []
        if text:
            label_parts.append(text[:20])  # 最多显示20个字符

        if config.show_confidence:
            label_parts.append(f"{conf:.2f}")

        if config.show_detection_source:
            label_parts.append(source)

        label = " | ".join(label_parts)

        if label:
            # 计算文本大小
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, config.text_size,
                config.text_thickness
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
                          config.text_background, -1)

            # 绘制文本
            cv2.putText(display_frame, label,
                        (label_x, label_y - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, config.text_size,
                        config.text_color, config.text_thickness)

    return display_frame


def add_fps_display(frame: np.ndarray, fps: float) -> np.ndarray:
    """添加FPS显示"""
    display_frame = frame.copy()
    cv2.putText(display_frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)
    return display_frame


def resize_image(image: np.ndarray, max_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
    """调整图像大小"""
    h, w = image.shape[:2]
    max_w, max_h = max_size

    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))

    return image
