"""
显示适配器 - 处理摄像头显示的自适应
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DisplayAdapter:
    """显示适配器 - 管理摄像头的自适应显示"""

    def __init__(self, target_size: Tuple[int, int] = (800, 600)):
        self.target_size = target_size
        self.current_size = target_size
        self.maintain_aspect_ratio = True
        self.scale_factor = 1.0
        self.border_color = (50, 50, 50)  # 边框颜色

        # 性能跟踪
        self.fps = 0
        self.frame_count = 0
        self.last_time = 0

    def set_target_size(self, width: int, height: int):
        """设置目标显示尺寸"""
        self.target_size = (width, height)
        logger.info(f"目标显示尺寸设置为: {width}x{height}")

    def adapt_frame(self, frame: np.ndarray, add_border: bool = True) -> np.ndarray:
        """自适应调整帧到目标尺寸"""
        if frame is None or frame.size == 0:
            return frame

        # 更新FPS
        self._update_fps()

        # 获取原始尺寸
        original_height, original_width = frame.shape[:2]
        target_width, target_height = self.target_size

        # 计算缩放比例
        width_scale = target_width / original_width
        height_scale = target_height / original_height

        if self.maintain_aspect_ratio:
            # 保持宽高比，计算最大内接矩形
            scale = min(width_scale, height_scale)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            self.scale_factor = scale

            # 调整大小
            resized = cv2.resize(frame, (new_width, new_height))

            if add_border and (new_width < target_width or new_height < target_height):
                # 添加边框以填充剩余空间
                border_top = (target_height - new_height) // 2
                border_bottom = target_height - new_height - border_top
                border_left = (target_width - new_width) // 2
                border_right = target_width - new_width - border_left

                # 添加边框
                bordered = cv2.copyMakeBorder(
                    resized,
                    border_top, border_bottom,
                    border_left, border_right,
                    cv2.BORDER_CONSTANT,
                    value=self.border_color
                )

                # 在边框上添加比例信息
                if border_top > 20:
                    info_text = f"缩放: {scale:.2f}x"
                    cv2.putText(bordered, info_text,
                                (10, border_top // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (200, 200, 200), 1)

                result = bordered
            else:
                result = resized

            self.current_size = (new_width, new_height)

        else:
            # 拉伸到目标尺寸
            resized = cv2.resize(frame, (target_width, target_height))
            self.scale_factor = max(width_scale, height_scale)
            self.current_size = (target_width, target_height)
            result = resized

        # 添加FPS显示
        result = self._add_fps_overlay(result)

        return result

    def scale_coordinates(self, x: int, y: int,
                          original_size: Tuple[int, int]) -> Tuple[int, int]:
        """缩放坐标到显示尺寸"""
        orig_width, orig_height = original_size
        display_width, display_height = self.current_size

        # 计算实际显示区域的位置（考虑边框）
        if self.maintain_aspect_ratio:
            scale_x = display_width / orig_width
            scale_y = display_height / orig_height
            scale = min(scale_x, scale_y)

            # 计算边框偏移
            border_x = (self.target_size[0] - display_width) // 2
            border_y = (self.target_size[1] - display_height) // 2

            scaled_x = int(x * scale) + border_x
            scaled_y = int(y * scale) + border_y
        else:
            scaled_x = int(x * self.scale_factor)
            scaled_y = int(y * self.scale_factor)

        return scaled_x, scaled_y

    def _update_fps(self):
        """更新FPS计算"""
        import time
        current_time = time.time()

        if self.last_time == 0:
            self.last_time = current_time

        self.frame_count += 1

        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time

    def _add_fps_overlay(self, frame: np.ndarray) -> np.ndarray:
        """添加FPS覆盖层"""
        overlay = frame.copy()

        # 在左上角显示FPS
        cv2.putText(overlay, f"FPS: {self.fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        # 显示分辨率信息
        height, width = frame.shape[:2]
        cv2.putText(overlay, f"{width}x{height}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 200), 1)

        return overlay

    def get_display_info(self) -> dict:
        """获取显示信息"""
        return {
            'target_size': self.target_size,
            'current_size': self.current_size,
            'scale_factor': self.scale_factor,
            'fps': self.fps,
            'maintain_aspect_ratio': self.maintain_aspect_ratio
        }