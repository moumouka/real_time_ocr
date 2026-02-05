"""
显示管理器
"""

import cv2
import os
import time
from src.config import Config


class DisplayManager:
    def __init__(self, window_name=Config.WINDOW_NAME):
        self.window_name = window_name
        self.show_fps = Config.SHOW_FPS
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name,
                         Config.WINDOW_WIDTH,
                         Config.WINDOW_HEIGHT)

        # 创建截图保存目录
        if not os.path.exists(Config.SAVE_DIRECTORY):
            os.makedirs(Config.SAVE_DIRECTORY)

    def draw_results(self, frame, results, camera_fps=0):
        """在帧上绘制结果"""
        display_frame = frame.copy()

        # 绘制每个检测结果
        for i, result in enumerate(results):
            x1, y1, x2, y2 = result['bbox']
            text = result['text']
            conf = result.get('detection_confidence', 0.5)

            # 选择颜色（根据置信度）
            if conf > 0.7:
                color = (0, 255, 0)  # 绿色 - 高置信度
            elif conf > 0.5:
                color = (0, 255, 255)  # 黄色 - 中置信度
            else:
                color = (0, 165, 255)  # 橙色 - 低置信度

            # 绘制边界框
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # 计算文本位置
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # 文本背景
            text_display = f"{text}"
            text_size = cv2.getTextSize(text_display, font, font_scale, thickness)[0]

            # 确保文本背景在图像范围内
            bg_y1 = max(0, y1 - text_size[1] - 10)
            bg_y2 = max(0, y1)

            cv2.rectangle(
                display_frame,
                (x1, bg_y1),
                (x1 + text_size[0], bg_y2),
                color,
                -1
            )

            # 绘制文本
            cv2.putText(
                display_frame,
                text_display,
                (x1, y1 - 5 if y1 - 5 > 0 else y1 + 15),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

        # 显示FPS
        if self.show_fps:
            # 更新FPS计算
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time

            # 显示摄像头FPS和处理FPS
            cv2.putText(
                display_frame,
                f"摄像头FPS: {camera_fps}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"显示FPS: {self.fps}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # 显示检测到的文本数量
        cv2.putText(
            display_frame,
            f"检测到 {len(results)} 个文本",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # 显示检测间隔
        cv2.putText(
            display_frame,
            f"检测间隔: 每{Config.DETECTION_INTERVAL}帧",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        return display_frame

    def show_frame(self, frame):
        """显示帧"""
        cv2.imshow(self.window_name, frame)

    def save_screenshot(self, frame):
        """保存截图"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(Config.SAVE_DIRECTORY, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"已保存截图: {filename}")
        return filename

    def show_help(self):
        """显示帮助信息"""
        print("\n" + "=" * 50)
        print("实时OCR识别系统 - 快捷键说明")
        print("=" * 50)
        print("q - 退出程序")
        print("s - 保存当前帧截图")
        print("d - 切换检测间隔 (5/10/30帧)")
        print("h - 显示帮助信息")
        print("=" * 50 + "\n")

    def destroy_windows(self):
        """销毁所有窗口"""
        cv2.destroyAllWindows()
