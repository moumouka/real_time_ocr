"""
摄像头管理器
"""

import cv2
import time
import threading
from queue import Queue
from src.config import Config


class CameraManager:
    def __init__(self, camera_id=Config.CAMERA_ID):
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.frame_queue = Queue(maxsize=1)  # 只保留最新帧
        self.latest_frame = None
        self.lock = threading.Lock()
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

    def initialize(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)

        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)

        # 验证设置
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"摄像头初始化完成")
        print(f"分辨率: {actual_width}x{actual_height}")
        print(f"FPS: {actual_fps}")

        return self.cap.isOpened()

    def start_capture(self):
        """开始捕获视频流"""
        if not self.cap or not self.cap.isOpened():
            if not self.initialize():
                raise RuntimeError(f"无法打开摄像头 {self.camera_id}")

        self.running = True
        print("开始捕获视频流...")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            # 计算FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time

            # 更新最新帧
            with self.lock:
                self.latest_frame = frame.copy()

            # 放入队列（如果有消费者）
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())

    def get_frame(self):
        """获取最新帧"""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def get_frame_from_queue(self, timeout=1.0):
        """从队列获取帧"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None

    def stop(self):
        """停止摄像头"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("摄像头已停止")

    def get_resolution(self):
        """获取摄像头分辨率"""
        if self.cap:
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            return int(width), int(height)
        return Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT

    def get_fps(self):
        """获取当前FPS"""
        return self.fps