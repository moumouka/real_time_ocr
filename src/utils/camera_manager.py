"""
摄像头管理器 - 手动控制摄像头
"""
import cv2
import threading
import time
import numpy as np
from typing import Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class CameraManager:
    """摄像头管理器 - 手动控制"""

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # 摄像头参数
        self.camera_id = 0
        self.resolution = (640, 480)
        self.fps = 30

        # 性能监控
        self.capture_fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

        # 回调函数
        self.on_frame_callback = None

        # 线程
        self.capture_thread = None

    def start(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480)):
        """启动摄像头"""
        if self.is_running:
            logger.warning("摄像头已经在运行")
            return False

        try:
            self.camera_id = camera_id
            self.resolution = resolution

            # 打开摄像头
            self.cap = cv2.VideoCapture(camera_id)

            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {camera_id}")

            # 设置参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # 启动捕获线程
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()

            logger.info(f"摄像头 {camera_id} 启动成功，分辨率: {resolution[0]}x{resolution[1]}")
            return True

        except Exception as e:
            logger.error(f"启动摄像头失败: {e}")
            return False

    def stop(self):
        """停止摄像头"""
        self.is_running = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        logger.info("摄像头已停止")

    def _capture_loop(self):
        """捕获循环"""
        logger.info("开始摄像头捕获循环")

        while self.is_running and self.cap and self.cap.isOpened():
            try:
                # 捕获帧
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("无法从摄像头读取帧")
                    time.sleep(0.01)
                    continue

                # 更新当前帧
                with self.frame_lock:
                    self.current_frame = frame.copy()

                # 更新FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.capture_fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time

                # 回调通知
                if self.on_frame_callback:
                    self.on_frame_callback(frame.copy())

                # 控制捕获速率
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"摄像头捕获出错: {e}")
                time.sleep(0.1)

        logger.info("摄像头捕获循环结束")

    def get_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def capture(self) -> Optional[np.ndarray]:
        """捕获一帧"""
        return self.get_frame()

    def set_callback(self, callback: Callable):
        """设置帧回调函数"""
        self.on_frame_callback = callback

    def get_info(self) -> dict:
        """获取摄像头信息"""
        info = {
            'is_running': self.is_running,
            'camera_id': self.camera_id,
            'resolution': self.resolution,
            'fps': self.capture_fps,
            'frame_size': None
        }

        if self.current_frame is not None:
            info['frame_size'] = self.current_frame.shape[:2]

        return info

    def change_resolution(self, width: int, height: int):
        """更改分辨率"""
        if self.is_running and self.cap:
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.resolution = (width, height)
                logger.info(f"分辨率已更改为: {width}x{height}")
                return True
            except Exception as e:
                logger.error(f"更改分辨率失败: {e}")
                return False
        return False