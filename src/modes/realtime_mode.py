"""
实时模式处理模块 - 修复版
"""
import cv2
import time
import threading
import queue
from typing import Optional, Callable, List
import logging
import numpy as np

from core.ocr_core import OCRCore
from core.recognizers import OCRResult

logger = logging.getLogger(__name__)


class RealTimeProcessor:
    """实时处理器 - 修复版"""

    def __init__(self, ocr_core: OCRCore):
        self.ocr_core = ocr_core
        self.config = ocr_core.config

        # 摄像头相关
        self.cap = None
        self.is_running = False
        self.current_frame = None

        # 多线程处理
        self.capture_thread = None
        self.processing_thread = None

        # 队列用于帧传递
        self.frame_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.result_queue = queue.Queue(maxsize=self.config.max_queue_size)

        # 性能监控
        self.fps = 0
        self.processing_fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.processed_frame_count = 0

        # 回调函数
        self.on_frame_callback = None
        self.on_result_callback = None

        # 同步锁
        self.frame_lock = threading.Lock()
        self.running_lock = threading.Lock()

        # 控制变量
        self.last_process_time = 0
        self.processing_active = True

    def start_camera(self, camera_id: int = None) -> bool:
        """启动摄像头 - 使用多线程架构"""
        if camera_id is None:
            camera_id = self.config.camera_id

        try:
            # 初始化摄像头
            self.cap = cv2.VideoCapture(camera_id)

            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {camera_id}")

            # 启动标志
            self.is_running = True
            self.processing_active = True

            # 启动捕获线程
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()

            # 启动处理线程
            self.processing_thread = threading.Thread(target=self._process_frames)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            # 启动显示线程（可选）
            self.display_thread = threading.Thread(target=self._display_loop)
            self.display_thread.daemon = True
            self.display_thread.start()

            logger.info(f"摄像头 {camera_id} 启动成功")
            return True

        except Exception as e:
            logger.error(f"启动摄像头失败: {e}")
            return False

    def _capture_frames(self):
        """捕获帧的线程函数 - 独立运行"""
        logger.info("开始捕获视频帧...")

        while self.is_running and self.cap and self.cap.isOpened():
            try:
                # 捕获帧
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("无法从摄像头读取帧")
                    time.sleep(0.01)
                    continue

                # 更新帧计数和FPS计算
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_time = current_time

                # 将帧放入队列（非阻塞）
                if self.frame_queue.qsize() < self.config.max_queue_size:
                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        pass  # 队列已满，丢弃最旧的帧
                else:
                    # 队列满时，替换最新的帧
                    try:
                        self.frame_queue.get(block=False)  # 移除最旧的
                        self.frame_queue.put(frame, block=False)
                    except:
                        pass

                # 稍微休息一下，避免占用太多CPU
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"捕获帧时出错: {e}")
                time.sleep(0.1)

        logger.info("帧捕获线程结束")

    def _process_frames(self):
        """处理帧的线程函数"""
        logger.info("开始处理视频帧...")

        while self.is_running and self.processing_active:
            try:
                # 从队列获取帧（带超时）
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    time.sleep(0.01)
                    continue

                # 检查处理间隔
                current_time = time.time()
                if current_time - self.last_process_time < self.config.process_interval:
                    # 还没到处理时间，跳过此帧
                    time.sleep(0.001)
                    continue

                # 更新处理时间
                self.last_process_time = current_time

                # OCR处理
                results = []
                if self.config.realtime_processing:
                    # 文本检测
                    boxes, detection_method = self.ocr_core.detect_text(frame)

                    # 文本识别
                    if boxes:
                        results = self.ocr_core.recognize_text(frame, boxes, detection_method)

                # 在帧上绘制结果
                display_frame = self.ocr_core.draw_results(frame.copy(), results)

                # 添加FPS显示
                if self.config.show_fps:
                    cv2.putText(display_frame, f"Capture FPS: {self.fps}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (0, 255, 0), 2)

                    cv2.putText(display_frame, f"Processed: {self.processed_frame_count}",
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (0, 255, 255), 2)

                # 保存当前帧
                with self.frame_lock:
                    self.current_frame = display_frame

                # 更新处理统计
                self.processed_frame_count += 1

                # 将结果放入队列
                if self.result_queue.qsize() < self.config.max_queue_size:
                    try:
                        self.result_queue.put({
                            'frame': display_frame,
                            'results': results,
                            'timestamp': current_time
                        }, block=False)
                    except queue.Full:
                        pass

            except Exception as e:
                logger.error(f"处理帧时出错: {e}")
                time.sleep(0.01)

        logger.info("帧处理线程结束")

    def _display_loop(self):
        """显示循环线程 - 持续更新GUI"""
        logger.info("开始显示循环...")

        while self.is_running:
            try:
                # 从结果队列获取最新结果
                latest_result = None
                while not self.result_queue.empty():
                    latest_result = self.result_queue.get()

                if latest_result:
                    # 回调通知
                    if self.on_frame_callback:
                        self.on_frame_callback(latest_result['frame'])

                    if self.on_result_callback and latest_result['results']:
                        self.on_result_callback(latest_result['results'])

                # 休眠以避免占用太多CPU
                time.sleep(0.03)  # ~30 FPS的GUI更新

            except Exception as e:
                logger.error(f"显示循环出错: {e}")
                time.sleep(0.1)

        logger.info("显示循环线程结束")

    def stop_camera(self):
        """停止摄像头"""
        logger.info("正在停止摄像头...")

        self.is_running = False
        self.processing_active = False

        # 等待线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)

        # 释放摄像头
        if self.cap:
            self.cap.release()
            self.cap = None

        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get(block=False)
            except:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get(block=False)
            except:
                break

        logger.info("摄像头已完全停止")

    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        with self.frame_lock:
            return self.current_frame

    def set_callback(self, frame_callback: Callable = None,
                    result_callback: Callable = None):
        """设置回调函数"""
        self.on_frame_callback = frame_callback
        self.on_result_callback = result_callback

    def get_fps(self) -> float:
        """获取捕获FPS"""
        return self.fps

    def get_processing_fps(self) -> float:
        """获取处理FPS"""
        if self.processed_frame_count > 0:
            elapsed = time.time() - self.last_process_time
            if elapsed > 0:
                return self.processed_frame_count / elapsed
        return 0.0

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'capture_fps': self.fps,
            'processing_fps': self.get_processing_fps(),
            'frames_captured': self.frame_count,
            'frames_processed': self.processed_frame_count,
            'queue_size': self.frame_queue.qsize()
        }
