"""
性能监控工具模块
"""
import time
from typing import Dict, Any
from collections import deque


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.fps_history = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)

    def start_frame(self):
        """开始处理一帧"""
        self.timestamps.append(time.time())

    def end_frame(self):
        """结束处理一帧"""
        if len(self.timestamps) > 1:
            processing_time = time.time() - self.timestamps[-1]
            self.processing_times.append(processing_time)

            # 计算FPS
            if len(self.timestamps) >= 2:
                time_diff = self.timestamps[-1] - self.timestamps[0]
                if time_diff > 0:
                    fps = len(self.timestamps) / time_diff
                    self.fps_history.append(fps)

    def get_fps(self) -> float:
        """获取当前FPS"""
        if len(self.fps_history) > 0:
            return self.fps_history[-1]
        return 0.0

    def get_avg_fps(self) -> float:
        """获取平均FPS"""
        if len(self.fps_history) > 0:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0

    def get_avg_processing_time(self) -> float:
        """获取平均处理时间"""
        if len(self.processing_times) > 0:
            return sum(self.processing_times) / len(self.processing_times)
        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取所有统计信息"""
        return {
            'current_fps': self.get_fps(),
            'avg_fps': self.get_avg_fps(),
            'avg_processing_time': self.get_avg_processing_time(),
            'frames_processed': len(self.timestamps),
        }

    def reset(self):
        """重置监控器"""
        self.timestamps.clear()
        self.fps_history.clear()
        self.processing_times.clear()
        