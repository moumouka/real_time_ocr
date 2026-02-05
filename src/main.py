"""
主程序入口
"""

import cv2
import threading
import time
from config import Config
from camera.camera_manager import CameraManager
from detector.text_detector import TextDetector
from detector.text_recognizer import TextRecognizer
from processor.frame_processor import FrameProcessor
from ui.display_manager import DisplayManager
from utils.helpers import print_system_info, check_gpu_availability


class RealTimeOCRSystem:
    def __init__(self, camera_id=Config.CAMERA_ID):
        # 打印系统信息
        print_system_info()
        check_gpu_availability()

        # 初始化各个组件
        print("\n初始化系统组件...")

        # 摄像头
        self.camera = CameraManager(camera_id)

        # 检测器
        self.text_detector = TextDetector()
        self.text_recognizer = TextRecognizer()

        # 处理器
        self.processor = FrameProcessor(self.text_detector, self.text_recognizer)

        # 显示
        self.display = DisplayManager()

        # 控制变量
        self.running = False
        self.detection_intervals = [5, 10, 30]  # 可用的检测间隔
        self.current_interval_index = 1  # 默认使用10帧间隔

    def run(self):
        """运行主循环"""
        # 初始化摄像头
        if not self.camera.initialize():
            print(f"错误: 无法打开摄像头 {Config.CAMERA_ID}")
            return

        # 显示帮助信息
        self.display.show_help()

        # 启动摄像头捕获线程
        self.running = True
        camera_thread = threading.Thread(target=self.camera.start_capture, daemon=True)
        camera_thread.start()

        print("\n开始实时OCR识别...")
        print("按 'h' 键查看帮助")

        # 主循环
        while self.running:
            try:
                # 获取最新帧
                frame = self.camera.get_frame_from_queue(timeout=0.1)
                if frame is None:
                    continue

                # 处理帧
                processed_frame = self.processor.process(frame)

                # 获取检测结果
                results = self.processor.get_results()

                # 获取摄像头FPS
                camera_fps = self.camera.get_fps()

                # 绘制结果显示
                display_frame = self.display.draw_results(frame, results, camera_fps)

                # 显示帧
                self.display.show_frame(display_frame)

                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                self.handle_keyboard(key, frame)

                # 打印检测结果（如果有）
                if results:
                    print(f"检测到文本: {[r['text'] for r in results]}")

            except KeyboardInterrupt:
                print("\n接收到中断信号，正在退出...")
                break
            except Exception as e:
                print(f"主循环错误: {e}")
                continue

        # 清理
        self.cleanup()

    def handle_keyboard(self, key, frame):
        """处理键盘输入"""
        if key == Config.KEY_QUIT:
            print("退出程序...")
            self.running = False
        elif key == Config.KEY_SAVE:
            self.display.save_screenshot(frame)
        elif key == Config.KEY_TOGGLE_DETECTION:
            # 切换检测间隔
            self.current_interval_index = (self.current_interval_index + 1) % len(self.detection_intervals)
            new_interval = self.detection_intervals[self.current_interval_index]
            self.processor.set_detection_interval(new_interval)
        elif key == ord('h'):
            self.display.show_help()

    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")
        self.running = False
        self.camera.stop()
        self.display.destroy_windows()
        print("程序已退出")


def main():
    """主函数"""
    try:
        # 创建系统实例
        # 可选: 从命令行参数获取摄像头ID
        import sys
        if len(sys.argv) > 1:
            camera_id = int(sys.argv[1])
        else:
            camera_id = Config.CAMERA_ID

        # 创建并运行系统
        ocr_system = RealTimeOCRSystem(camera_id=camera_id)
        ocr_system.run()

    except Exception as e:
        print(f"程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
