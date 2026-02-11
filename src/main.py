# 更新 main.py 或创建专门的实时测试脚本
"""
实时OCR测试脚本
"""
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ocr_core import OCRCore
from core.config import SystemConfig
from utils.display_adapter import DisplayAdapter


def test_adaptive_display():
    """测试自适应显示功能"""
    print("测试自适应摄像头显示...")
    
    # 创建配置
    config = SystemConfig(
        use_yolo_first=True,
        confidence_threshold=0.4,
        use_gpu=False,
        realtime_processing=True,
        process_interval=0.033  # ~30 FPS
    )
    
    # 创建OCR核心
    ocr_core = OCRCore(config=config)
    
    # 创建显示适配器
    display_adapter = DisplayAdapter(target_size=(800, 600))
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("摄像头已打开，按 'q' 退出，按 'f' 全屏切换")
    
    window_name = "自适应摄像头显示"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    is_fullscreen = False
    frame_counter = 0
    
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break
            
            # 自适应调整显示
            display_frame = display_adapter.adapt_frame(frame, add_border=True)
            
            # 每30帧处理一次OCR（保持性能）
            results = []
            if frame_counter % 30 == 0:
                boxes, detection_method = ocr_core.detect_text(frame)
                if boxes:
                    results = ocr_core.recognize_text(frame, boxes, detection_method)
                    
                    # 在显示帧上绘制结果（需要坐标转换）
                    if results:
                        for result in results:
                            box = result.bbox
                            # 这里需要将原始坐标转换到显示坐标
                            # 简化处理：直接在原始帧上处理
                            pass
            
            # 显示帧
            cv2.imshow(window_name, display_frame)
            
            # 获取显示信息
            info = display_adapter.get_display_info()
            
            # 在控制台显示信息
            if frame_counter % 30 == 0:
                print(f"\rFPS: {info['fps']:.1f}, 显示尺寸: {info['current_size'][0]}x{info['current_size'][1]}, 缩放: {info['scale_factor']:.2f}x", end="")
            
            frame_counter += 1
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            elif key == ord('r'):
                # 重置显示尺寸
                display_adapter.set_target_size(800, 600)
            elif key == ord('1'):
                # 切换到小尺寸
                display_adapter.set_target_size(640, 480)
            elif key == ord('2'):
                # 切换到中尺寸
                display_adapter.set_target_size(800, 600)
            elif key == ord('3'):
                # 切换到大尺寸
                display_adapter.set_target_size(1024, 768)
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n程序结束")


if __name__ == "__main__":
    test_adaptive_display()