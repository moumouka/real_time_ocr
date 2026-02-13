"""
测试实时识别功能
"""
import cv2
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ocr_core import OCRCore
from core.config import SystemConfig


def test_realtime_recognition():
    """测试实时识别功能"""
    print("=" * 60)
    print("实时识别功能测试")
    print("=" * 60)

    # 创建配置
    config = SystemConfig(
        use_yolo_first=True,
        confidence_threshold=0.3,
        use_gpu=False
    )

    # 创建OCR核心
    ocr_core = OCRCore(config=config)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("摄像头已打开")
    print("按 'q' 退出")
    print("按 'r' 开启/关闭实时识别")
    print("按 'c' 单次识别")
    print("按 '1'/'2'/'3' 调整识别间隔")
    print("-" * 60)

    window_name = "实时识别测试"
    cv2.namedWindow(window_name)

    # 状态变量
    enable_realtime = True
    recognition_interval = 0.5
    last_recognition_time = 0
    recognition_fps = 0
    recognition_count = 0
    last_fps_time = time.time()
    frame_count = 0
    camera_fps = 0

    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break

            # 更新相机FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                camera_fps = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time

            display_frame = frame.copy()

            # 实时识别
            if enable_realtime:
                if current_time - last_recognition_time >= recognition_interval:
                    # 进行OCR识别
                    results = ocr_core.process_image(frame)

                    # 更新识别FPS
                    recognition_count += 1
                    if current_time - last_recognition_time >= 1.0:
                        recognition_fps = recognition_count
                        recognition_count = 0

                    # 绘制结果
                    if results:
                        display_frame = ocr_core.draw_results(display_frame, results)
                        print(f"\r[{time.strftime('%H:%M:%S')}] 识别到 {len(results)} 个文本", end="")

                    last_recognition_time = current_time

            # 添加信息显示
            cv2.putText(display_frame, f"相机FPS: {camera_fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"识别FPS: {recognition_fps:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示实时识别状态
            status_color = (0, 255, 0) if enable_realtime else (0, 0, 255)
            cv2.putText(display_frame, f"实时识别: {'开启' if enable_realtime else '关闭'}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, f"间隔: {recognition_interval:.1f}s",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

            # 显示
            cv2.imshow(window_name, display_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                enable_realtime = not enable_realtime
                print(f"\n实时识别: {'开启' if enable_realtime else '关闭'}")
            elif key == ord('c'):
                # 单次识别
                results = ocr_core.process_image(frame)
                display_frame = ocr_core.draw_results(frame.copy(), results)
                cv2.imshow(window_name, display_frame)
                print(f"\n单次识别完成，找到 {len(results)} 个文本")
            elif key == ord('1'):
                recognition_interval = 0.2
                print(f"\n识别间隔设置为: 0.2s")
            elif key == ord('2'):
                recognition_interval = 0.5
                print(f"\n识别间隔设置为: 0.5s")
            elif key == ord('3'):
                recognition_interval = 1.0
                print(f"\n识别间隔设置为: 1.0s")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n测试结束")


if __name__ == "__main__":
    test_realtime_recognition()
