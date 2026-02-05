"""
工具函数
"""

import torch
import sys
import cv2


def check_gpu_availability():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU:")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("未检测到GPU，将使用CPU")
        return False


def get_available_cameras(max_check=10):
    """检测可用的摄像头"""
    available_cameras = []

    print("扫描可用摄像头...")
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  摄像头 {i}: {width}x{height} @ {fps:.1f} FPS")
            available_cameras.append(i)
            cap.release()
        else:
            cap.release()
            break

    return available_cameras


def print_system_info():
    """打印系统信息"""
    import platform

    print("=" * 50)
    print("系统信息")
    print("=" * 50)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"OpenCV版本: {cv2.__version__}")

    try:
        import ultralytics
        print(f"Ultralytics版本: {ultralytics.__version__}")
    except:
        pass

    try:
        import easyocr
        print(f"EasyOCR版本: {easyocr.__version__}")
    except:
        pass

    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
    except:
        pass

    print("=" * 50)
