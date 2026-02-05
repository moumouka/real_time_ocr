"""
配置文件
"""


class Config:
    # 摄像头配置
    CAMERA_ID = 0  # 默认摄像头
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30

    # YOLO配置
    YOLO_MODEL = 'model/yolo11n.pt'
    DETECTION_CONFIDENCE = 0.3
    DETECTION_INTERVAL = 10  # 每10帧检测一次

    # OCR配置
    OCR_LANGUAGES = ['en', 'ch_sim']
    OCR_CONFIDENCE = 0.3
    OCR_PADDING = 5  # 文本区域padding

    # 显示配置
    WINDOW_NAME = "实时OCR识别系统"
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    SHOW_FPS = True

    # 预处理配置
    TARGET_WIDTH = 640  # 预处理后的目标宽度

    # 键盘快捷键
    KEY_QUIT = ord('q')
    KEY_SAVE = ord('s')
    KEY_TOGGLE_DETECTION = ord('d')

    # 文件保存
    SAVE_DIRECTORY = "screenshots/"
