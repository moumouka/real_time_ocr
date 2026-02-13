"""
图形用户界面模块 - 完整版
支持手动摄像头控制 + 实时识别
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import logging
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import queue
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ocr_core import OCRCore
from core.config import ConfigManager, SystemConfig
from modes.image_mode import ImageProcessor

logger = logging.getLogger(__name__)


class OCRGUI:
    """OCR图形用户界面 - 支持手动摄像头控制和实时识别"""

    def __init__(self, config_path: str = None):
        self.root = tk.Tk()
        self.root.title("智能OCR系统 v4.0 - 手动摄像头+实时识别")
        self.root.geometry("1200x700")

        # 初始化OCR核心
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.ocr_core = OCRCore(config=self.config)

        # 摄像头相关变量
        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # 实时识别相关变量
        self.enable_realtime_recognition = False
        self.last_recognition_time = 0
        self.recognition_interval = 0.5
        self.last_recognition_results = []
        self.recognition_queue = queue.Queue(maxsize=5)

        # 显示相关变量
        self.display_scale = 1.0
        self.original_frame_size = (0, 0)
        self.last_frame_time = 0
        self.frame_update_interval = 0.033
        self.target_display_size = (800, 600)

        # 性能监控
        self.camera_fps = 0
        self.recognition_fps = 0
        self.frame_count = 0
        self.recognition_count = 0
        self.last_fps_time = time.time()
        self.last_recognition_fps_time = time.time()

        # 当前状态
        self.current_image = None
        self.current_image_path = None
        self.current_results = []
        self.current_display_image = None

        # 创建界面
        self._create_widgets()

        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 启动GUI更新定时器
        self._start_gui_update_timer()

        # 启动识别结果处理线程
        self._start_recognition_processor()

    def _create_widgets(self):
        """创建界面组件"""
        # 顶部菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开图片", command=self.open_image)
        file_menu.add_command(label="批量处理", command=self.batch_process)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)

        # 摄像头菜单
        camera_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="摄像头", menu=camera_menu)
        camera_menu.add_command(label="启动摄像头", command=self.start_camera)
        camera_menu.add_command(label="停止摄像头", command=self.stop_camera)
        camera_menu.add_separator()
        camera_menu.add_command(label="拍照识别", command=self.capture_and_recognize)
        camera_menu.add_command(label="实时识别", command=self.toggle_realtime_recognition)

        # 显示菜单
        display_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="显示", menu=display_menu)
        display_menu.add_command(label="适合窗口", command=self._fit_to_window)
        display_menu.add_command(label="原始大小", command=self._original_size)
        display_menu.add_separator()
        display_menu.add_command(label="全屏显示", command=self.toggle_fullscreen)

        # 主框架 - 左右布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧 - 图像显示区
        left_frame = ttk.Frame(main_frame, width=800)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 图像显示画布
        self.canvas = tk.Canvas(left_frame, bg='gray', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 绑定画布大小变化事件
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        # 摄像头控制面板
        camera_control_frame = ttk.LabelFrame(left_frame, text="摄像头控制", padding=5)
        camera_control_frame.pack(fill=tk.X, pady=(0, 5))

        # 第一行：摄像头选择和基础控制
        control_row1 = ttk.Frame(camera_control_frame)
        control_row1.pack(fill=tk.X, pady=2)

        ttk.Label(control_row1, text="摄像头:").pack(side=tk.LEFT, padx=5)
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(control_row1, textvariable=self.camera_var,
                                    values=["0", "1", "2", "3"], width=8)
        camera_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_row1, text="启动",
                   command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_row1, text="停止",
                   command=self.stop_camera).pack(side=tk.LEFT, padx=5)

        # 摄像头状态显示
        self.camera_status = ttk.Label(control_row1, text="未连接")
        self.camera_status.pack(side=tk.RIGHT, padx=10)

        # 第二行：分辨率和高级控制
        control_row2 = ttk.Frame(camera_control_frame)
        control_row2.pack(fill=tk.X, pady=2)

        ttk.Label(control_row2, text="分辨率:").pack(side=tk.LEFT, padx=5)
        self.resolution_var = tk.StringVar(value="640x480")
        resolution_combo = ttk.Combobox(control_row2, textvariable=self.resolution_var,
                                        values=["320x240", "640x480", "800x600",
                                                "1024x768", "1280x720", "1920x1080"],
                                        width=12)
        resolution_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_row2, text="拍照",
                   command=self.capture_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_row2, text="单次识别",
                   command=self.recognize_camera_frame).pack(side=tk.LEFT, padx=5)

        # 实时识别开关
        self.realtime_recognition_var = tk.BooleanVar(value=False)
        self.realtime_recognition_check = ttk.Checkbutton(
            control_row2,
            text="实时识别",
            variable=self.realtime_recognition_var,
            command=self.toggle_realtime_recognition
        )
        self.realtime_recognition_check.pack(side=tk.LEFT, padx=10)

        # 性能显示
        self.performance_label = ttk.Label(control_row2, text="相机FPS: 0.0 | 识别FPS: 0.0")
        self.performance_label.pack(side=tk.RIGHT, padx=10)

        # 图像控制面板
        image_control_frame = ttk.Frame(left_frame)
        image_control_frame.pack(fill=tk.X, pady=5)

        # 缩放控制
        scale_frame = ttk.Frame(image_control_frame)
        scale_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(scale_frame, text="缩放:").pack(side=tk.LEFT)

        self.scale_var = tk.DoubleVar(value=100)
        scale_slider = ttk.Scale(scale_frame, from_=25, to=200,
                                 orient=tk.HORIZONTAL, variable=self.scale_var,
                                 command=self._on_scale_changed, length=150)
        scale_slider.pack(side=tk.LEFT, padx=5)

        self.scale_label = ttk.Label(scale_frame, text="100%")
        self.scale_label.pack(side=tk.LEFT)

        ttk.Button(scale_frame, text="适合窗口",
                   command=self._fit_to_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(scale_frame, text="原始大小",
                   command=self._original_size).pack(side=tk.LEFT, padx=5)

        # 显示模式切换
        ttk.Button(image_control_frame, text="摄像头模式",
                   command=self.switch_to_camera_mode).pack(side=tk.LEFT, padx=10)
        ttk.Button(image_control_frame, text="图片模式",
                   command=self.switch_to_image_mode).pack(side=tk.LEFT, padx=5)

        # 右侧 - 控制面板
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # OCR控制面板
        ocr_control_frame = ttk.LabelFrame(right_frame, text="OCR控制", padding=10)
        ocr_control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(ocr_control_frame, text="打开图片", 
                  command=self.open_image).pack(fill=tk.X, pady=2)
        ttk.Button(ocr_control_frame, text="识别当前画面", 
                  command=self.recognize_current_frame).pack(fill=tk.X, pady=2)
        ttk.Button(ocr_control_frame, text="保存结果", 
                  command=self.save_results).pack(fill=tk.X, pady=2)

        # 在实时识别设置部分添加置信度阈值显示
        realtime_settings_frame = ttk.LabelFrame(right_frame, text="实时识别设置", padding=10)
        realtime_settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(realtime_settings_frame, text="识别间隔:").pack(anchor=tk.W)
        self.interval_var = tk.StringVar(value="0.5")
        interval_combo = ttk.Combobox(realtime_settings_frame, textvariable=self.interval_var,
                                      values=["0.1", "0.2", "0.3", "0.5", "1.0", "2.0"],
                                      width=10)
        interval_combo.pack(fill=tk.X, pady=2)

        # 置信度阈值设置
        ttk.Label(realtime_settings_frame, text="置信度阈值:").pack(anchor=tk.W, pady=(5, 0))
        self.confidence_var = tk.DoubleVar(value=0.4)
        confidence_scale = ttk.Scale(realtime_settings_frame, from_=0.1, to=0.9,
                                     variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.pack(fill=tk.X, pady=2)

        # 显示当前置信度阈值
        confidence_frame = ttk.Frame(realtime_settings_frame)
        confidence_frame.pack(fill=tk.X, pady=2)

        ttk.Label(confidence_frame, text="当前阈值:").pack(side=tk.LEFT)
        self.confidence_label = ttk.Label(confidence_frame, text="0.40")
        self.confidence_label.pack(side=tk.LEFT, padx=5)

        # 添加置信度阈值变化的回调
        def on_confidence_changed(*args):
            value = self.confidence_var.get()
            self.confidence_label.config(text=f"{value:.2f}")
            # 更新配置
            self.config.confidence_threshold = value

        self.confidence_var.trace_add('write', on_confidence_changed)

        # 摄像头预览控制
        preview_frame = ttk.LabelFrame(right_frame, text="预览控制", padding=10)
        preview_frame.pack(fill=tk.X, pady=(0, 10))

        self.quality_var = tk.StringVar(value="高")
        ttk.Radiobutton(preview_frame, text="高 (清晰)", variable=self.quality_var,
                        value="高").pack(anchor=tk.W)
        ttk.Radiobutton(preview_frame, text="中 (平衡)", variable=self.quality_var,
                        value="中").pack(anchor=tk.W)
        ttk.Radiobutton(preview_frame, text="低 (流畅)", variable=self.quality_var,
                        value="低").pack(anchor=tk.W)

        # 显示选项
        ttk.Label(preview_frame, text="显示选项:").pack(anchor=tk.W, pady=(5, 0))

        self.show_fps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preview_frame, text="显示FPS",
                        variable=self.show_fps_var).pack(anchor=tk.W)

        self.show_info_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preview_frame, text="显示摄像头信息",
                        variable=self.show_info_var).pack(anchor=tk.W)

        self.show_results_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preview_frame, text="显示识别结果",
                        variable=self.show_results_var).pack(anchor=tk.W)

        # 结果显示面板
        results_frame = ttk.LabelFrame(right_frame, text="识别结果", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # 结果文本框
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_realtime_recognition(self):
        """切换实时识别状态 - 更新置信度阈值"""
        self.enable_realtime_recognition = self.realtime_recognition_var.get()

        if self.enable_realtime_recognition:
            # 更新识别间隔
            try:
                self.recognition_interval = float(self.interval_var.get())
            except:
                self.recognition_interval = 0.5

            # 更新置信度阈值
            self.config.confidence_threshold = self.confidence_var.get()

            self.status_var.set(f"Real-time OCR ON | "
                                f"Interval: {self.recognition_interval}s | "
                                f"Conf: {self.config.confidence_threshold:.2f}")
            logger.info(f"实时识别已开启 (阈值: {self.config.confidence_threshold:.2f})")
        else:
            self.status_var.set("Real-time OCR OFF")
            logger.info("实时识别已关闭")

    def _start_recognition_processor(self):
        """启动识别结果处理线程"""

        def process_recognition_results():
            while True:
                try:
                    # 从队列获取识别结果
                    results = self.recognition_queue.get(timeout=0.1)

                    if results and self.enable_realtime_recognition:
                        # 更新显示结果
                        self.root.after(0, lambda: self._update_realtime_results(results))

                        # 更新识别FPS
                        self.recognition_count += 1
                        current_time = time.time()
                        if current_time - self.last_recognition_fps_time >= 1.0:
                            self.recognition_fps = self.recognition_count
                            self.recognition_count = 0
                            self.last_recognition_fps_time = current_time

                        # 更新性能显示
                        self.root.after(0, self._update_performance_display)

                except queue.Empty:
                    time.sleep(0.01)
                except Exception as e:
                    logger.error(f"处理识别结果出错: {e}")
                    time.sleep(0.1)

        # 启动处理线程
        processor_thread = threading.Thread(target=process_recognition_results, daemon=True)
        processor_thread.start()

    def _start_gui_update_timer(self):
        """启动GUI更新定时器"""

        def update_gui():
            # 更新摄像头显示
            self._update_camera_display()

            # 继续下一次更新
            self.root.after(33, update_gui)

        self.root.after(100, update_gui)

    def start_camera(self):
        """手动启动摄像头"""
        if self.is_camera_running:
            messagebox.showinfo("提示", "摄像头已经在运行")
            return

        try:
            camera_id = int(self.camera_var.get())
            resolution = self.resolution_var.get().split('x')
            width = int(resolution[0])
            height = int(resolution[1])

            # 打开摄像头
            self.cap = cv2.VideoCapture(camera_id)

            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {camera_id}")

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 启动摄像头线程
            self.is_camera_running = True
            self.camera_thread = threading.Thread(target=self._camera_capture_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()

            self.camera_status.config(text="运行中", foreground="green")
            self.status_var.set(f"摄像头 {camera_id} 已启动 - {width}x{height}")

            logger.info(f"摄像头 {camera_id} 启动成功，分辨率: {width}x{height}")

        except Exception as e:
            messagebox.showerror("错误", f"启动摄像头失败: {e}")
            logger.error(f"启动摄像头失败: {e}")

    def stop_camera(self):
        """停止摄像头"""
        self.is_camera_running = False
        self.enable_realtime_recognition = False
        self.realtime_recognition_var.set(False)

        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.camera_status.config(text="已停止", foreground="red")
        self.status_var.set("摄像头已停止")

        # 清除画布
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="摄像头已停止",
                                font=("Arial", 16), fill="white")

        logger.info("摄像头已停止")

    def _camera_capture_loop(self):
        """摄像头捕获循环 - 包含实时识别"""
        logger.info("开始摄像头捕获循环")

        while self.is_camera_running and self.cap and self.cap.isOpened():
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

                # 更新相机FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.camera_fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time

                # 实时识别 - 根据间隔执行
                if self.enable_realtime_recognition:
                    current_time = time.time()
                    if current_time - self.last_recognition_time >= self.recognition_interval:
                        # 在新线程中执行识别，避免阻塞摄像头
                        threading.Thread(target=self._async_recognize_frame,
                                         args=(frame.copy(),), daemon=True).start()
                        self.last_recognition_time = current_time

                # 控制捕获速率
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"摄像头捕获出错: {e}")
                time.sleep(0.1)

        logger.info("摄像头捕获循环结束")

    def _async_recognize_frame(self, frame):
        """异步识别帧 - 应用置信度过滤"""
        try:
            # 降低分辨率以提高识别速度
            if self.quality_var.get() == "低":
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (w // 2, h // 2))
            elif self.quality_var.get() == "中":
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * 0.75), int(h * 0.75)))

            # OCR识别 - 传入置信度阈值
            confidence_threshold = self.confidence_var.get()
            results = self.ocr_core.process_image(frame, confidence_threshold)

            # 将结果放入队列
            if results:
                self.recognition_queue.put(results)

        except Exception as e:
            logger.error(f"异步识别帧出错: {e}")

    def _update_camera_display(self):
        """更新摄像头显示 - 添加实时识别结果"""
        if not self.is_camera_running or self.current_frame is None:
            return

        # 控制显示帧率
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_update_interval:
            return

        self.last_frame_time = current_time

        try:
            with self.frame_lock:
                if self.current_frame is None:
                    return
                frame = self.current_frame.copy()

            # 自适应调整显示
            display_frame = self._prepare_display_frame(frame)

            # 如果有实时识别结果，绘制到画面上
            if self.enable_realtime_recognition and self.show_results_var.get():
                if hasattr(self, 'last_recognition_results') and self.last_recognition_results:
                    # 调整识别结果的坐标到显示尺寸
                    adjusted_results = self._adjust_results_for_display(
                        self.last_recognition_results,
                        frame.shape,
                        display_frame.shape
                    )
                    # 绘制结果
                    display_frame = self.ocr_core.draw_results(display_frame, adjusted_results)

            # 添加信息覆盖层
            display_frame = self._add_info_overlay(display_frame, frame)

            # 显示图像
            self.display_image(display_frame)

        except Exception as e:
            logger.error(f"更新摄像头显示时出错: {e}")

    def _prepare_display_frame(self, frame):
        """准备用于显示的帧"""
        # 获取画布尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width, canvas_height = self.target_display_size

        frame_height, frame_width = frame.shape[:2]

        # 根据预览质量调整处理
        quality = self.quality_var.get()
        if quality == "低":
            # 降低处理分辨率以提高性能
            frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
            frame_height, frame_width = frame.shape[:2]
        elif quality == "中":
            frame = cv2.resize(frame, (int(frame_width * 0.75), int(frame_height * 0.75)))
            frame_height, frame_width = frame.shape[:2]

        # 计算缩放比例
        width_scale = canvas_width / frame_width
        height_scale = canvas_height / frame_height

        # 保持宽高比
        scale = min(width_scale, height_scale)

        # 限制缩放范围
        scale = max(0.1, min(scale, 2.0))

        # 应用用户设置的缩放
        scale *= self.display_scale

        # 调整大小
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        if new_width != frame_width or new_height != frame_height:
            resized = cv2.resize(frame, (new_width, new_height))
        else:
            resized = frame

        return resized

    def _add_info_overlay(self, display_frame, original_frame):
        """添加信息覆盖层 - 显示置信度阈值"""
        overlay = display_frame.copy()

        y_pos = 25

        # 1. FPS显示
        if self.show_fps_var.get():
            cam_fps_text = f"FPS:{self.camera_fps:.1f}"
            cv2.putText(overlay, cam_fps_text,
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            y_pos += 25

            if self.enable_realtime_recognition:
                ocr_fps_text = f"OCR:{self.recognition_fps:.1f}"
                cv2.putText(overlay, ocr_fps_text,
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)
                y_pos += 25

        # 2. 识别状态
        status_color = (0, 255, 0) if self.enable_realtime_recognition else (100, 100, 100)
        status_text = "OCR:ON" if self.enable_realtime_recognition else "OCR:OFF"
        cv2.putText(overlay, status_text,
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    status_color, 2)
        y_pos += 25

        # 3. 识别间隔和置信度阈值
        if self.enable_realtime_recognition:
            interval_text = f"INT:{self.recognition_interval:.1f}s"
            cv2.putText(overlay, interval_text,
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 255, 200), 1)
            y_pos += 20

            # 显示置信度阈值
            conf_text = f"CONF:{self.confidence_var.get():.2f}"
            cv2.putText(overlay, conf_text,
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 200, 100), 1)
            y_pos += 20

        # 4. 摄像头信息
        if self.show_info_var.get():
            h, w = original_frame.shape[:2]
            src_text = f"SRC:{w}x{h}"
            cv2.putText(overlay, src_text,
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
            y_pos += 20

            dh, dw = display_frame.shape[:2]
            disp_text = f"DIS:{dw}x{dh}"
            cv2.putText(overlay, disp_text,
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 200, 0), 1)
            y_pos += 20

            scale_text = f"SCALE:{self.display_scale:.1f}x"
            cv2.putText(overlay, scale_text,
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 0), 1)

        return overlay

    def _update_performance_display(self):
        """更新性能显示"""
        self.performance_label.config(
            text=f"相机FPS: {self.camera_fps:.1f} | 识别FPS: {self.recognition_fps:.1f}"
        )

    def recognize_camera_frame(self):
        """单次识别摄像头当前帧"""
        if not self.is_camera_running or self.current_frame is None:
            messagebox.showwarning("警告", "摄像头未运行或没有可用的帧")
            return

        try:
            with self.frame_lock:
                if self.current_frame is None:
                    return
                frame = self.current_frame.copy()

            self.status_var.set("正在识别摄像头画面...")
            self.root.update()

            # OCR识别
            results = self.ocr_core.process_image(frame)
            self.current_results = results
            self.last_recognition_results = results

            # 在帧上绘制结果
            display_frame = self.ocr_core.draw_results(frame.copy(), results)
            self.display_image(display_frame)

            # 显示文本结果
            self.display_results(results)

            self.status_var.set(f"识别完成，找到 {len(results)} 个文本区域")

            logger.info(f"摄像头帧识别完成，找到 {len(results)} 个文本")

        except Exception as e:
            messagebox.showerror("错误", f"识别摄像头画面失败: {e}")
            self.status_var.set("识别失败")
            logger.error(f"识别摄像头画面失败: {e}")

    def recognize_current_frame(self):
        """识别当前显示的画面（可以是摄像头或图片）"""
        if not hasattr(self, 'current_display_image') or self.current_display_image is None:
            messagebox.showwarning("警告", "没有可识别的画面")
            return

        try:
            # 如果是摄像头模式，需要获取原始帧
            if self.is_camera_running and hasattr(self, 'current_frame'):
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                    else:
                        return
            else:
                # 图片模式，使用当前显示图像
                frame = self.current_display_image

            self.status_var.set("正在识别...")
            self.root.update()

            # OCR识别
            results = self.ocr_core.process_image(frame)
            self.current_results = results
            self.last_recognition_results = results

            # 绘制结果并显示
            display_frame = self.ocr_core.draw_results(frame.copy(), results)
            self.display_image(display_frame)

            # 显示文本结果
            self.display_results(results)

            self.status_var.set(f"识别完成，找到 {len(results)} 个文本")

        except Exception as e:
            messagebox.showerror("错误", f"识别失败: {e}")
            self.status_var.set("识别失败")

    def display_image(self, image):
        """显示图片到GUI"""
        if image is None:
            return

        # 保存用于缩放的图像
        self.current_display_image = image

        # 转换颜色空间
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为PIL图像
        pil_image = Image.fromarray(rgb_image)

        # 获取画布尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 10 and canvas_height > 10:
            # 调整PIL图像大小以适应画布
            pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(pil_image)

        # 清除画布并显示新图像
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def open_image(self):
        """打开图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path: str):
        """加载图片"""
        try:
            # 切换到图片模式
            self.switch_to_image_mode()

            # 读取图片
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("错误", "无法读取图片文件")
                return

            self.current_image = image
            self.current_image_path = file_path

            # 显示图片
            self.display_image(image)

            # 更新状态
            self.status_var.set(f"已加载: {Path(file_path).name}")

        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {e}")

    def save_results(self):
        """保存结果"""
        if not self.current_results:
            messagebox.showwarning("警告", "没有可保存的结果")
            return

        try:
            # 创建图片处理器
            processor = ImageProcessor(self.ocr_core)

            # 如果有当前图片，保存结果
            if hasattr(self, 'current_image_path') and self.current_image_path:
                processor._save_results(
                    self.current_image_path,
                    self.current_image,
                    self.current_results
                )
                messagebox.showinfo("成功", "结果已保存")
            else:
                # 如果是摄像头模式，先保存当前帧
                if self.is_camera_running and hasattr(self, 'current_frame'):
                    self.capture_frame()
                    if hasattr(self, 'current_image_path'):
                        processor._save_results(
                            self.current_image_path,
                            self.current_image,
                            self.current_results
                        )
                        messagebox.showinfo("成功", "图像和结果已保存")
                else:
                    messagebox.showwarning("警告", "没有可保存的图像")

        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")

    def capture_frame(self):
        """捕获当前帧并保存"""
        if not self.is_camera_running or self.current_frame is None:
            messagebox.showwarning("警告", "摄像头未运行或没有可用的帧")
            return

        try:
            # 获取当前帧
            with self.frame_lock:
                if self.current_frame is None:
                    return
                frame = self.current_frame.copy()

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_capture_{timestamp}.jpg"
            filepath = f"{self.config.output_dir}/{filename}"

            # 确保输出目录存在
            import os
            os.makedirs(self.config.output_dir, exist_ok=True)

            # 保存图像
            cv2.imwrite(filepath, frame)

            # 切换到图片模式
            self.switch_to_image_mode()
            self.current_image = frame
            self.current_image_path = filepath
            self.display_image(frame)

            self.status_var.set(f"已捕获并保存: {filename}")
            messagebox.showinfo("成功", f"图像已保存到: {filepath}")

            logger.info(f"摄像头帧已保存: {filepath}")

        except Exception as e:
            messagebox.showerror("错误", f"捕获帧失败: {e}")
            logger.error(f"捕获帧失败: {e}")

    def capture_and_recognize(self):
        """拍照并识别"""
        self.capture_frame()
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.recognize_current_frame()

    def batch_process(self):
        """批量处理"""
        file_paths = filedialog.askopenfilenames(
            title="选择多个图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )

        if not file_paths:
            return

        try:
            self.status_var.set("批量处理中...")
            self.root.update()

            # 创建图片处理器
            processor = ImageProcessor(self.ocr_core)

            # 处理图片
            all_results = processor.process_batch(list(file_paths), save_results=True)

            # 统计结果
            total_images = len(all_results)
            successful = sum(1 for r in all_results.values() if r)

            # 显示结果
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"批量处理完成\n")
            self.results_text.insert(tk.END, f"总图片数: {total_images}\n")
            self.results_text.insert(tk.END, f"成功处理: {successful}\n")
            self.results_text.insert(tk.END, f"结果保存在: {self.config.output_dir}\n")

            self.status_var.set(f"批量处理完成: {successful}/{total_images}")

            messagebox.showinfo("完成", f"批量处理完成！\n成功处理: {successful}/{total_images} 张图片")

        except Exception as e:
            messagebox.showerror("错误", f"批量处理失败: {e}")
            self.status_var.set("批量处理失败")

    def switch_to_camera_mode(self):
        """切换到摄像头模式"""
        if not self.is_camera_running:
            # 显示提示
            self.canvas.delete("all")
            self.canvas.create_text(400, 300, text="点击'启动'按钮开始摄像头",
                                    font=("Arial", 16), fill="white")

        self.status_var.set("摄像头模式")

    def switch_to_image_mode(self):
        """切换到图片模式"""
        # 显示提示
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="点击'打开图片'按钮加载图像",
                                font=("Arial", 16), fill="white")

        self.status_var.set("图片模式")

    def _on_canvas_resize(self, event):
        """画布大小变化事件"""
        # 更新目标显示尺寸
        self.target_display_size = (event.width, event.height)

        # 如果是适合窗口模式，重新调整显示
        if hasattr(self, 'current_display_image'):
            self.display_image(self.current_display_image)

    def _on_scale_changed(self, value):
        """缩放比例改变"""
        scale_percent = float(value)
        self.display_scale = scale_percent / 100.0
        self.scale_label.config(text=f"{scale_percent:.0f}%")

        if hasattr(self, 'current_display_image'):
            self.display_image(self.current_display_image)

    def _fit_to_window(self):
        """适应窗口大小"""
        if hasattr(self, 'canvas') and self.canvas.winfo_width() > 10:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if hasattr(self, 'current_display_image'):
                img_height, img_width = self.current_display_image.shape[:2]

                width_scale = canvas_width / img_width
                height_scale = canvas_height / img_height
                scale = min(width_scale, height_scale) * 0.95

                scale_percent = scale * 100
                self.scale_var.set(scale_percent)
                self.display_scale = scale
                self.scale_label.config(text=f"{scale_percent:.0f}%")

                self.display_image(self.current_display_image)

    def _original_size(self):
        """原始大小"""
        self.scale_var.set(100)
        self.display_scale = 1.0
        self.scale_label.config(text="100%")

        if hasattr(self, 'current_display_image'):
            self.display_image(self.current_display_image)

    def toggle_fullscreen(self):
        """切换全屏状态"""
        is_fullscreen = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not is_fullscreen)

    def on_closing(self):
        """关闭窗口事件"""
        if self.is_camera_running:
            self.stop_camera()

        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            self.root.destroy()

    def run(self):
        """运行GUI"""
        self.root.mainloop()

    def _update_realtime_results(self, results):
        """更新实时识别结果 - 过滤低置信度"""
        if not results:
            return

        # 过滤低置信度的结果
        confidence_threshold = self.confidence_var.get()
        filtered_results = [r for r in results if r.confidence >= confidence_threshold]

        if not filtered_results:
            return

        self.last_recognition_results = filtered_results

        # 更新结果显示
        self.results_text.delete(1.0, tk.END)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] 实时识别结果 (置信度≥{confidence_threshold:.2f}):\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")

        for i, result in enumerate(filtered_results):
            if result.text:
                self.results_text.insert(tk.END, f"[区域 {i + 1}]\n")
                self.results_text.insert(tk.END, f"文本: {result.text}\n")
                self.results_text.insert(tk.END, f"置信度: {result.confidence:.3f}\n")
                self.results_text.insert(tk.END, f"检测方法: {result.detection_source}\n")
                self.results_text.insert(tk.END, "-" * 40 + "\n")

        # 显示过滤统计
        filtered_count = len(results) - len(filtered_results)
        if filtered_count > 0:
            self.results_text.insert(tk.END, f"\n[过滤] {filtered_count} 个低置信度结果被忽略\n")

        # 滚动到顶部
        self.results_text.see(1.0)

    def display_results(self, results):
        """显示识别结果 - 过滤低置信度"""
        self.results_text.delete(1.0, tk.END)

        if not results:
            self.results_text.insert(tk.END, "未识别到文本\n")
            return

        # 过滤低置信度的结果
        confidence_threshold = self.confidence_var.get()
        filtered_results = [r for r in results if r.confidence >= confidence_threshold]

        if not filtered_results:
            self.results_text.insert(tk.END, f"未找到置信度≥{confidence_threshold:.2f}的文本\n")
            return

        for i, result in enumerate(filtered_results):
            if result.text:
                self.results_text.insert(tk.END, f"[区域 {i + 1}]\n")
                self.results_text.insert(tk.END, f"文本: {result.text}\n")
                self.results_text.insert(tk.END, f"置信度: {result.confidence:.3f}\n")
                self.results_text.insert(tk.END, f"检测方法: {result.detection_source}\n")
                self.results_text.insert(tk.END, "-" * 40 + "\n")

        # 显示过滤统计
        filtered_count = len(results) - len(filtered_results)
        if filtered_count > 0:
            self.results_text.insert(tk.END, f"\n[过滤] {filtered_count} 个低置信度结果被忽略\n")

    def _adjust_results_for_display(self, results, original_shape, display_shape):
        """调整识别结果到显示尺寸 - 同时过滤低置信度"""
        import copy

        # 先过滤低置信度的结果
        confidence_threshold = self.confidence_var.get()
        filtered_results = [r for r in results if r.confidence >= confidence_threshold]

        adjusted_results = []
        for result in filtered_results:
            # 复制结果
            adjusted = copy.copy(result)

            # 调整边界框坐标
            box = result.bbox
            original_h, original_w = original_shape[:2]
            display_h, display_w = display_shape[:2]

            # 计算缩放比例
            scale_x = display_w / original_w
            scale_y = display_h / original_h

            # 应用缩放
            adjusted_box = [
                int(box[0] * scale_x),
                int(box[1] * scale_y),
                int(box[2] * scale_x),
                int(box[3] * scale_y)
            ]
            adjusted.bbox = adjusted_box

            adjusted_results.append(adjusted)

        return adjusted_results