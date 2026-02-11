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

logger = logging.getLogger(__name__)


class OCRGUI:
    """OCR图形用户界面 - 支持手动摄像头控制"""
    
    def __init__(self, config_path: str = None):
        self.root = tk.Tk()
        self.root.title("智能OCR系统 v3.0 - 手动摄像头控制")
        self.root.geometry("1200x700")
        
        # 初始化OCR核心
        from core.ocr_core import OCRCore
        from core.config import ConfigManager
        
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.ocr_core = OCRCore(config=self.config)
        
        # 摄像头相关变量
        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # 显示相关变量
        self.display_scale = 1.0
        self.original_frame_size = (0, 0)
        self.last_frame_time = 0
        self.frame_update_interval = 0.033  # 30 FPS的GUI更新
        self.target_display_size = (800, 600)  # 目标显示尺寸
        
        # 性能监控
        self.camera_fps = 0
        self.display_fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # 当前状态
        self.current_image = None
        self.current_image_path = None
        self.current_results = []
        self.is_realtime_mode = False
        
        # 创建界面
        self._create_widgets()
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 启动GUI更新定时器
        self._start_gui_update_timer()
    
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
        camera_menu.add_command(label="录制视频", command=self.toggle_recording)
        
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
        
        # 摄像头控制面板（在画布上方）
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
        ttk.Button(control_row2, text="识别画面", 
                  command=self.recognize_camera_frame).pack(side=tk.LEFT, padx=5)
        
        # 性能显示
        self.performance_label = ttk.Label(control_row2, text="FPS: 0.0")
        self.performance_label.pack(side=tk.RIGHT, padx=10)
        
        # 图像控制面板（在画布下方）
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
        
        # 摄像头预览控制
        preview_frame = ttk.LabelFrame(right_frame, text="预览控制", padding=10)
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 预览质量
        ttk.Label(preview_frame, text="预览质量:").pack(anchor=tk.W)
        self.quality_var = tk.StringVar(value="高")
        ttk.Radiobutton(preview_frame, text="高 (清晰)", variable=self.quality_var,
                       value="高").pack(anchor=tk.W)
        ttk.Radiobutton(preview_frame, text="中 (平衡)", variable=self.quality_var,
                       value="中").pack(anchor=tk.W)
        ttk.Radiobutton(preview_frame, text="低 (流畅)", variable=self.quality_var,
                       value="低").pack(anchor=tk.W)
        
        # 显示选项
        ttk.Label(preview_frame, text="显示选项:").pack(anchor=tk.W, pady=(5,0))
        
        self.show_fps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preview_frame, text="显示FPS", 
                       variable=self.show_fps_var).pack(anchor=tk.W)
        
        self.show_info_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preview_frame, text="显示摄像头信息", 
                       variable=self.show_info_var).pack(anchor=tk.W)
        
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
                scale = min(width_scale, height_scale) * 0.95  # 留一点边距
                
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
        """摄像头捕获循环"""
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
                
                # 更新FPS计算
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.camera_fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # 稍微休息，避免占用太多CPU
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"摄像头捕获出错: {e}")
                time.sleep(0.1)
        
        logger.info("摄像头捕获循环结束")
    
    def _start_gui_update_timer(self):
        """启动GUI更新定时器"""
        def update_gui():
            # 更新摄像头显示
            self._update_camera_display()
            
            # 继续下一次更新
            self.root.after(33, update_gui)  # 约30 FPS
        
        self.root.after(100, update_gui)
    
    def _update_camera_display(self):
        """更新摄像头显示"""
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
            
            # 添加信息覆盖层
            display_frame = self._add_info_overlay(display_frame, frame)
            
            # 显示图像
            self.display_image(display_frame)
            
            # 更新性能显示
            self.performance_label.config(text=f"FPS: {self.camera_fps:.1f}")
            
        except Exception as e:
            logger.error(f"更新摄像头显示时出错: {e}")
    
    def _prepare_display_frame(self, frame):
        """准备用于显示的帧 - 自适应调整大小"""
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
        """添加信息覆盖层"""
        overlay = display_frame.copy()
        
        if self.show_fps_var.get():
            # 添加FPS显示
            cv2.putText(overlay, f"FPS: {self.camera_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0), 2)
        
        if self.show_info_var.get():
            # 添加摄像头信息
            height, width = original_frame.shape[:2]
            cv2.putText(overlay, f"原始: {width}x{height}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 255), 2)
            
            disp_height, disp_width = display_frame.shape[:2]
            cv2.putText(overlay, f"显示: {disp_width}x{disp_height}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (255, 255, 0), 2)
            
            cv2.putText(overlay, f"缩放: {self.display_scale:.2f}x", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (255, 200, 0), 2)
        
        return overlay
    
    def display_image(self, image):
        """显示图片到GUI"""
        # 保存用于缩放的图像
        self.current_display_image = image
        
        # 转换颜色空间
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # 清除画布并显示新图像
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # 保持引用
    
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
    
    def recognize_camera_frame(self):
        """识别摄像头当前帧"""
        if not self.is_camera_running or self.current_frame is None:
            messagebox.showwarning("警告", "摄像头未运行或没有可用的帧")
            return
        
        try:
            # 获取当前帧
            with self.frame_lock:
                if self.current_frame is None:
                    return
                frame = self.current_frame.copy()
            
            # 显示处理中的状态
            self.status_var.set("正在识别摄像头画面...")
            self.root.update()
            
            # 进行OCR识别
            results = self.ocr_core.process_image(frame)
            self.current_results = results
            
            # 在帧上绘制结果
            display_frame = self.ocr_core.draw_results(frame.copy(), results)
            self.display_image(display_frame)
            
            # 显示文本结果
            self.display_results(results)
            
            # 更新状态
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
            
            # 绘制结果并显示
            display_frame = self.ocr_core.draw_results(frame.copy(), results)
            self.display_image(display_frame)
            
            # 显示文本结果
            self.display_results(results)
            
            self.status_var.set(f"识别完成，找到 {len(results)} 个文本")
            
        except Exception as e:
            messagebox.showerror("错误", f"识别失败: {e}")
            self.status_var.set("识别失败")
    
    def display_results(self, results):
        """显示识别结果"""
        self.results_text.delete(1.0, tk.END)
        
        if not results:
            self.results_text.insert(tk.END, "未识别到文本\n")
            return
        
        for i, result in enumerate(results):
            if result.text:  # 只显示非空结果
                self.results_text.insert(tk.END, f"[区域 {i+1}]\n")
                self.results_text.insert(tk.END, f"文本: {result.text}\n")
                self.results_text.insert(tk.END, f"置信度: {result.confidence:.3f}\n")
                self.results_text.insert(tk.END, f"检测方法: {result.detection_source}\n")
                self.results_text.insert(tk.END, "-" * 40 + "\n")
    
    def switch_to_camera_mode(self):
        """切换到摄像头模式"""
        self.is_realtime_mode = True
        
        if not self.is_camera_running:
            # 显示提示
            self.canvas.delete("all")
            self.canvas.create_text(400, 300, text="点击'启动'按钮开始摄像头", 
                                   font=("Arial", 16), fill="white")
        
        self.status_var.set("摄像头模式")
    
    def switch_to_image_mode(self):
        """切换到图片模式"""
        self.is_realtime_mode = False
        
        # 显示提示
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="点击'打开图片'按钮加载图像", 
                               font=("Arial", 16), fill="white")
        
        self.status_var.set("图片模式")
    
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
            
            # 自动适应窗口
            self._fit_to_window()
            
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
            from modes.image_mode import ImageProcessor
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
                    # 然后保存结果
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
            
            from modes.image_mode import ImageProcessor
            processor = ImageProcessor(self.ocr_core)
            
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
    
    def capture_and_recognize(self):
        """拍照并识别"""
        self.capture_frame()
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.process_current_image()
    
    def toggle_recording(self):
        """切换录制状态"""
        # 这里可以添加视频录制功能
        messagebox.showinfo("提示", "视频录制功能开发中...")
    
    def toggle_fullscreen(self):
        """切换全屏状态"""
        is_fullscreen = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not is_fullscreen)
    
    def process_current_image(self):
        """处理当前图片"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先打开图片或启动摄像头")
            return
        
        try:
            self.status_var.set("正在识别...")
            self.root.update()
            
            results = self.ocr_core.process_image(self.current_image)
            self.current_results = results
            
            display_image = self.ocr_core.draw_results(self.current_image.copy(), results)
            self.display_image(display_image)
            self.display_results(results)
            
            self.status_var.set(f"识别完成，找到 {len(results)} 个文本区域")
            
        except Exception as e:
            messagebox.showerror("错误", f"识别失败: {e}")
            self.status_var.set("识别失败")
    
    def on_closing(self):
        """关闭窗口事件"""
        if self.is_camera_running:
            self.stop_camera()
        
        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            self.root.destroy()
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()


# 简单的测试函数
def test_camera_manual():
    """测试手动摄像头控制"""
    print("测试手动摄像头控制...")
    app = OCRGUI()
    app.run()


if __name__ == "__main__":
    test_camera_manual()