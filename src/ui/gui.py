"""
图形用户界面模块
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import logging
import cv2
from PIL import Image, ImageTk

from core.ocr_core import OCRCore
from core.config import SystemConfig, ConfigManager
from modes.image_mode import ImageProcessor
from modes.realtime_mode import RealTimeProcessor

logger = logging.getLogger(__name__)


class OCRGUI:
    """OCR图形用户界面"""

    def __init__(self, config_path: str = None):
        self.root = tk.Tk()
        self.root.title("智能OCR系统 v3.0")
        self.root.geometry("1200x700")

        # 初始化配置
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        # 初始化OCR核心
        self.ocr_core = OCRCore(config=self.config)

        # 初始化处理器
        self.image_processor = None
        self.realtime_processor = None

        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.current_results = []
        self.is_realtime_mode = False

        # 创建界面
        self._create_widgets()

        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

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

        # 模式菜单
        mode_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="模式", menu=mode_menu)
        mode_menu.add_command(label="图片模式", command=self.switch_to_image_mode)
        mode_menu.add_command(label="实时模式", command=self.switch_to_realtime_mode)

        # 主框架 - 左右布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧 - 图像显示区
        left_frame = ttk.Frame(main_frame, width=800)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 图像显示画布
        self.canvas = tk.Canvas(left_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 图像控制按钮
        img_control_frame = ttk.Frame(left_frame)
        img_control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(img_control_frame, text="打开图片",
                   command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(img_control_frame, text="开始识别",
                   command=self.process_current_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(img_control_frame, text="保存结果",
                   command=self.save_results).pack(side=tk.LEFT, padx=5)

        # 右侧 - 控制面板
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # 模式选择
        mode_frame = ttk.LabelFrame(right_frame, text="工作模式", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        self.mode_var = tk.StringVar(value="image")
        ttk.Radiobutton(mode_frame, text="图片模式", variable=self.mode_var,
                        value="image", command=self.on_mode_changed).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="实时模式", variable=self.mode_var,
                        value="realtime", command=self.on_mode_changed).pack(anchor=tk.W)

        # 实时控制面板
        self.realtime_control_frame = ttk.LabelFrame(right_frame, text="实时控制", padding=10)
        self.realtime_control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(self.realtime_control_frame, text="启动摄像头",
                   command=self.start_realtime).pack(fill=tk.X, pady=2)
        ttk.Button(self.realtime_control_frame, text="停止摄像头",
                   command=self.stop_realtime).pack(fill=tk.X, pady=2)

        # 摄像头选择
        ttk.Label(self.realtime_control_frame, text="摄像头ID:").pack(anchor=tk.W)
        self.camera_var = tk.StringVar(value="0")
        ttk.Combobox(self.realtime_control_frame, textvariable=self.camera_var,
                     values=["0", "1", "2", "3"]).pack(fill=tk.X, pady=2)

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

    def on_mode_changed(self):
        """模式切换事件"""
        mode = self.mode_var.get()
        if mode == "image":
            self.switch_to_image_mode()
        else:
            self.switch_to_realtime_mode()

    def switch_to_image_mode(self):
        """切换到图片模式"""
        if self.is_realtime_mode:
            self.stop_realtime()

        self.is_realtime_mode = False
        self.status_var.set("图片模式")

        # 清除画布
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="请打开图片文件",
                                font=("Arial", 16), fill="white")

    def switch_to_realtime_mode(self):
        """切换到实时模式"""
        self.is_realtime_mode = True
        self.status_var.set("实时模式 - 准备启动摄像头")

        # 清除画布
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="摄像头显示区域",
                                font=("Arial", 16), fill="white")

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
            # 读取图片
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                messagebox.showerror("错误", "无法读取图片文件")
                return

            self.current_image_path = file_path

            # 显示图片
            self.display_image(self.current_image)

            # 更新状态
            self.status_var.set(f"已加载: {Path(file_path).name}")

        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {e}")

    def display_image(self, image):
        """显示图片到GUI"""
        # 调整图片大小以适应显示区域
        h, w = image.shape[:2]
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 800
            canvas_height = 600

        # 计算缩放比例
        scale = min(canvas_width / w, canvas_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w < 1 or new_h < 1:
            new_w = w
            new_h = h

        # 调整大小并转换颜色空间
        resized = cv2.resize(image, (new_w, new_h))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 转换为PIL图像
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)

        # 更新画布
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # 保持引用

    def process_current_image(self):
        """处理当前图片"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先打开图片")
            return

        try:
            self.status_var.set("正在识别...")
            self.root.update()

            # 处理图片
            results = self.ocr_core.process_image(self.current_image)
            self.current_results = results

            # 在图片上绘制结果
            display_image = self.ocr_core.draw_results(self.current_image.copy(), results)

            # 显示结果
            self.display_image(display_image)

            # 显示文本结果
            self.display_results(results)

            # 更新状态
            self.status_var.set(f"识别完成，找到 {len(results)} 个文本区域")

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
                self.results_text.insert(tk.END, f"[区域 {i + 1}]\n")
                self.results_text.insert(tk.END, f"文本: {result.text}\n")
                self.results_text.insert(tk.END, f"置信度: {result.confidence:.3f}\n")
                self.results_text.insert(tk.END, f"检测方法: {result.detection_source}\n")
                self.results_text.insert(tk.END, "-" * 40 + "\n")

    def save_results(self):
        """保存结果"""
        if not self.current_results:
            messagebox.showwarning("警告", "没有可保存的结果")
            return

        try:
            # 创建图片处理器
            self.image_processor = ImageProcessor(self.ocr_core)

            # 保存结果
            self.image_processor._save_results(
                self.current_image_path,
                self.current_image,
                self.current_results
            )

            messagebox.showinfo("成功", "结果已保存")

        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")

    # 在 OCRGUI 类的 start_realtime 方法中添加：
    def start_realtime(self):
        """启动实时OCR"""
        if not self.is_realtime_mode:
            self.switch_to_realtime_mode()

        try:
            camera_id = int(self.camera_var.get())

            # 创建实时处理器
            self.realtime_processor = RealTimeProcessor(self.ocr_core)

            # 设置回调 - 使用线程安全的更新
            self.realtime_processor.set_callback(
                frame_callback=self._safe_realtime_frame_callback,
                result_callback=self._safe_realtime_results_callback
            )

            # 启动摄像头
            if self.realtime_processor.start_camera(camera_id):
                self.status_var.set(f"实时模式 - 摄像头 {camera_id} 已启动")
                # 启动GUI更新定时器
                self._start_gui_update_timer()
            else:
                messagebox.showerror("错误", "无法启动摄像头")
                self.realtime_processor = None

        except Exception as e:
            messagebox.showerror("错误", f"启动实时OCR失败: {e}")
            self.realtime_processor = None

    def _safe_realtime_frame_callback(self, frame):
        """线程安全的帧回调"""
        # 使用队列或直接在主线程中处理
        if hasattr(self, 'frame_queue'):
            try:
                self.frame_queue.put(frame, block=False)
            except:
                pass

    def _safe_realtime_results_callback(self, results):
        """线程安全的结果回调"""
        # 在主线程中更新结果
        self.root.after(0, lambda: self._update_realtime_results(results))

    def _start_gui_update_timer(self):
        """启动GUI更新定时器"""

        def update_gui():
            if hasattr(self, 'realtime_processor') and self.realtime_processor:
                # 获取最新帧
                frame = self.realtime_processor.get_current_frame()
                if frame is not None:
                    # 在GUI线程中更新显示
                    self.display_image(frame)

                # 获取统计信息
                stats = self.realtime_processor.get_statistics()
                self.status_var.set(
                    f"实时模式 - 捕获FPS: {stats['capture_fps']}, 处理FPS: {stats['processing_fps']:.1f}")

            # 继续下一次更新
            self.root.after(50, update_gui)  # 每50ms更新一次 (~20 FPS的GUI更新)

        # 初始化帧队列
        if not hasattr(self, 'frame_queue'):
            import queue
            self.frame_queue = queue.Queue(maxsize=10)

        # 启动定时器
        self.root.after(100, update_gui)

    def _update_realtime_results(self, results):
        """更新实时结果"""
        if not results:
            return

        # 只显示最新的结果
        self.results_text.delete(1.0, tk.END)

        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}]\n")

        for i, result in enumerate(results):
            if result.text:
                self.results_text.insert(tk.END, f"文本: {result.text}\n")
                self.results_text.insert(tk.END, f"置信度: {result.confidence:.3f}\n")
                self.results_text.insert(tk.END, "-" * 30 + "\n")

    def stop_realtime(self):
        """停止实时OCR"""
        if self.realtime_processor:
            self.realtime_processor.stop_camera()
            self.realtime_processor = None
            self.status_var.set("实时模式已停止")

    def on_realtime_frame(self, frame):
        """实时帧回调"""
        # 在GUI线程中更新显示
        self.root.after(0, lambda: self.display_image(frame))

    def on_realtime_results(self, results):
        """实时结果回调"""
        # 在GUI线程中更新结果
        if results:
            self.root.after(0, lambda: self.display_realtime_results(results))

    def display_realtime_results(self, results):
        """显示实时结果"""
        # 只显示最新的结果
        self.results_text.delete(1.0, tk.END)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}]\n")

        for i, result in enumerate(results):
            if result.text:
                self.results_text.insert(tk.END, f"文本: {result.text}\n")
                self.results_text.insert(tk.END, f"置信度: {result.confidence:.3f}\n")
                self.results_text.insert(tk.END, "-" * 30 + "\n")

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
            self.image_processor = ImageProcessor(self.ocr_core)

            # 处理图片
            all_results = self.image_processor.process_batch(list(file_paths), save_results=True)

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

    def on_closing(self):
        """关闭窗口事件"""
        if hasattr(self, 'realtime_processor') and self.realtime_processor:
            self.realtime_processor.stop_camera()

        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            self.root.destroy()

    def run(self):
        """运行GUI"""
        self.root.mainloop()
