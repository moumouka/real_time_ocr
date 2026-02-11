"""
命令行接口模块
"""
import argparse
import sys
import os
from pathlib import Path
import cv2
from datetime import datetime

from core.ocr_core import OCRCore
from core.config import ConfigManager
from modes.image_mode import ImageProcessor
from modes.realtime_mode import RealTimeProcessor


class CommandLineInterface:
    """命令行接口"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='智能OCR系统 - 命令行版本',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  %(prog)s --mode image --input test.jpg
  %(prog)s --mode realtime --camera 0
  %(prog)s --mode batch --input images/
            """
        )

        self._setup_arguments()

    def _setup_arguments(self):
        """设置命令行参数"""
        # 模式选择
        self.parser.add_argument('--mode', choices=['image', 'realtime', 'batch'],
                                 default='image', help='工作模式')

        # 输入输出
        self.parser.add_argument('--input', help='输入图片路径或目录')
        self.parser.add_argument('--output', default='outputs', help='输出目录')

        # 摄像头设置
        self.parser.add_argument('--camera', type=int, default=0, help='摄像头ID')

        # 配置
        self.parser.add_argument('--config', help='配置文件路径')
        self.parser.add_argument('--debug', action='store_true', help='调试模式')

        # 性能选项
        self.parser.add_argument('--gpu', action='store_true', help='使用GPU')
        self.parser.add_argument('--confidence', type=float, default=0.4,
                                 help='置信度阈值')

    def run(self):
        """运行命令行接口"""
        args = self.parser.parse_args()

        # 设置日志级别
        import logging
        if args.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # 配置
        if args.config and os.path.exists(args.config):
            config_manager = ConfigManager(args.config)
            config = config_manager.get_config()
        else:
            from core.config import SystemConfig
            config = SystemConfig()

        # 覆盖命令行参数
        config.use_gpu = args.gpu
        config.confidence_threshold = args.confidence
        config.output_dir = args.output
        os.makedirs(config.output_dir, exist_ok=True)

        # 创建OCR核心
        ocr_core = OCRCore(config=config)

        # 根据模式执行
        if args.mode == 'image':
            self._run_image_mode(args, ocr_core)
        elif args.mode == 'realtime':
            self._run_realtime_mode(args, ocr_core)
        elif args.mode == 'batch':
            self._run_batch_mode(args, ocr_core)

    def _run_image_mode(self, args, ocr_core):
        """图片模式"""
        if not args.input:
            print("错误: 图片模式需要指定输入图片路径")
            return

        if not os.path.exists(args.input):
            print(f"错误: 文件不存在: {args.input}")
            return

        print(f"处理图片: {args.input}")

        # 读取图片
        image = cv2.imread(args.input)
        if image is None:
            print(f"错误: 无法读取图片 {args.input}")
            return

        # 处理图片
        results = ocr_core.process_image(image)

        # 显示结果
        print(f"\n识别结果 ({len(results)} 个文本区域):")
        for i, result in enumerate(results):
            if result.text:
                print(f"[{i + 1}] {result.text}")
                print(f"    置信度: {result.confidence:.3f}")
                print(f"    检测方法: {result.detection_source}")
                print()

        # 保存结果
        if results:
            processor = ImageProcessor(ocr_core)
            processor._save_results(args.input, image, results)
            print(f"结果已保存到: {ocr_core.config.output_dir}")

    def _run_realtime_mode(self, args, ocr_core):
        """实时模式"""
        print(f"启动实时OCR，摄像头ID: {args.camera}")
        print("按 'q' 键退出")
        print("-" * 50)

        # 创建实时处理器
        realtime_processor = RealTimeProcessor(ocr_core)

        # 结果回调
        def display_results(results):
            if results:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
                for result in results:
                    if result.text:
                        print(f"  - {result.text} ({result.confidence:.3f})")

        realtime_processor.set_callback(result_callback=display_results)

        # 启动摄像头
        if realtime_processor.start_camera(args.camera):
            try:
                while True:
                    frame = realtime_processor.get_current_frame()
                    if frame is not None:
                        cv2.imshow('Real-time OCR', frame)

                    # 按q退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                print("\n正在停止摄像头...")
            finally:
                realtime_processor.stop_camera()
                cv2.destroyAllWindows()
                print("实时OCR已停止")
        else:
            print("错误: 无法启动摄像头")

    def _run_batch_mode(self, args, ocr_core):
        """批量模式"""
        if not args.input:
            print("错误: 批量模式需要指定输入目录或文件列表")
            return

        # 获取图片列表
        image_paths = []
        if os.path.isdir(args.input):
            # 目录模式
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_paths.extend(Path(args.input).glob(ext))
            image_paths = [str(f) for f in image_paths]
        elif os.path.isfile(args.input) and args.input.endswith('.txt'):
            # 文件列表模式
            with open(args.input, 'r', encoding='utf-8') as f:
                image_paths = [line.strip() for line in f if line.strip()]
        else:
            # 单个文件
            image_paths = [args.input]

        if not image_paths:
            print(f"错误: 在 {args.input} 中未找到图片文件")
            return

        print(f"批量处理 {len(image_paths)} 张图片...")

        # 创建处理器
        processor = ImageProcessor(ocr_core)

        # 处理图片
        all_results = processor.process_batch(image_paths, save_results=True)

        # 统计结果
        successful = sum(1 for r in all_results.values() if r)
        total = len(all_results)

        print(f"\n批量处理完成: {successful}/{total} 张图片处理成功")
        print(f"结果保存在: {ocr_core.config.output_dir}")

        # 显示统计
        stats = ocr_core.get_statistics()
        if stats.total_images > 0:
            avg_time = stats.total_time / stats.total_images
            print(f"平均处理时间: {avg_time:.2f}秒/图片")
            print(f"YOLO成功率: {stats.yolo_success / stats.total_images * 100:.1f}%")


def main():
    """命令行主函数"""
    cli = CommandLineInterface()
    cli.run()


if __name__ == "__main__":
    main()
    