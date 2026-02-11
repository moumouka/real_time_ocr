"""
主程序入口
"""
import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 检查命令行参数
    if len(sys.argv) > 1 and '--gui' in sys.argv:
        # GUI模式
        from ui.gui import OCRGUI
        app = OCRGUI()
        app.run()
    else:
        # 命令行模式
        from ui.cli import CommandLineInterface
        cli = CommandLineInterface()
        cli.run()


if __name__ == "__main__":
    main()
