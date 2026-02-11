"""
运行命令行界面
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.cli import CommandLineInterface

if __name__ == "__main__":
    cli = CommandLineInterface()
    cli.run()
