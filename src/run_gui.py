"""
运行图形界面
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.gui import OCRGUI

if __name__ == "__main__":
    app = OCRGUI()
    app.run()
    