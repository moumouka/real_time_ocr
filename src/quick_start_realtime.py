"""
快速启动实时识别
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=" * 60)
    print("智能OCR系统 - 实时识别快速启动")
    print("=" * 60)
    print("\n请选择启动方式:")
    print("1. 图形界面版 (推荐)")
    print("2. 命令行测试版")
    print()

    choice = input("请输入选择 (1/2): ").strip()

    if choice == "1":
        # 启动图形界面
        from ui.gui import OCRGUI

        app = OCRGUI()
        app.run()
    elif choice == "2":
        # 启动命令行测试
        from test_realtime_recognition import test_realtime_recognition

        test_realtime_recognition()
    else:
        print("无效选择")