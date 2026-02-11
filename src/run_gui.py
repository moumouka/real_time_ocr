"""
运行图形界面 - 手动摄像头控制版本
"""
import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """主函数"""
    print("=" * 60)
    print("智能OCR系统 - 手动摄像头控制版本")
    print("=" * 60)
    print("\n功能说明:")
    print("1. 手动控制摄像头启动/停止")
    print("2. 自适应显示区域")
    print("3. 支持拍照和实时识别")
    print("4. 图片和摄像头模式切换")
    print("\n使用提示:")
    print("- 在'摄像头控制'面板中启动/停止摄像头")
    print("- 可以调整分辨率以获得最佳效果")
    print("- 使用缩放控制调整显示大小")
    print("- '适合窗口'按钮自动调整到最佳显示")
    print()

    from ui.gui import OCRGUI

    try:
        app = OCRGUI()
        app.run()
    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()
        input("\n按Enter键退出...")


if __name__ == "__main__":
    main()