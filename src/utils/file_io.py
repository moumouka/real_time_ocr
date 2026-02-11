"""
文件IO工具模块
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import cv2

from core.recognizers import OCRResult


def save_results_to_files(image_path: str, results: List[OCRResult],
                          output_dir: str, config) -> Dict[str, str]:
    """保存结果到文件"""
    img_name = Path(image_path).stem

    # 文件路径
    txt_path = os.path.join(output_dir, f"{img_name}_results.txt")
    json_path = os.path.join(output_dir, f"{img_name}_results.json")

    # 保存文本文件
    save_text_results(txt_path, image_path, results)

    # 保存JSON文件
    save_json_results(json_path, image_path, results)

    return {
        'text': txt_path,
        'json': json_path
    }


def save_text_results(filepath: str, image_path: str, results: List[OCRResult]):
    """保存为文本文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"图像: {image_path}\n")
        f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"检测到 {len(results)} 个文本区域\n")
        f.write("=" * 50 + "\n\n")

        for i, result in enumerate(results):
            if result.text:
                f.write(f"[区域 {i + 1}]\n")
                f.write(f"文本: {result.text}\n")
                f.write(f"边界框: {result.bbox}\n")
                f.write(f"置信度: {result.confidence:.3f}\n")
                f.write(f"检测方法: {result.detection_source}\n")
                f.write("-" * 30 + "\n")


def save_json_results(filepath: str, image_path: str, results: List[OCRResult]):
    """保存为JSON文件"""
    results_dict = {
        'image_path': image_path,
        'timestamp': datetime.now().isoformat(),
        'results': [
            {
                'text': r.text,
                'bbox': r.bbox,
                'confidence': float(r.confidence),
                'detection_source': r.detection_source
            }
            for r in results if r.text
        ]
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)


def save_visualization(image: np.ndarray, results: List[OCRResult],
                       filepath: str, config):
    """保存可视化结果"""
    if len(results) == 0:
        return

    from utils.visualization import draw_ocr_results
    display_frame = draw_ocr_results(image.copy(), results, config)
    cv2.imwrite(filepath, display_frame)


def get_image_files(directory: str) -> List[str]:
    """获取目录中的所有图片文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []

    for file in Path(directory).iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(str(file))

    return image_files
