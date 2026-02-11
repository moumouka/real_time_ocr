"""
图片模式处理模块
"""
import cv2
import os
from pathlib import Path
from typing import List, Dict
import logging

from core.ocr_core import OCRCore
from core.recognizers import OCRResult

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图片处理器"""

    def __init__(self, ocr_core: OCRCore):
        self.ocr_core = ocr_core
        self.config = ocr_core.config

    def process_single_image(self, image_path: str, save_results: bool = True) -> List[OCRResult]:
        """处理单张图片"""
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图片: {image_path}")
                return []

            logger.info(f"处理图片: {image_path}")

            # OCR处理
            results = self.ocr_core.process_image(image)

            # 保存结果
            if save_results:
                self._save_results(image_path, image, results)

            logger.info(f"完成处理，识别到 {len(results)} 个文本")
            return results

        except Exception as e:
            logger.error(f"处理图片 {image_path} 时出错: {e}")
            return []

    def process_batch(self, image_paths: List[str], save_results: bool = True) -> Dict[str, List[OCRResult]]:
        """批量处理图片"""
        all_results = {}

        for i, img_path in enumerate(image_paths):
            try:
                logger.info(f"处理图片 [{i + 1}/{len(image_paths)}]: {img_path}")

                results = self.process_single_image(img_path, save_results)
                all_results[img_path] = results

            except Exception as e:
                logger.error(f"处理图片 {img_path} 时出错: {e}")
                all_results[img_path] = []

        return all_results

    def _save_results(self, image_path: str, image: any, results: List[OCRResult]):
        """保存单张图片的结果"""
        import json
        from datetime import datetime

        img_name = Path(image_path).stem

        # 保存文本结果
        txt_path = os.path.join(self.config.output_dir, f"{img_name}_results.txt")
        json_path = os.path.join(self.config.output_dir, f"{img_name}_results.json")

        # 保存为文本文件
        with open(txt_path, 'w', encoding='utf-8') as f:
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

        # 保存为JSON文件
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

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        # 保存可视化结果
        self._save_visualization(image_path, image, results)

    def _save_visualization(self, image_path: str, image: any, results: List[OCRResult]):
        """保存可视化结果"""
        if len(results) == 0:
            return

        display_frame = self.ocr_core.draw_results(image.copy(), results)

        img_name = Path(image_path).stem
        vis_path = os.path.join(self.config.output_dir, f"{img_name}_visualized.jpg")
        cv2.imwrite(vis_path, display_frame)
        logger.info(f"可视化结果已保存: {vis_path}")
        