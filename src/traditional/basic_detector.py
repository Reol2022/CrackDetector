# src/traditional/basic_detector.py
import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


class TraditionalCrackDetector:
    def __init__(self):
        self.min_contour_area = 100

    def load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 直方图均衡化增强对比度
        equalized = cv2.equalizeHist(blurred)

        return equalized

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """边缘检测"""
        # Canny边缘检测
        edges = cv2.Canny(image, 50, 150)

        # 形态学闭运算连接断裂边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return closed

    def find_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """查找轮廓"""
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 过滤小轮廓
        filtered_contours = [
            cnt for cnt in contours
            if cv2.contourArea(cnt) > self.min_contour_area
        ]

        return filtered_contours

    def analyze_contours(self, contours: List[np.ndarray]) -> dict:
        """分析轮廓特征"""
        analysis = {
            'total_cracks': len(contours),
            'areas': [],
            'perimeters': [],
            'lengths': []
        }

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # 估算长度（通过边界矩形）
            rect = cv2.minAreaRect(contour)
            length = max(rect[1])

            analysis['areas'].append(area)
            analysis['perimeters'].append(perimeter)
            analysis['lengths'].append(length)

        return analysis

    def visualize_result(self, original_image: np.ndarray,
                         contours: List[np.ndarray],
                         save_path: str = None):
        """可视化结果"""
        # 绘制轮廓
        result_image = original_image.copy()
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

        # 显示结果
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        edges = self.detect_edges(self.preprocess(original_image))
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Cracks')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    def process(self, image_path: str) -> dict:
        """完整处理流程"""
        # 1. 加载图像
        image = self.load_image(image_path)

        # 2. 预处理
        processed = self.preprocess(image)

        # 3. 边缘检测
        edges = self.detect_edges(processed)

        # 4. 轮廓检测
        contours = self.find_contours(edges)

        # 5. 分析结果
        analysis = self.analyze_contours(contours)

        # 6. 可视化
        self.visualize_result(image, contours)

        return {
            'image_path': image_path,
            'contours': contours,
            'analysis': analysis,
            'total_cracks_detected': len(contours)
        }