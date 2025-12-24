import torch
import torch.nn as nn
from torchvision import models
from .resnet import CrackDetector
from .yolo_detect import YOLOv8Detector
from .yolo_seg import YOLOv8Segmenter

"""
模型工厂：支持 ResNet50 分类与 YOLOv8 分类
- ResNet50：用于二分类（有裂缝/无裂缝）
- YOLOv8 分类：使用 ultralytics 的 YOLO 分类权重进行推理

后续可扩展：YOLOv8 检测/分割（位置识别与掩膜），可在此文件提供统一封装
"""

"""
将 ResNet50 分类与 YOLOv8 检测拆分到独立文件：
- models/resnet.py: CrackDetector
- models/yolo_detect.py: YOLOv8Detector
本文件保留工厂与 YOLOv8 分类/分割封装，保持向后兼容。
"""

class YOLOv8Classifier:
    """YOLOv8 分类模型封装，用于推理。
    提供 predict(image) 接口，返回 (top1_index, probabilities_array)。
    """
    def __init__(self, weights_path=None, device=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics") from e

        self.device = device
        self.weights_path = weights_path or 'checkpoints/best_yolov8.pt'
        # 使用 Ultralytics 的模型加载权重
        self.model = YOLO(self.weights_path)

    # 兼容 nn.Module 常用调用
    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def predict(self, image):
        """对单张图像进行分类预测。
        image: PIL.Image 或 numpy.ndarray 或路径字符串
        返回: (pred_index, probabilities_array)
        """
        results = self.model.predict(image, device=self.device, verbose=False)
        r = results[0]
        # 分类概率
        if hasattr(r, 'probs') and hasattr(r.probs, 'data'):
            probs = r.probs.data.cpu().numpy()
            top1 = int(r.probs.top1)
            return top1, probs
        else:
            # 兜底：若结果中不含 probs（理论上不会发生）
            return 0, None


# YOLOv8Detector 已迁移至 models/yolo_detect.py
# YOLOv8Segmenter 已迁移至 models/yolo_seg.py

def get_model(model_name='CrackDetector', model_type='classification', num_classes=2, pretrained=True, **kwargs):
    """通用模型工厂
    参数：
    - model_name: 'CrackDetector'（ResNet50分类）或 'yolov8'（YOLOv8分类）
    - model_type: 当前支持 'classification'（保留参数以兼容旧代码）
    - num_classes: 分类类别数（ResNet50使用；YOLOv8分类由权重决定，此处仅保留用于接口统一）
    - pretrained: ResNet50是否加载本地预训练
    - kwargs: 其他参数，如 weights（YOLOv8分类权重路径）、device 等
    """
    # 兼容旧调用：若只传了 model_type=classification，则默认返回 ResNet50
    if model_name is None and model_type == 'classification':
        model_name = 'CrackDetector'

    if model_name == 'CrackDetector':
        return CrackDetector(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() in ('yolov8', 'yolov8-cls', 'yolov8_classifier'):
        weights = kwargs.get('weights')
        device = kwargs.get('device')
        return YOLOv8Classifier(weights_path=weights, device=device)
    elif model_name.lower() in ('yolov8-detect', 'yolov8_detector', 'yolov8-det'):
        weights = kwargs.get('weights')
        device = kwargs.get('device')
        return YOLOv8Detector(weights_path=weights, device=device)
    elif model_name.lower() in ('yolov8-seg', 'yolov8_segmenter', 'yolov8-segmentation'):
        weights = kwargs.get('weights')
        device = kwargs.get('device')
        return YOLOv8Segmenter(weights_path=weights, device=device)
    else:
        raise ValueError(f"模型 '{model_name}' 不支持")
