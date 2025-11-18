import torch
import torch.nn as nn
from torchvision import models

"""
模型工厂：支持 ResNet50 分类与 YOLOv8 分类
- ResNet50：用于二分类（有裂缝/无裂缝）
- YOLOv8 分类：使用 ultralytics 的 YOLO 分类权重进行推理

后续可扩展：YOLOv8 检测/分割（位置识别与掩膜），可在此文件提供统一封装
"""

class CrackDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(CrackDetector, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False) # pretrained=False因为我们要加载本地权重
        
        # 尝试加载本地预训练权重
        if pretrained:
            try:
                self.resnet50.load_state_dict(torch.load('checkpoints/resnet50-0676ba61.pth'), strict=False)
                print("成功加载本地预训练权重: checkpoints/resnet50-0676ba61.pth")
            except FileNotFoundError:
                print("警告: 未找到本地预训练权重 'checkpoints/resnet50-0676ba61.pth'。模型将从头开始训练。")

        # 冻结所有层
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # 只解冻layer4和全连接层
        for param in self.resnet50.layer4.parameters():
            param.requires_grad = True

        # 替换全连接层以匹配我们的任务
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
        
        # 确保新的全连接层是可训练的
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet50(x)

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


class YOLOv8Detector:
    """YOLOv8 检测模型封装。
    predict(image) 返回 Ultralytics 的单图结果对象，包含 boxes 等。
    """
    def __init__(self, weights_path=None, device=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics") from e
        self.device = device
        self.weights_path = weights_path or 'yolov8n.pt'
        self.model = YOLO(self.weights_path)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def predict(self, image):
        results = self.model.predict(image, device=self.device, verbose=False)
        return results[0]


class YOLOv8Segmenter:
    """YOLOv8 分割模型封装。
    predict(image) 返回单图结果对象，包含 masks。
    """
    def __init__(self, weights_path=None, device=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics") from e
        self.device = device
        self.weights_path = weights_path or 'yolov8n-seg.pt'
        self.model = YOLO(self.weights_path)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def predict(self, image):
        results = self.model.predict(image, device=self.device, verbose=False)
        return results[0]


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