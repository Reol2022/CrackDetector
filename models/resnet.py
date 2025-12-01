import torch
import torch.nn as nn
from torchvision import models

class CrackDetector(nn.Module):
    """
    ResNet50 二分类模型（有裂缝 / 无裂缝）。
    与原 models/model.py 中的 CrackDetector 等价，便于拆分与复用。
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(CrackDetector, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)

        if pretrained:
            try:
                self.resnet50.load_state_dict(torch.load('checkpoints/resnet50-0676ba61.pth'), strict=False)
                print("成功加载本地预训练权重: checkpoints/resnet50-0676ba61.pth")
            except FileNotFoundError:
                print("警告: 未找到本地预训练权重 'checkpoints/resnet50-0676ba61.pth'。模型将从头开始训练。")

        for param in self.resnet50.parameters():
            param.requires_grad = False

        for param in self.resnet50.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)

        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet50(x)

