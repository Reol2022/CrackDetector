import torch
import torch.nn as nn
from torchvision import models

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

def get_model(model_name, num_classes=2, pretrained=True, **kwargs):
    if model_name == 'CrackDetector':
        # 忽略kwargs, 因为这个版本不使用它们
        return CrackDetector(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"模型 '{model_name}' 不支持")