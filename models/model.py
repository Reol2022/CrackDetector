import torch
import torch.nn as nn
import torchvision.models as models

class CrackDetector(nn.Module):
    """
    裂缝检测模型，基于预训练的ResNet50
    """
    def __init__(self, num_classes=2):
        super(CrackDetector, self).__init__()
        # 使用预训练的ResNet50作为特征提取器
        self.resnet = models.resnet50(pretrained=True)
        
        # 冻结部分层以加快训练
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
            
        # 替换最后的全连接层
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)


class UNet(nn.Module):
    """
    UNet模型用于裂缝分割
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        
        # 下采样路径
        self.inc = self._double_conv(n_channels, 64)
        self.down1 = self._down(64, 128)
        self.down2 = self._down(128, 256)
        self.down3 = self._down(256, 512)
        self.down4 = self._down(512, 512)
        
        # 上采样路径
        self.up1 = self._up(1024, 256)
        self.up2 = self._up(512, 128)
        self.up3 = self._up(256, 64)
        self.up4 = self._up(128, 64)
        
        # 输出层
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 下采样路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 上采样路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        x = self.outc(x)
        x = self.sigmoid(x)
        
        return x
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._double_conv(in_channels, out_channels)
        )
    
    def _up(self, in_channels, out_channels):
        return nn.Module()  # 占位，下面会重写
    
    def _up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2),
            self._double_conv(in_channels, out_channels)
        )


def get_model(model_type="classification", **kwargs):
    """
    获取模型实例
    
    Args:
        model_type: 模型类型，可选 "classification" 或 "segmentation"
        
    Returns:
        模型实例
    """
    if model_type == "classification":
        return CrackDetector(**kwargs)
    elif model_type == "segmentation":
        return UNet(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")