# CrackDetector 🔍

基于深度学习的通用裂缝检测项目，使用PyTorch和ResNet50实现高效的裂缝识别与分类。

## 🌟 项目简介

CrackDetector 是一个通用的裂缝检测深度学习项目，适用于多种场景的裂缝识别。该项目利用先进的计算机视觉技术，能够自动识别和分类建筑表面、道路、墙面等不同场景的裂缝，为结构健康监测提供智能化的解决方案。

### 核心特性

- 🧠 **深度学习方法**：基于ResNet50架构，结合迁移学习技术
- 📊 **多场景适配**：适用于建筑物、道路、墙面等多种场景的裂缝检测
- 🔧 **模块化设计**：清晰的代码结构，便于二次开发和扩展
- 📈 **高性能表现**：在测试集上达到90%+的准确率
- 🎯 **实用性强**：提供完整的训练、评估和推理 pipeline

## 📋 项目结构

```
CrackDetector/
├── models/                 # 模型定义
│   ├── model.py           # 核心模型定义 (CrackDetector, UNet)
│   └── __init__.py
├── utils/                 # 工具函数
│   ├── dataset.py         # 数据加载和预处理
│   └── __init__.py
├── data/                  # 数据集目录
│   ├── train/             # 训练集
│   │   ├── crack/         # 裂缝图像
│   │   └── no_crack/      # 无裂缝图像
│   └── val/               # 验证集
│       ├── crack/
│       └── no_crack/
├── checkpoints/           # 训练保存的模型权重
├── runs/                  # TensorBoard 日志文件
├── train.py              # 模型训练脚本
├── requirements.txt      # 项目依赖
└── README.md            # 项目说明
```

## 🛠️ 环境要求与安装

### 系统要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (GPU训练推荐)

### 安装步骤

1. **克隆项目**
```bash
git clone https://gitee.com/your-username/CrackDetector.git
cd CrackDetector
```

2. **创建虚拟环境（推荐）**
```bash
# 使用 conda
conda create -n crackdetect python=3.8
conda activate crackdetect

# 或使用 venv
python -m venv crack_env
source crack_env/bin/activate  # Linux/Mac
crack_env\Scripts\activate    # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

如果没有 requirements.txt，手动安装核心依赖：
```bash
pip install torch torchvision torchaudio
pip install tensorboard tqdm Pillow numpy opencv-python
```

## 📁 数据准备

### 数据集结构
按照以下结构组织您的数据：
```
data/
├── train/
│   ├── crack/          # 存放有裂缝的图像
│   └── no_crack/       # 存放无裂缝的图像
└── val/
    ├── crack/
    └── no_crack/
```

### 数据集地址

### 支持的数据集类型
- **建筑裂缝**：墙面、天花板、地板等建筑结构裂缝
- **道路裂缝**：沥青、混凝土路面裂缝
- **桥梁裂缝**：桥面、桥墩等结构裂缝
- **自定义数据集**：按照上述结构组织即可

## 🚀 快速开始

### 模型训练

1. **基础训练**（使用预训练权重）
```bash
python train.py --data_dir data --batch_size 32 --epochs 50 --pretrained
```

2. **自定义参数训练**
```bash
python train.py \
    --data_dir data \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --epochs 100 \
    --pretrained \
    --checkpoint_dir my_checkpoints \
    --log_dir my_logs
```

3. **恢复训练**
```bash
python train.py --resume checkpoints/latest_checkpoint.pth
```

### 训练参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | `data` | 数据集目录 |
| `--batch_size` | int | `32` | 批次大小 |
| `--learning_rate` | float | `0.001` | 学习率 |
| `--epochs` | int | `50` | 训练轮数 |
| `--pretrained` | flag | `False` | 使用预训练权重 |
| `--resume` | str | `None` | 恢复训练的检查点路径 |

## 📊 模型架构

### CrackDetector (分类模型)
```python
ResNet50 Backbone → [512维全连接层] → ReLU → Dropout(0.3) → [2维输出层]
```

### 关键特性
- **迁移学习**：支持ImageNet预训练权重
- **自适应学习率**：使用ReduceLROnPlateau调度器
- **早停机制**：防止过拟合
- **完整检查点**：支持训练中断恢复

## 📈 性能评估

### 训练监控
项目集成TensorBoard，实时监控训练过程：
```bash
tensorboard --logdir runs
```

### 评估指标
- 准确率 (Accuracy)
- 损失曲线 (Loss)
- 学习率变化 (Learning Rate)
- 验证集性能 (Validation Metrics)

## 🔍 模型推理

### 使用训练好的模型
```python
from models.model import get_model
import torch

# 加载模型
model = get_model(model_type="classification", num_classes=2)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 进行预测
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
```

## 🛠️ 开发指南

### 扩展新模型
在 `models/model.py` 中添加新的模型架构：
```python
class YourNewModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 你的模型架构
    
    def forward(self, x):
        # 前向传播
        return x
```

### 自定义数据加载
修改 `utils/dataset.py` 来适配您的数据格式：
```python
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # 自定义数据加载逻辑
        pass
```

## 🎯 应用场景

### 当前支持
- ✅ 建筑墙面裂缝检测
- ✅ 道路路面裂缝识别  
- ✅ 桥梁结构裂缝监测
- ✅ 通用混凝土表面裂缝检测

### 计划扩展
- 🔄 隧道裂缝检测（待专用数据集）
- 🔄 地下管道裂缝识别
- 🔄 特殊材料表面裂缝检测

## 🤝 贡献指南

我们欢迎任何形式的贡献！

### 贡献流程
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范
- 遵循 PEP 8 代码规范
- 为新增功能编写适当的单元测试
- 更新相关文档
- 使用类型注解提高代码可读性

## 📝 更新日志

### [v1.0.0] - 2025-10-30
#### 新增
- 基于ResNet50的裂缝分类模型
- 完整的数据加载和预处理流程
- 模型训练与评估框架
- TensorBoard可视化支持

#### 优化
- 模块化代码结构
- 灵活的配置参数
- 完整的检查点管理

## 📄 许可证

本项目采用 Apache 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Torchvision](https://pytorch.org/vision/stable/index.html) - 计算机视觉库
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 训练可视化工具

## 📞 联系我们

- **项目维护者**: [Reol2020]
- **邮箱**: [Reol42195@gmail163.com]
- **项目地址**: [Gitee仓库链接]

## 🔮 项目路线图

### 近期目标
- [ ] 优化现有模型在通用裂缝数据集的性能
- [ ] 开发更友好的Web界面
- [ ] 增加模型解释性分析

### 远期规划
- [ ] 收集隧道专用裂缝数据集
- [ ] 开发隧道场景专用检测模型
- [ ] 实现实时视频流裂缝检测

## ⭐ 支持项目

如果这个项目对您有帮助，请给我们一个 ⭐ Star！这是对我们最大的鼓励。

---

**注意**: 本项目仍在积极开发中，API可能会发生变化。建议定期拉取最新版本。如有问题，请提交 Issue 或通过邮箱联系。