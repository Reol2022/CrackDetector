# CrackDetector 🔍

基于深度学习的通用裂缝检测项目，支持多种深度学习模型实现高效的裂缝识别与检测。

## 🌟 项目简介

CrackDetector 是一个通用的裂缝检测深度学习项目，适用于多种场景的裂缝识别。该项目提供多种先进的计算机视觉模型，能够自动识别和检测建筑表面、道路、墙面等不同场景的裂缝，为结构健康监测提供智能化的解决方案。

### 核心特性

- 🧠 **多模型支持**: 支持YOLOv8目标检测和ResNet50分类两种架构
- 🎯 **目标检测能力**: YOLOv8实现像素级裂缝定位和边界框检测
- 📊 **多场景适配**: 适用于建筑物、道路、墙面等多种场景的裂缝检测
- 🔧 **模块化设计**: 清晰的代码结构，便于二次开发和扩展
- 📈 **高性能表现**: YOLOv8在检测任务上达到优异性能
- 🚀 **端到端流程**: 提供完整的训练、评估和推理pipeline

## 📋 项目结构

CrackDetector/
├── models/                         # 模型定义  
│   ├── model.py                  # ResNet50分类模型  
│   ├── yolo_detector.py      # YOLOv8检测模型  
│   └── __init__.py  
├── utils/                              # 工具函数  
│   ├── dataset.py                # 数据加载和预处理  
│   ├── yolo_utils.py             # YOLO专用工具函数  
│   └── __init__.py  
├── data/                              # 数据集目录  
│   ├── images/                    # 图像文件  
│   │   ├── train/  
│   │   └── val/  
│   └── labels/                      # YOLO格式标注文件  
│       ├── train/  
│       └── val/  
├── checkpoints/                 # 训练保存的模型权重  
├── runs/                             # TensorBoard日志文件  
├── train.py                         # ResNet50训练脚本  
├── train_yolo.py                # YOLOv8训练脚本  
├── detect.py                      # 裂缝检测推理脚本  
├── requirements.txt          # 项目依赖  
└── README.md                # 项目说明  

## 🛠️ 环境要求与安装

### 系统要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (GPU训练推荐)
- Ultralytics YOLOv8

### 安装步骤

1. **克隆项目**

```sh
git clone https://gitee.com/Reol2022/SmartTunnel-CrackDetector
cd CrackDetector

# 切换到YOLOv8分支
git checkout yolov8
```

2. **创建虚拟环境（推荐）**

```sh
# 使用conda
conda create -n crackdetect python=3.8
conda activate crackdetect

# 或使用venv
python -m venv crack_env
source crack_env/bin/activate  # Linux/Mac
crack_env\Scripts\activate    # Windows
```

3. **安装依赖**

```sh
pip install -r requirements.txt

# 安装YOLOv8
pip install ultralytics

# 如果没有requirements.txt，手动安装核心依赖
pip install torch torchvision torchaudio
pip install ultralytics tensorboard tqdm Pillow numpy opencv-python
```

## 📁 数据准备

### YOLOv8数据集格式

```
data/
├── images/
│   ├── train/           # 训练图像
│   └── val/             # 验证图像
└── labels/
    ├── train/           # 训练标注文件 (.txt)
    └── val/             # 验证标注文件 (.txt)
```

### YOLO标注格式

每个标注文件对应一个图像，包含：

```
<class_id> <x_center> <y_center> <width> <height>
```

- 坐标值都是相对于图像宽高的归一化值(0-1)
- class_id: 0表示裂缝

### 数据集地址

- **SDNET2018**: 道路和墙面裂缝数据集
地址： https://digitalcommons.usu.edu/all_datasets/48/
- **Crack Detection.v2-v2.yolov8**: 带标注的裂缝数据集
  
  地址：https://universe.roboflow.com/antonio-raimundo/crack-detection-y5kyg/dataset/2

### Roboflow 数据集接入（detect.yaml 示例）

使用 Roboflow 导出的 YOLOv8 数据集时，推荐在 `data/detect.yaml` 中配置根路径与子目录：

```yaml
path: "e:/CrackDetector/data/Crack Detection.v2-v2.yolov8"
train: train/images
val: valid/images
test: test/images

nc: 1
names: ["crack"]
```

- 训练命令中的 `--detect_data data/detect.yaml` 会读取上述配置
- 如没有 `test` 集，可删除该行；`train/val` 即可完成训练与验证

## 🚀 快速开始

### YOLOv8检测训练（推荐）

使用 `train.py` 直接进行检测训练，无需 `train_yolo.py`：

```sh
# GPU（自动选择可用设备）
python train.py \
  --use_yolov8_detect \
  --detect_data data/detect.yaml \
  --detect_model yolov8n.pt \
  --detect_epochs 100 \
  --detect_imgsz 640 \
  --detect_project runs/detect \
  --detect_name crack_yolov8

# CPU
python train.py \
  --use_yolov8_detect \
  --detect_data data/detect.yaml \
  --detect_model yolov8n.pt \
  --detect_epochs 100 \
  --detect_imgsz 640
```

- 训练完成后：最佳权重会自动复制到 `checkpoints/best_yolov8_detect.pt`
- 原有分类训练入口保持不变：`python train.py --data_dir data --batch_size 32 --epochs 50 --pretrained`

### ResNet50分类训练（传统方法）

```sh
python train.py --data_dir data --batch_size 32 --epochs 50 --pretrained
```

### 裂缝检测推理

1. **图像检测**

```sh
python detect.py --source image.jpg --weights checkpoints/best_yolov8_detect.pt --conf 0.5
```

2. **视频检测**

```sh
python detect.py --source video.mp4 --weights checkpoints/best_yolov8_detect.pt --conf 0.5
```

3. **实时摄像头检测**

```sh
python detect.py --source 0 --weights checkpoints/best_yolov8_detect.pt --conf 0.5
```

## 📊 模型架构

### YOLOv8检测模型

```python
Backbone: CSPDarknet → Neck: PAN-FPN → Head: Classifier + BBox Regressor
```

### ResNet50分类模型（传统）

```python
ResNet50 Backbone → [512维全连接层] → ReLU → Dropout(0.3) → [2维输出层]
```

### 模型对比

|特性|YOLOv8|ResNet50|
|--|--|--|
|任务类型|目标检测|图像分类|
|输出|边界框+置信度|分类概率|
|优势|精确定位、多目标|简单快速、二分类|
|适用场景|需要定位裂缝位置|只需判断有无裂缝|

## 📈 性能评估

### 训练监控

```sh
# 启动TensorBoard
tensorboard --logdir yolov8

# 查看训练结果
tensorboard --logdir runs
```

### 评估指标

- **YOLOv8**: mAP@0.5, mAP@0.5:0.95, 精确率, 召回率
- **ResNet50**: 准确率, F1-score, 混淆矩阵
- 损失曲线, 学习率变化

## 🔍 模型推理

### 使用YOLOv8模型

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('checkpoints/best_yolov8_detect.pt')

# 进行预测
results = model('image.jpg')

# 可视化结果
for r in results:
    im_array = r.plot()  # 绘制检测结果
    cv2.imwrite('result.jpg', im_array)
```

### 使用ResNet50模型

```python
from models.model import get_model
import torch

model = get_model(model_type="classification", num_classes=2)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 🛠️ 开发指南

### 扩展新模型

在 `models/` 目录中添加新的模型架构：

```python
# models/yolo_detector.py
from ultralytics import YOLO

class CrackYOLODetector:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
    
    def detect(self, image_path):
        return self.model(image_path)
```

### 自定义数据加载

修改 `utils/dataset.py` 支持不同数据格式：

```python
class YOLODataset:
    def __init__(self, data_yaml, augment=True):
        self.dataset = self.load_yolo_dataset(data_yaml)
    
    def load_yolo_dataset(self, data_yaml):
        # YOLO格式数据加载
        pass
```

## 🎯 应用场景

### 当前支持

- ✅ 建筑墙面裂缝检测与定位 (YOLOv8)
- ✅ 道路路面裂缝识别与边界框检测 (YOLOv8)  
- ✅ 桥梁结构裂缝监测 (YOLOv8)
- ✅ 快速裂缝存在性判断 (ResNet50)

## 🤝 贡献指南

我们欢迎任何形式的贡献！

### 分支策略

- `main`: 稳定版本（当前为ResNet50）
- `yolov8`: YOLOv8开发分支
- `feature/*`: 功能开发分支

### 贡献流程

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/AmazingFeature`
3. 提交更改: `git commit -m 'Add some AmazingFeature'`
4. 推送到分支: `git push origin feature/AmazingFeature`
5. 开启Pull Request

## 📝 更新日志

### [v2.1.0] - 2025-11-18 集成训练与兼容性修复

#### 新增

- ✅ 在 `train.py` 集成 YOLOv8 检测训练入口（无需单独脚本）
- ✅ 新增 `data/detect.yaml`，可直接接入 Roboflow YOLOv8 数据集
- ✅ 训练结束自动复制最佳权重到 `checkpoints/best_yolov8_detect.pt`

#### 修复

- 🛠 解决 ResNet50 推理时报错：`torch.cuda.FloatTensor` 与 `torch.FloatTensor` 设备不一致
（加载权重与输入张量统一到相同设备）

### [v2.0.0] - 2025-11-02 YOLOv8重大更新

#### 新增

- ✅ 集成YOLOv8目标检测模型
- ✅ 支持裂缝边界框检测和定位
- ✅ 新增YOLO格式数据集支持
- ✅ 实时检测和视频处理功能
- ✅ 完整的模型评估指标

#### 优化

- 🔄 模块化代码结构，支持多模型
- 🔄 改进的训练pipeline
- 🔄 增强的数据预处理
- 🔄 更好的可视化输出

### [v1.0.0] - 2025-10-21 初始化项目

### [v1.0.1] - 2025-10-25 添加数据集,更新了模型训练代码

### [v1.0.2] - 2025-10-29 更新了readme,修改模型以使用预训练权重,优化图像预处理步骤

## 📄 许可证

本项目采用 Apache 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 先进的目标检测框架
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 联系我们

- **项目维护者**: [Reol2020]
- **邮箱**: [Reol42195@gmail163.com]
- **项目地址**: [https://gitee.com/Reol2022/SmartTunnel-CrackDetector]
- **当前主要分支**: `yolov8`

## ⭐ 支持项目

如果这个项目对您有帮助，请给我们一个 ⭐ Star！这是对我们最大的鼓励。

---

**注意**: 项目当前主要开发在 `yolov8` 分支，YOLOv8版本仍在积极开发中。ResNet50版本保持在 `main` 分支作为稳定版本。

```

## 🎯 Gitee仓库操作建议

### 1. **创建新分支**
```bash
# 创建并切换到yolov8分支
git checkout -b yolov8

# 推送新分支到远程
git push -u origin yolov8
```

### 2. **分支说明**

- **main分支**: 保留ResNet50稳定版本
- **yolov8分支**: 新功能开发，包含YOLOv8实现
- **特性分支**: 从yolov8分支创建，用于特定功能开发

### 3. **版本管理建议**

在README中明确标注当前主要开发分支，方便用户选择：

```markdown
## 🎯 当前版本
- **稳定版**: `main` 分支 (ResNet50分类)
- **开发版**: `yolov8` 分支 (YOLOv8检测) ← 推荐使用
```
