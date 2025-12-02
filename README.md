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

## 📚 文档索引

- 入门指南（安装、训练、推理）：[docs/getting_started.md](docs/getting_started.md)
- 数据与标注（YOLOv8、Roboflow、分割预留）：[docs/datasets.md](docs/datasets.md)
- 基准与评测（指标、对比与方法）：[docs/benchmark.md](docs/benchmark.md)
- 桌面版打包与资源路径：[docs/desktop.md](docs/desktop.md)
- 贡献规范与分支策略：[docs/contributing.md](docs/contributing.md)
- 版本更新历史：[docs/changelog.md](docs/changelog.md)

## 📂 项目信息架构与结构

CrackDetector/
├── models/                          # 模型定义
│   ├── model.py                     # 模型工厂与YOLOv8分类/分割封装
│   ├── resnet.py                    # ResNet50二分类（拆分）
│   ├── yolo_detect.py               # YOLOv8检测（拆分）
│   └── __init__.py
├── configs/                         # 任务配置
│   ├── detect.yaml                  # YOLOv8检测数据配置（Roboflow/YOLOv8）
│   ├── classify.yaml                # 分类训练参数模板
│   └── seg.yaml                     # 分割任务模板（预留）
├── scripts/                         # 数据/评测/导出与训练脚本
│   ├── convert_detect_to_cls.py     # 检测标注派生分类数据
│   ├── evaluate_detect_as_cls.py    # 检测模型用于图像级分类评测
│   ├── export_onnx.py               # 导出ONNX
│   └── train_detect.py              # YOLOv8检测训练（可选）
├── data/                            # 原始/示例数据集目录
├── checkpoints/                     # 训练保存的模型权重
├── runs/                            # 训练日志与结果
├── docs/                            # 独立文档（安装、数据、基准、桌面版）
├── train.py                         # 集成训练入口（含检测与分类）
├── detect.py                        # 裂缝检测推理脚本
├── requirements.txt                 # 项目依赖
└── README.md                        # 项目说明（面向用户路径）


## 🚀 快速开始

### YOLOv8检测训练（推荐）

使用 `train.py` 直接进行检测训练，无需 `train_yolo.py`：

```sh
# GPU（自动选择可用设备）
python train.py \
  --use_yolov8_detect \
  --detect_data configs/detect.yaml \
  --detect_model yolov8n.pt \
  --detect_epochs 100 \
  --detect_imgsz 640 \
  --detect_project runs/detect \
  --detect_name crack_yolov8

# CPU
python train.py \
  --use_yolov8_detect \
  --detect_data configs/detect.yaml \
  --detect_model yolov8n.pt \
  --detect_epochs 100 \
  --detect_imgsz 640
```

- 训练完成后：最佳权重会自动复制到 `checkpoints/best_yolov8_detect.pt`
- 可选：使用 `scripts/train_detect.py --data configs/detect.yaml --model yolov8n.pt` 启动检测训练
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

## 🧩 模型与对比（概览）

- 检测：YOLOv8（定位与边界框）。
- 分类：ResNet50（有/无裂缝判断）。
- 详细架构与对比请见 `docs/benchmark.md`。

## 📈 性能评估

### 训练监控

```sh
tensorboard --logdir runs
```

### 评估指标

- **YOLOv8**: mAP@0.5, mAP@0.5:0.95, 精确率, 召回率
- **ResNet50**: 准确率, F1-score, 混淆矩阵
- 损失曲线, 学习率变化

## 🔍 模型推理（简版）

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

## 🛠️ 开发指南（简版）

### 扩展新模型

开发细则与分支策略请参阅 `CONTRIBUTING.md`。

## 🎯 应用场景（概览）

- 建筑、道路、桥梁裂缝检测（YOLOv8）
- 快速裂缝存在性判断（ResNet50）

## 🤝 贡献指南

我们欢迎任何形式的贡献！详细规范见 `CONTRIBUTING.md`。版本更新历史请参阅 `CHANGELOG.md`。

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

完整的版本更新记录请查看 `changelog.md`。

##  许可证

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
- **当前主要分支**: `detect`

## ⭐ 支持项目

如果这个项目对您有帮助，请给我们一个 ⭐ Star！这是对我们最大的鼓励。

---

**注意**: 项目当前主要开发在 `detect` 分支，YOLOv8版本仍在积极开发中。ResNet50版本保持在 `master` 分支作为稳定版本。


## 🖼️ 结果示例

- 检测结果示例：`runs/detect/train_crack/val_batch0_pred.jpg`
- 训练批次示例：`runs/detect/train_crack/train_batch0.jpg`
- 视频帧示例：通过 `detect.py --source video.mp4` 推理后手工保存帧图（示例帧可在脚本中使用 OpenCV 保存）

## ❓ 常见问题

- 未安装 Ultralytics：执行 `pip install ultralytics`
- 权重路径不一致：训练后权重复制到 `checkpoints/best_yolov8_detect.pt`
- detect.yaml 路径：请使用 `configs/detect.yaml`
- GPU 不可用：传入 `--device cpu` 或确保 CUDA 驱动正确

<!-- 仓库操作建议已合并至 CONTRIBUTING.md，避免主 README 过长 -->
