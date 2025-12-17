# pip 包构建与使用指南

面向发布与使用：将项目打包为可安装的 Python 包（pip 安装），提供 API 与命令行工具（CLI）。

## 安装与使用

- 安装（本地源码）：
  - 开发模式：`pip install -e .`
  - 普通安装：`pip install .`

- 安装（发布后）：
  - 从 PyPI/TestPyPI：`pip install crackdetector`
  - 注意：分类/检测依赖 `torch`，若未自动安装，请先安装对应平台的 PyTorch。

### CLI 使用

- 检测：`crackdetector-detect --source path/to/image_or_dir --weights checkpoints/best_yolov8_detect.pt`
- 分类：`crackdetector-classify --source path/to/image_or_dir --weights checkpoints/best_model.pth`

参数说明：
- `--device`（可选）：如 `0`、`cpu`
- `--conf`、`--iou`、`--imgsz`（检测）
- 默认输出在 `runs/detect/<name>`，自动保存可视化结果与（可选）YOLO 文本格式

### API 使用

```python
from crackdetector.api import detect, classify

# YOLOv8 检测
detect_results = detect(
    source="images/", weights="checkpoints/best_yolov8_detect.pt", conf=0.25
)

# ResNet50 二分类
cls_results = classify(
    source="images/", weights="checkpoints/best_model.pth"
)
for path, p0, p1, pred in cls_results:
    print(path, p0, p1, pred)
```


## 依赖与数据文件

- 运行时核心依赖：`ultralytics`、`opencv-python`、`numpy`、`pillow`（已在 `pyproject.toml`）
- 分类与训练相关：`torch`、`torchvision`（体积较大，可能需手动安装）
- 权重与配置：不随包分发。请自行放置在项目或运行目录，例如 `checkpoints/` 与 `configs/`。

## 常见问题（pip 使用场景）

- `torch` 未安装或版本不匹配：请先安装与平台匹配的 PyTorch
- 权重路径不存在：传入 `--weights` 或将权重置于 `checkpoints/` 下
- 路径包含空格/中文：建议使用引号包裹路径或切换到 ASCII 路径
- GPU 不可用：指定 `--device cpu` 或安装/配置 CUDA

更多问题与解答见根目录 `question.md`。
