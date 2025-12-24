# 入门指南

本指南覆盖安装环境、训练与推理的基础使用。

## 环境要求与安装

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

## 训练模块（详细）

本项目提供三种训练方式（一个集成入口和四个辅助脚本）：

- 使用 `train.py` 进行统一训练：支持 YOLOv8 检测、YOLOv8 分类、ResNet50 二分类
- 使用 `scripts/` 下脚本：检测训练、由检测生成分类数据集、检测模型做图像级二分类评测、导出 ONNX

### 使用 train.py（统一入口）

1) YOLOv8 检测训练

```sh
python train.py \
  --use_yolov8_detect \
  --detect_data configs/detect.yaml \
  --detect_model yolov8n.pt \
  --detect_epochs 100 \
  --detect_imgsz 640 \
  --batch_size 16 \
  --yolo_device 0 \
  --workers 4 \
  --detect_project runs/detect \
  --detect_name train_crack
```

- 训练模型：YOLOv8 检测（如 `yolov8n.pt`、`yolov8s.pt`）
- 数据配置：`configs/detect.yaml`（YOLOv8 检测格式）
- 输出与保存：训练日志在 `runs/detect/...`；自动复制最佳权重到 `checkpoints/best_yolov8_detect.pt`

2) YOLOv8 分类训练（图像级二分类）

```sh
python train.py \
  --use_yolov8 \
  --data_dir data/classify \
  --yolo_model yolov8n-cls.pt \
  --epochs 30 \
  --imgsz 224 \
  --batch_size 32 \
  --yolo_device 0 \
  --workers 4
```

- 训练模型：YOLOv8 分类（如 `yolov8n-cls.pt`）
- 数据格式：`data/classify/{train,val}/{crack,no_crack}/*.jpg`
- 输出与保存：训练日志在 `runs/classify/...`；自动复制最佳权重到 `checkpoints/best_yolov8.pt`

3) ResNet50 二分类训练

```sh
python train.py \
  --data_dir data/classify \
  --epochs 30 \
  --batch_size 32 \
  --learning_rate 0.001
```

- 训练模型：`CrackDetector`（基于 ResNet50 的二分类：有裂缝/无裂缝）
- 数据格式：同上 `data/classify/{train,val}/{crack,no_crack}/*.jpg`
- 输出与保存：最佳模型 `checkpoints/best_model.pth`，最终模型 `checkpoints/final_model.pth`

参数要点：
- `--data_dir` 指向分类数据根目录；YOLOv8 检测使用 `--detect_data` 指向 `detect.yaml`
- `--yolo_device` 设为 GPU 编号（如 `0`）或 `cpu`；`--workers` 为数据加载并行度
- `--detect_project`/`--detect_name` 控制 YOLOv8 检测训练的日志目录与任务名

### 使用 scripts/ 下的四个训练相关脚本

1) `scripts/train_detect.py` — YOLOv8 检测训练

```sh
python scripts/train_detect.py \
  --data configs/detect.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --name train_crack \
  --device 0
```

- 训练模型：YOLOv8 检测（`yolov8n.pt`/`yolov8s.pt` 等）
- 数据配置：`configs/detect.yaml`
- 输出与保存：日志在 `runs/detect/<name>/...`；自动复制最佳权重到 `checkpoints/best_yolov8_detect.pt`

2) `scripts/convert_detect_to_cls.py` — 根据检测标注生成二分类数据集

```sh
python scripts/convert_detect_to_cls.py \
  --data configs/detect.yaml \
  --out data/classify_derived
```

- 功能：读取 `detect.yaml` 指定的 `train/images` 与 `val/images`，按照是否存在检测标注生成二分类目录结构
- 输出结构：`data/classify_derived/{train,val}/{crack,no_crack}/*.jpg`
- 适用：可直接用作 `train.py` 的分类训练数据源

3) `scripts/evaluate_detect_as_cls.py` — 用检测模型做图像级二分类评测

```sh
python scripts/evaluate_detect_as_cls.py \
  --weights checkpoints/best_yolov8_detect.pt \
  --data configs/detect.yaml \
  --device 0 \
  --conf 0.25
```

- 功能：对 `val/images` 中每张图推理，按是否检测到框视为裂缝/无裂缝，计算 Acc/Precision/Recall/F1
- 输入：检测权重 `--weights`，数据配置 `--data`

4) `scripts/export_onnx.py` — 导出 YOLOv8 检测为 ONNX

```sh
python scripts/export_onnx.py \
  --weights checkpoints/best_yolov8_detect.pt \
  --out checkpoints/best_yolov8_detect.onnx \
  --imgsz 640 \
  --opset 12
```

- 功能：将检测模型导出为 ONNX，以便后续部署与推理加速
- 输出：尝试将 Ultralytics 生成的 `best.onnx` 移动到指定 `--out`

依赖提示：上述与 YOLOv8 相关的训练/评测/导出需安装 `ultralytics`。

## 推理（检测）

```sh
python detect.py --source image.jpg --weights checkpoints/best_yolov8_detect.pt --conf 0.5
python detect.py --source video.mp4 --weights checkpoints/best_yolov8_detect.pt --conf 0.5
```

## 评测（检测模型做图像级二分类）

```sh
python scripts/evaluate_detect_as_cls.py --weights checkpoints/best_yolov8_detect.pt --data configs/detect.yaml
```

## 分割（训练与推理）

### 训练（YOLOv8 分割）
```sh
python scripts/train_seg.py --data configs/crack_seg_official.yaml --model yolov8n-seg.pt --epochs 50 --imgsz 640
```
- 数据路径默认使用 `data/crack-seg`（见 `configs/crack_seg_official.yaml`）
- 训练完成自动复制最佳权重到 `checkpoints/best_yolov8_seg.pt`

### 推理（掩膜显示）
```sh
# 使用训练好的分割权重
python detect.py --source path/to.jpg --weights checkpoints/best_yolov8_seg.pt --task segment
```
- 结果保存在 `runs/segment/...`，包含掩膜叠加可视化图像

## GUI界面（可视化操作）
在这个界面可以进行裂缝二分类、检测与分割的操作
![GUI.png](img%2FGUI.png)
### 使用说明
- 选择“任务模式”：分类/检测/分割
- 加载图像后：
  - 分类：点击“检测裂缝”（显示概率与预测）
  - 检测：点击“检测裂缝”（显示框与置信度）
  - 分割：点击“分割识别”（显示掩膜叠加）

## 数据集配置

示例见 `configs/detect.yaml`；Roboflow 导出的 YOLOv8 格式可直接使用。
