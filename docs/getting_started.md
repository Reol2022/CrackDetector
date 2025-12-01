# 入门指南

本指南覆盖安装环境、训练与推理的基础使用。

## 安装

- Python 3.8+
- 推荐使用虚拟环境（conda 或 venv）
- 安装依赖：

```sh
pip install -r requirements.txt
pip install ultralytics
```

## 训练（YOLOv8 检测）

使用集成入口 `train.py`：

```sh
python train.py \
  --use_yolov8_detect \
  --detect_data configs/detect.yaml \
  --detect_model yolov8n.pt \
  --detect_epochs 100 \
  --detect_imgsz 640 \
  --detect_project runs/detect \
  --detect_name train_crack
```

或使用脚本封装：

```sh
python scripts/train_detect.py --data configs/detect.yaml --model yolov8n.pt --epochs 100
```

训练结束后最佳权重会复制到 `checkpoints/best_yolov8_detect.pt`。

## 推理（检测）

```sh
python detect.py --source image.jpg --weights checkpoints/best_yolov8_detect.pt --conf 0.5
python detect.py --source video.mp4 --weights checkpoints/best_yolov8_detect.pt --conf 0.5
```

## 评测（检测模型做图像级二分类）

```sh
python scripts/evaluate_detect_as_cls.py --weights checkpoints/best_yolov8_detect.pt --data configs/detect.yaml
```

## 数据集配置

示例见 `configs/detect.yaml`；Roboflow 导出的 YOLOv8 格式可直接使用。

