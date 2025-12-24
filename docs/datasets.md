# 数据集与标注

## 数据集地址

- **SDNET2018**: 道路和墙面裂缝数据集
地址： https://digitalcommons.usu.edu/all_datasets/48/
- **Crack Detection.v2-v2.yolov8**: 带标注的裂缝数据集
地址：https://universe.roboflow.com/antonio-raimundo/crack-detection-y5kyg/dataset/2
- **自定义数据集**: 按照YOLO格式组织即可

## YOLOv8 标注格式

每张图片对应一个同名 `.txt` 文件，内容为：

```
<class_id> <x_center> <y_center> <width> <height>
```

- 坐标为归一化到 [0, 1]
- `class_id` 为 0 表示裂缝

目录结构示例：

```
dataset_root/
├── train/images
├── train/labels
├── valid/images
└── valid/labels
```

## detect.yaml 配置

`configs/detect.yaml` 示例：

```yaml
path: "e:/.../Crack Detection.v2-v2.yolov8"
train: train/images
val: valid/images
test: test/images
nc: 1
names: ["crack"]
```

## Roboflow 接入

- 在 Roboflow 导出 YOLOv8 格式，下载并解压到本地
- 将 `path` 指向数据根目录，`train/val/test` 使用默认子目录即可

## 分割数据集 (Segmentation)

### 1. 官方推荐：Crack-seg (自动下载)
YOLOv8 官方直接集成了一个高质量的裂缝分割数据集 `crack-seg`。

- **获取方式**: 无需手动下载，运行训练命令时会自动下载。
- **配置**: 直接使用 `crack-seg.yaml` (Ultralytics 内置) 或本项目提供的 `configs/crack_seg_official.yaml`（默认路径 `data/crack-seg`）。
- **手动下载链接**: [https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip](https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip)
- **演示脚本**: 运行 `python scripts/download_and_train_seg.py` 即可体验。
- **训练命令**: `python scripts/train_seg.py --data configs/crack_seg_official.yaml --epochs 50`

### 2. Roboflow Universe
Roboflow 上有大量公开的裂缝分割数据集，支持一键导出为 YOLOv8 Seg 格式。
- **推荐数据集**: [Crack Segmentation Dataset on Roboflow](https://universe.roboflow.com/university-bswxt/crack-bphdr/2)
- **使用方法**:
  1. 注册 Roboflow 账号。
  2. 点击 "Download Dataset"。
  3. 选择格式 "YOLOv8 Oriented Bounding Boxes (OBB)" 或 "YOLOv8 Instance Segmentation"。
  4. 解压并配置 `configs/seg.yaml` 指向该目录。

## 分割标注格式 (YOLO Seg)

### 目录结构
```
data/crack-seg/
├─ images/
│  ├─ train
│  ├─ val
│  └─ test
└─ labels/
   ├─ train
   ├─ val
   └─ test
```

### 推理与展示
- 命令行推理：`python detect.py --source path/to.jpg --weights checkpoints/best_yolov8_seg.pt --task segment`
- GUI 分割识别：在模式选择“分割”后，点击右侧按钮“分割识别”进行掩膜可视化
