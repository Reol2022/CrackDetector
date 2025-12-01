# 数据集与标注

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

## 分割标注（预留）

- 推荐使用 YOLOv8 `*-seg` 模型；标注需包含 `masks`（如 `train/masks`）
- 基本配置可参考 `configs/seg.yaml`

