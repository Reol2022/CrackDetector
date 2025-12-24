# 变更日志

本文件记录项目的主要更新与修复。

## [v2.1.5] - 2025-12-24

### 新增
- 集成分割模型封装：`models/yolo_seg.py`
- `detect.py` 支持 `--task segment` 显式分割推理
- 新增 GUI “分割识别”按钮，分割模式下可一键掩膜可视化（`app.py`）
- 新增官方数据集配置：`configs/crack_seg_official.yaml`（默认指向 `data/crack-seg`）
- 新增自动下载与训练演示脚本：`scripts/download_and_train_seg.py`

### 修复
- 修复 GUI 分割掩膜尺寸不匹配报错，改用 `result.plot()` 保证显示尺寸对齐
- 统一分割数据集路径为 `data/crack-seg`，避免路径不一致导致训练失败

### 说明
- 分割训练示例：`python scripts/train_seg.py --data configs/crack_seg_official.yaml`
- 分割推理示例：`python detect.py --source path/to.jpg --weights checkpoints/best_yolov8_seg.pt --task segment`

## [v2.1.4]- 2025-12-17

### 新增

- 新增 YOLOv8 分割训练脚本：`scripts/train_seg.py`
- 更新了文档

## [v2.1.3] - 2025-12-11

### 新增

- 新增视频抽帧与自动筛选脚本：`scripts/video_extract_filter_and_detect.py`
  - 支持批量从 `data/video` 抽帧（按 `--fps` 或 `--every_n`）
  - 基于亮度与清晰度（Laplacian 方差）进行质量筛选
  - 可选启用 YOLO 检测筛选，仅保留检测到裂缝的帧
  - 为每个视频生成 `diagnostics.json`，记录筛选指标与结果
  - 可对原始视频直接运行 YOLO 并输出带框视频至 `runs/detect/<name>/`
  - 默认输出目录：抽帧 `data/video/frames/<stem>/`，筛选 `data/video/filtered/<stem>/`


### 说明

- 脚本依赖：`opencv-python`、`ultralytics`；权重默认读取 `checkpoints/best_yolov8_detect.pt`
- 常用示例：
  - 抽帧+质量筛选：`python scripts/video_extract_filter_and_detect.py --video_dir data/video --fps 2`
  - 启用检测筛选并输出检测视频：`python scripts/video_extract_filter_and_detect.py --video_dir data/video --fps 2 --filter_by_detect --weights checkpoints/best_yolov8_detect.pt --device 0 --detect_video`

## [v2.1.2] - 2025-12-03

### 新增

- 新增通用 YOLOv8 裂缝检测推理脚本：`detect.py`
- 在 `app.py` 中引入 `resource_path()`，适配打包后资源路径（权重、配置）
- 新增 PyInstaller 规范文件：`crackdetector.spec`
- 新增 Windows 打包脚本：`scripts/build_exe.ps1`
- 更新桌面版文档：重写并扩展 `docs/desktop.md`
- README 增补“桌面版（EXE）”小节：快速打包步骤与常见问题

### 修复

- 修正单文件模式打包时缺失权重导致中断：`scripts/build_exe.ps1` 现仅收集存在的文件

## [v2.1.1] - 2025-12-01

### 新增

- README 精简与信息架构重组：新增“文档索引”，将详细章节迁移到 `docs/`
- 更新快速开始与示例：统一示例类名为 `YOLOv8Detector`，修订结果示例与视频帧说明
- 新增“常见问题”版块：包括 Ultralytics 安装、权重路径、`configs/detect.yaml` 与 GPU 设备
- 全站链接改为可点击的 Markdown 链接，提升可读性
- 维护文档补充：`contributing.md` , `changelog.md`

### 修复

- 统一文档中的路径与命名不一致问题（例如模型示例类名、数据配置路径）

## [v2.1.0] - 2025-11-18

### 新增

- 在 `train.py` 集成 YOLOv8 检测训练入口（无需单独脚本）
- 新增 `configs/detect.yaml`，可直接接入 Roboflow YOLOv8 数据集
- 训练结束自动复制最佳权重到 `checkpoints/best_yolov8_detect.pt`
- 拆分模型文件：`models/resnet.py` 与 `models/yolo_detect.py`
- 新增脚本：`scripts/train_detect.py`、`scripts/convert_detect_to_cls.py`、`scripts/evaluate_detect_as_cls.py`、`scripts/export_onnx.py`
- 新增文档：`docs/getting_started.md`、`docs/datasets.md`、`docs/benchmark.md`、`docs/desktop.md`
- 新增维护文档：`contributing.md` 与 `changelog.md`

### 修复

- 解决 ResNet50 推理时报错：设备不一致（统一加载权重与输入张量到相同设备）
- README 路径与训练命令统一到 `configs/detect.yaml` 与 `checkpoints/best_yolov8_detect.pt`

## [v2.0.0] - 2025-11-02

### 新增

- 集成 YOLOv8 目标检测模型
- 支持裂缝边界框检测和定位
- 新增 YOLO 格式数据集支持
- 实时检测和视频处理功能
- 完整的模型评估指标

### 优化

- 模块化代码结构，支持多模型
- 改进的训练 pipeline
- 增强的数据预处理
- 更好的可视化输出

## [v1.0.x] - 2025-10

- 初始化项目与基础训练代码
- 添加数据集与使用说明
- 使用预训练权重与优化图像预处理步骤
