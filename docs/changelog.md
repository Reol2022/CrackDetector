# 变更日志

本文件记录项目的主要更新与修复。

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
