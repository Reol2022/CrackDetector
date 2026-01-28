
# 🔍 CrackDetector: Tunnel Surface Anomaly Detection

> **Current Branch: `feature/anomaly-detection` (Experimental)**

<div align="center">

**基于 PatchCore 的隧道表面无监督异常检测**
**只看好图，就能发现未知缺陷**

[核心原理](https://www.google.com/search?q=%23-%E6%A0%B8%E5%BF%83%E5%8E%9F%E7%90%86) • [环境准备](https://www.google.com/search?q=%23-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87) • [数据准备](https://www.google.com/search?q=%23-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87) • [开始训练](https://www.google.com/search?q=%23-%E5%BC%80%E5%A7%8B%E8%AE%AD%E7%BB%83) • [可视化结果](https://www.google.com/search?q=%23-%E5%8F%AF%E8%A7%86%E5%8C%96%E7%BB%93%E6%9E%9C)

</div>

---

## 📖 分支简介 (Branch Introduction)

本分支 (`feature/anomaly-detection`) 是 `CrackDetector` 项目的实验性分支，专注于探索 **无监督学习 (Unsupervised Learning)** 在隧道及交通基础设施缺陷检测中的应用。

与主分支（Master）基于 YOLO 的监督学习不同，本分支采用 **PatchCore** 算法，解决了隧道场景下**负样本（缺陷数据）稀缺**和**缺陷形态不可预测**（如不规则裂缝、渗水、剥落）的痛点。

### ✨ 核心特性

* **🧠 无监督驱动**: 仅需少量**正常（无缺陷）**图片即可训练，无需费力标注裂缝框。
* **🔥 PatchCore 算法**: 引入先进的特征记忆库机制，实现 SOTA 级别的异常检测精度。
* **🎯 像素级定位**: 输出高分辨率的热力图 (Heatmap) 和分割掩码 (Segmentation Mask)。
* **🚀 边缘适配 (进行中)**: 探索 MobileNet/EfficientNet 轻量化骨干网络，适配隧道巡检机器人。
* **🏗️ 复杂环境鲁棒性**: 针对隧道暗光、管片拼接缝、水渍等干扰进行了专门优化。

---

## 🏗️ 架构对比

| 特性 | 主分支 (`master`) | **当前分支 (`feature/anomaly-detection`)** |
| --- | --- | --- |
| **核心模型** | YOLOv8 / ResNet50 | **PatchCore / Padim (via Anomalib)** |
| **学习方式** | 全监督 (Supervised) | **无监督 (Unsupervised)** |
| **数据要求** | 需要大量画框标注 | **仅需正常图片 (少量故障图用于测试)** |
| **检测目标** | 已知类别 (裂缝) | **未知异常 (裂缝、渗水、异物等)** |
| **输出结果** | 边界框 (BBox) | **热力图 (Heatmap) & 异常评分** |

---

## 🛠️ 环境准备

本分支依赖 `anomalib` 库。请确保在虚拟环境中安装：

```bash
# 基础依赖
pip install -r requirements.txt

# 安装异常检测核心库
pip install anomalib

# (可选) 如果需要处理旧数据
pip install opencv-python tqdm

```

---

## 📂 数据准备 (Data Preparation)

本分支使用全新的数据结构。请在 `data/` 目录下创建 `tunnel_anomaly` 文件夹，并严格按照以下结构放置图片：

```text
data/tunnel_anomaly/
├── train/
│   └── good/              # 【训练集】只放正常的隧道/混凝土壁面图
│       ├── ttd_norm_001.jpg
│       └── ...
│
├── test/
│   ├── good/              # (可选) 测试用的好图
│   └── crack/             # 【测试集】放包含裂缝、渗水等异常的图
│       ├── ttd_crack_001.jpg
│       └── ...
│
└── ground_truth/          # (可选) 对应的像素级掩码，用于计算 Pixel-AUC
    └── crack/
        ├── ttd_crack_001_mask.png
        └── ...

```

> **提示**: 如果您只有旧的 YOLO/分类数据集，请使用 `scripts/convert_detect_to_cls.py` 或 `rename_data.py` 脚本进行转换和清洗。

---

## 🚀 快速开始 (Quick Start)

### 1. 配置模型

项目根目录下已预置配置文件 `configs/patchcore.yaml`。

* 默认骨干网络：`wide_resnet50_2` (高精度)
* 轻量化尝试：可修改为 `mobilenet_v3_large`

### 2. 启动训练

无需编写复杂代码，一行命令即可启动：

```bash
anomalib train --config configs/patchcore.yaml

```

### 3. 查看结果

训练完成后，结果将保存在 `results/` 目录下：

```text
results/
└── patchcore/
    └── tunnel_custom/
        └── latest/
            ├── images/       # 包含原图与热力图的叠加对比
            └── logs/         # 训练日志

```

---

## 🖼️ 可视化结果预期

成功运行后，您将获得如下形式的异常定位图：

| 原始输入 (Input) | 异常热力图 (Heatmap) | 分割结果 (Segmentation) |
| --- | --- | --- |
| *(隧道壁原图)* | *(红色高亮区域)* | *(裂缝二值化掩码)* |

*(注：此处等待实验结果补充截图)*

---

## 📝 开发计划 (To-Do)

* [x] 搭建 Anomalib 环境与 PatchCore 基准
* [ ] 数据集清洗与标准化 (TTD + FY387)
* [ ] **实验一**: 验证 ResNet50 在隧道场景下的有效性 (Baseline)
* [ ] **实验二**: 替换 MobileNet/EfficientNet 进行轻量化对比
* [ ] **实验三**: 引入光照增强预处理 (Gamma/Retinex)
* [ ] 导出 ONNX 模型并部署测试

---

## 🤝 贡献与反馈

本分支是科研实验性质，欢迎提交 Issue 讨论关于 "Unsupervised Defect Detection" 的想法。

* **Author**: Reol2020
* **Focus**: AI for Civil Engineering (Tunnel Inspection)

---
