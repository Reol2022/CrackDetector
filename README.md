# 智慧隧道裂缝检测系统

这是一个基于深度学习的隧道裂缝检测系统，可以自动识别隧道壁上的裂缝。

## 功能特点

- 自动下载和处理隧道裂缝数据集
- 基于深度学习的裂缝检测模型
- 模型训练与评估
- 裂缝检测结果可视化
- 简单的用户界面

## 项目结构

- `data/`: 数据集目录
- `models/`: 模型定义
- `utils/`: 工具函数
- `train.py`: 模型训练脚本
- `test.py`: 模型测试脚本


## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 下载数据集：
```bash
python download_dataset.py
```

2. 训练模型：
```bash
python train.py
```

3. 测试模型：
```bash
python test.py --image_path path/to/your/image.jpg
```

4. 启动用户界面：
```bash
python app.py
```

