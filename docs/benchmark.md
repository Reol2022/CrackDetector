# 基准与评测

本文档说明不同模型的评测指标与简单对比方法。

## 指标

- 检测：mAP@0.5、mAP@0.5:0.95、Precision、Recall
- 分类：Accuracy、Precision、Recall、F1、混淆矩阵

## 检测模型的分类评测

使用脚本：

```sh
python scripts/evaluate_detect_as_cls.py --weights checkpoints/best_yolov8_detect.pt --data configs/detect.yaml
```

输出包含 TP/TN/FP/FN 与 Acc、P、R、F1。

## 可选对比项

- ResNet50（分类） vs YOLOv8n-cls（分类） vs YOLOv8n（检测派生分类）
- 数据量与标注质量对效果的影响（建议 ≥5k 样本）

