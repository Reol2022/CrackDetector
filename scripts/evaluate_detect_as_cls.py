import os
import argparse
from typing import List

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def main():
    parser = argparse.ArgumentParser(description="用检测模型进行图像级二分类评测")
    parser.add_argument("--weights", default="checkpoints/best_yolov8_detect.pt", help="检测权重路径")
    parser.add_argument("--data", default="configs/detect.yaml", help="detect.yaml 路径")
    parser.add_argument("--device", default=None, help="设备，如 '0' 或 'cpu'")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()

    if YOLO is None:
        raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics")

    # 读取 detect.yaml 获取 val 路径
    import yaml
    with open(args.data, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    dataset_root = y.get("path", "")
    val_images_rel = y.get("val", "valid/images")
    val_dir = os.path.join(dataset_root, val_images_rel)

    model = YOLO(args.weights)

    images = list_images(val_dir)
    tp = fp = tn = fn = 0
    for img in images:
        res = model.predict(img, device=args.device, conf=args.conf, verbose=False)[0]
        has_crack = (len(res.boxes) > 0)

        # 标签文件位置推断（val/labels 与 val/images 平级）
        labels_rel = val_images_rel.replace("images", "labels")
        label_dir = os.path.join(dataset_root, labels_rel)
        stem = os.path.splitext(os.path.basename(img))[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        gt_has_crack = os.path.exists(label_path) and os.path.getsize(label_path) > 0

        if gt_has_crack and has_crack:
            tp += 1
        elif (not gt_has_crack) and (not has_crack):
            tn += 1
        elif gt_has_crack and (not has_crack):
            fn += 1
        else:
            fp += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print(f"Images: {total}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()

