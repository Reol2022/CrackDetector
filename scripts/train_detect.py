import argparse
import os
import shutil

def train_detect():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics")

    parser = argparse.ArgumentParser(description="YOLOv8 裂缝检测训练")
    parser.add_argument("--model", default="yolov8n.pt", help="初始权重，如 yolov8n.pt/yolov8s.pt")
    parser.add_argument("--data", default="configs/detect.yaml", help="数据集配置路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--name", default="train_crack", help="训练任务名")
    parser.add_argument("--device", default=None, help="设备，如 '0' 或 'cpu'")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs/detect",
    )

    # 自动复制 best.pt 到 checkpoints/best_yolov8_detect.pt
    run_dir = os.path.join("runs", "detect", args.name)
    best_src = os.path.join(run_dir, "weights", "best.pt")
    os.makedirs("checkpoints", exist_ok=True)
    best_dst = os.path.join("checkpoints", "best_yolov8_detect.pt")
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"已复制最佳权重到: {best_dst}")
    else:
        print("警告：未找到 best.pt，可能训练未完成或目录不同。")

if __name__ == "__main__":
    train_detect()

