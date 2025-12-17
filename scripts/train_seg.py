import argparse
import os
import shutil
import sys

def train_seg():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics")

    parser = argparse.ArgumentParser(description="YOLOv8 裂缝分割训练")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="初始权重，如 yolov8n-seg.pt")
    parser.add_argument("--data", default="configs/seg.yaml", help="数据集配置路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--name", default="train_crack_seg", help="训练任务名")
    parser.add_argument("--device", default=None, help="设备，如 '0' 或 'cpu'")
    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.data):
        print(f"错误: 配置文件 {args.data} 不存在。请确保已准备好分割数据集配置。")
        return

    print(f"开始分割训练: Model={args.model}, Data={args.data}, Epochs={args.epochs}")
    
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs/segment",
    )

    # 自动复制 best.pt 到 checkpoints/best_yolov8_seg.pt
    run_dir = os.path.join("runs", "segment", args.name)
    best_src = os.path.join(run_dir, "weights", "best.pt")
    os.makedirs("checkpoints", exist_ok=True)
    best_dst = os.path.join("checkpoints", "best_yolov8_seg.pt")
    
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"已复制最佳权重到: {best_dst}")
    else:
        print("警告：未找到 best.pt，可能训练未完成或目录不同。")

if __name__ == "__main__":
    train_seg()
