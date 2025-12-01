import argparse
import os

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics")

    parser = argparse.ArgumentParser(description="导出 YOLOv8 检测为 ONNX")
    parser.add_argument("--weights", default="checkpoints/best_yolov8_detect.pt", help="权重路径")
    parser.add_argument("--out", default="checkpoints/best_yolov8_detect.onnx", help="导出文件路径")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本")
    args = parser.parse_args()

    model = YOLO(args.weights)
    # Ultralytics 会将导出文件放在运行目录下，命名规则固定。
    model.export(format="onnx", imgsz=args.imgsz, opset=args.opset)
    # 尝试移动到指定输出路径
    # 常见输出路径：runs/detect/weights/best.onnx 或当前目录
    candidates = [
        "best.onnx",
        os.path.join("runs", "detect", "weights", "best.onnx"),
        os.path.join("runs", "detect", "train", "weights", "best.onnx"),
    ]
    for c in candidates:
        if os.path.exists(c):
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            try:
                os.replace(c, args.out)
                print(f"ONNX 已导出到: {args.out}")
                return
            except Exception:
                pass
    print("提示：未在候选位置找到 best.onnx，请查看 Ultralytics 输出目录。")

if __name__ == "__main__":
    main()

