import argparse
import os


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics")

    parser = argparse.ArgumentParser(description="裂缝检测推理（YOLOv8）")
    parser.add_argument("--source", required=True, help="输入源：图像/视频路径或目录")
    parser.add_argument("--weights", default="checkpoints/best_yolov8_detect.pt", help="检测权重路径或模型名")
    parser.add_argument("--device", default=None, help="设备，如 '0' 或 'cpu'")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--project", default="runs/detect", help="输出项目目录")
    parser.add_argument("--name", default="predict", help="输出任务名称")
    parser.add_argument("--save_txt", action="store_true", help="保存预测为YOLO文本格式")
    parser.add_argument("--half", action="store_true", help="使用半精度推理（需GPU支持）")
    parser.add_argument("--task", default=None, choices=["detect", "segment", "classify"], help="指定任务类型（可选）")
    args = parser.parse_args()

    weights = args.weights
    # 智能 fallback 逻辑：仅当 weights 不存在且看起来不像官方预训练模型（如 yolov8n-seg.pt）时才 fallback
    if not os.path.exists(weights) and not weights.startswith("yolov8"):
        # 尝试查找本地检测权重
        det_fallback = os.path.join("checkpoints", "best_yolov8_detect.pt")
        # 尝试查找本地分割权重
        seg_fallback = os.path.join("checkpoints", "best_yolov8_seg.pt")
        
        if "seg" in weights and os.path.exists(seg_fallback):
             weights = seg_fallback
             print(f"未找到指定权重，自动切换到本地分割最佳权重: {weights}")
        elif os.path.exists(det_fallback):
            weights = det_fallback
            print(f"未找到指定权重，改用本地检测权重: {weights}")
        else:
             print(f"警告: 权重 '{weights}' 未找到，且无本地 fallback，Ultralytics 可能尝试下载官方权重。")

    model = YOLO(weights, task=args.task)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        half=args.half,
        verbose=True,
    )

    try:
        if results and hasattr(results[0], "save_dir"):
            print(f"输出已保存至: {results[0].save_dir}")
        else:
            print("推理完成。")
    except Exception:
        print("推理完成。")


if __name__ == "__main__":
    main()

