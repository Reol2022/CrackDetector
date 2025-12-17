import os
import cv2
import argparse
import json
from typing import List, Tuple, Optional


def list_videos(folder: str) -> List[str]:
    exts = (".mp4", ".avi", ".mov", ".mkv")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_frames(
    video_path: str,
    out_dir: str,
    every_n: int = 10,
    fps: Optional[float] = None,
) -> Tuple[List[str], dict]:
    """
    从视频抽帧。两种模式：
    - every_n：每隔 N 帧保存一张
    - fps：近似按固定帧率保存（优先使用 fps，如果提供）

    返回：(保存的帧路径列表, 元数据字典)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    saved_paths: List[str] = []
    idx = 0
    save_interval = every_n
    if fps and fps > 0:
        # 将目标 fps 转换为帧间隔
        save_interval = max(1, int(round(src_fps / fps)))

    ensure_dir(out_dir)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % save_interval == 0:
            out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)
        idx += 1

    cap.release()
    meta = {
        "video": video_path,
        "src_fps": src_fps,
        "total_frames": total,
        "width": w,
        "height": h,
        "saved_frames": len(saved_paths),
        "mode": "fps" if fps and fps > 0 else "every_n",
        "interval": save_interval,
    }
    return saved_paths, meta


def measure_quality(img) -> Tuple[float, float]:
    """返回 (亮度均值, Laplacian 方差) 作为质量指标。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())
    return brightness, blur_var


def auto_filter(
    frame_path: str,
    bright_min: float,
    bright_max: float,
    blur_th: float,
    yolo_model=None,
    conf: float = 0.25,
    iou: float = 0.45,
    device: Optional[str] = None,
) -> Tuple[bool, dict]:
    """
    自动筛选：先质量过滤，再可选 YOLO 检测过滤。
    返回 (是否保留, 诊断字典)
    """
    img = cv2.imread(frame_path)
    if img is None:
        return False, {"error": "read_failed"}
    brightness, blur_var = measure_quality(img)
    passed_quality = (bright_min <= brightness <= bright_max) and (blur_var >= blur_th)
    diag = {
        "brightness": brightness,
        "blur_var": blur_var,
        "passed_quality": passed_quality,
        "boxes": 0,
        "conf": conf,
        "iou": iou,
    }
    if not passed_quality:
        return False, diag

    if yolo_model is not None:
        results = yolo_model.predict(
            source=frame_path,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        r = results[0]
        boxes = 0
        try:
            if hasattr(r, "boxes") and hasattr(r.boxes, "data"):
                boxes = int(r.boxes.data.shape[0])
        except Exception:
            boxes = 0
        diag["boxes"] = boxes
        keep = boxes > 0
        return keep, diag

    # 未启用检测过滤，仅通过质量过滤
    return True, diag


def run_video_detect(video_path: str, weights: str, device: Optional[str], project: str, name: str, conf: float, iou: float, imgsz: int):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics")
    model = YOLO(weights)
    results = model.predict(
        source=video_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        save=True,
        verbose=True,
    )
    out_dir = None
    try:
        if results and hasattr(results[0], "save_dir"):
            out_dir = results[0].save_dir
    except Exception:
        pass
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="视频抽帧 + 自动筛选 + YOLO检测输出视频")
    parser.add_argument("--video_dir", default="data/video", help="视频输入目录（批量处理）")
    parser.add_argument("--out_frames_root", default="data/video/frames", help="抽帧输出根目录")
    parser.add_argument("--out_filtered_root", default="data/video/filtered", help="筛选输出根目录")
    parser.add_argument("--every_n", type=int, default=10, help="每隔N帧抽取一帧（与fps二选一）")
    parser.add_argument("--fps", type=float, default=0.0, help="按固定帧率近似抽帧（>0生效）")
    parser.add_argument("--bright_min", type=float, default=30.0, help="最低亮度阈值")
    parser.add_argument("--bright_max", type=float, default=220.0, help="最高亮度阈值")
    parser.add_argument("--blur_th", type=float, default=150.0, help="Laplacian方差阈值，用于清晰度过滤")
    parser.add_argument("--filter_by_detect", action="store_true", help="启用YOLO检测过滤，仅保留检测到裂缝的帧")
    parser.add_argument("--weights", default="checkpoints/best_yolov8_detect.pt", help="YOLO检测权重路径")
    parser.add_argument("--device", default=None, help="推理设备，如'0'或'cpu'")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO输入尺寸")
    parser.add_argument("--detect_video", action="store_true", help="对原始视频运行YOLO并输出带框视频")
    parser.add_argument("--detect_project", default="runs/detect", help="检测输出项目目录")
    parser.add_argument("--detect_name_prefix", default="video", help="检测输出任务名前缀，如 video_<stem>")
    args = parser.parse_args()

    # 准备YOLO模型用于过滤（可选）
    yolo_model = None
    if args.filter_by_detect:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics")
        weights = args.weights
        if not os.path.exists(weights):
            fallback = os.path.join("checkpoints", "best_yolov8_detect.pt")
            if os.path.exists(fallback):
                weights = fallback
                print(f"未找到指定权重，改用: {weights}")
        yolo_model = YOLO(weights)

    videos = list_videos(args.video_dir)
    if not videos:
        print(f"未在 {args.video_dir} 找到视频文件。")
        return

    summary = []
    for vp in videos:
        stem = os.path.splitext(os.path.basename(vp))[0]
        frames_out = os.path.join(args.out_frames_root, stem)
        filtered_out = os.path.join(args.out_filtered_root, stem)
        ensure_dir(frames_out)
        ensure_dir(filtered_out)

        # 抽帧
        frame_paths, meta = extract_frames(vp, frames_out, every_n=args.every_n, fps=args.fps)
        print(f"[{stem}] 抽帧完成：{meta['saved_frames']} 张，模式={meta['mode']}, interval={meta['interval']}")

        # 筛选
        kept = []
        diagnostics = {}
        for fp in frame_paths:
            keep, diag = auto_filter(
                fp,
                bright_min=args.bright_min,
                bright_max=args.bright_max,
                blur_th=args.blur_th,
                yolo_model=yolo_model,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
            )
            diagnostics[os.path.basename(fp)] = diag
            if keep:
                dst = os.path.join(filtered_out, os.path.basename(fp))
                img = cv2.imread(fp)
                if img is not None:
                    cv2.imwrite(dst, img)
                    kept.append(dst)

        # 保存筛选诊断
        diag_path = os.path.join(filtered_out, "diagnostics.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump({"video": vp, "meta": meta, "kept": kept, "diagnostics": diagnostics}, f, ensure_ascii=False, indent=2)
        print(f"[{stem}] 筛选完成：保留 {len(kept)} 张，诊断写入 {diag_path}")

        # 对原始视频输出检测结果视频
        if args.detect_video:
            name = f"{args.detect_name_prefix}_{stem}"
            out_dir = run_video_detect(
                video_path=vp,
                weights=args.weights,
                device=args.device,
                project=args.detect_project,
                name=name,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
            )
            if out_dir:
                print(f"[{stem}] 检测视频输出目录：{out_dir}")
        summary.append({"video": vp, "frames": len(frame_paths), "kept": len(kept)})

    print("处理完成：")
    for item in summary:
        print(f"- {os.path.basename(item['video'])}: frames={item['frames']}, kept={item['kept']}")


if __name__ == "__main__":
    main()

