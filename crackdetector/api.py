import os
from typing import Optional, List


def detect(
    source: str,
    weights: Optional[str] = None,
    device: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    project: Optional[str] = None,
    name: str = "predict",
    save_txt: bool = False,
    half: bool = False,
    verbose: bool = True,
):
    """YOLOv8 裂缝检测（批量/单图/视频）。

    返回 ultralytics 结果对象列表。会自动保存可视化结果到 project/name。
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics") from e

    weights_path = weights or "checkpoints/best_yolov8_detect.pt"
    if not os.path.exists(weights_path) and weights:
        # 如果传入的权重不存在，回退到 checkpoints/best_yolov8_detect.pt（若存在）
        fallback = os.path.join("checkpoints", "best_yolov8_detect.pt")
        if os.path.exists(fallback):
            weights_path = fallback

    model = YOLO(weights_path)
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        project=project or "runs/detect",
        name=name,
        save=True,
        save_txt=save_txt,
        half=half,
        verbose=verbose,
    )
    return results


def classify(
    source: str,
    weights: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 1,
):
    """ResNet50 二分类推理。

    - source: 单图路径或目录（仅图像文件）
    - weights: 训练好的权重路径（默认查找 checkpoints/best_model.pth）
    返回：[(path, prob_no_crack, prob_crack, pred_class)] 列表
    """
    import glob
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from models.resnet import CrackDetector as ResNetCrack

    # 设备
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 模型与权重
    model = ResNetCrack(num_classes=2, pretrained=False)
    ckpt = weights or os.path.join("checkpoints", "best_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"未找到分类权重文件: {ckpt}。请传入 --weights 或将权重放置到 checkpoints/ 下。"
        )
    state = torch.load(ckpt, map_location="cpu")
    try:
        model.load_state_dict(state)
    except Exception:
        # 兼容 state_dict 包裹
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"]) 
        else:
            raise
    model.eval().to(dev)

    # 变换
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 收集输入
    paths: List[str] = []
    if os.path.isdir(source):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(glob.glob(os.path.join(source, ext)))
    else:
        paths.append(source)

    outputs = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(dev)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
            pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
            outputs.append((p, probs[0], probs[1], pred))
    return outputs

