import argparse
import os


def main():
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from models.resnet import CrackDetector as ResNetCrack

    parser = argparse.ArgumentParser(description="CrackDetector: ResNet50 二分类 CLI")
    parser.add_argument("--source", required=True, help="输入源：图像路径或目录")
    parser.add_argument("--weights", default=None, help="分类权重路径（默认尝试 checkpoints/best_model.pth）")
    parser.add_argument("--device", default=None, help="设备，如 '0' 或 'cpu'")
    args = parser.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = args.weights or os.path.join("checkpoints", "best_model.pth")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"未找到分类权重文件: {weights}")

    model = ResNetCrack(num_classes=2, pretrained=False)
    state = torch.load(weights, map_location="cpu")
    try:
        model.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"]) 
        else:
            raise
    model.eval().to(dev)

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    import glob
    import sys
    from pathlib import Path

    def iter_paths(src: str):
        if os.path.isdir(src):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                yield from Path(src).glob(ext)
        else:
            yield Path(src)

    with torch.no_grad():
        for p in iter_paths(args.source):
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(dev)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
            pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
            print(f"{p}: no_crack={probs[0]:.4f}, crack={probs[1]:.4f}, pred={pred}")


if __name__ == "__main__":
    main()

