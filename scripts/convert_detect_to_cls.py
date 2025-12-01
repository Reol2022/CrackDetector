import os
import argparse
import shutil
import yaml

def main():
    parser = argparse.ArgumentParser(description="根据检测标注生成二分类数据集")
    parser.add_argument("--data", default="configs/detect.yaml", help="detect.yaml 配置路径")
    parser.add_argument("--out", default="data/classify_derived", help="输出分类数据集根目录")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    root = y.get("path")
    train_images_rel = y.get("train", "train/images")
    val_images_rel = y.get("val", "valid/images")

    for split, images_rel in [("train", train_images_rel), ("val", val_images_rel)]:
        img_dir = os.path.join(root, images_rel)
        lbl_dir = os.path.join(root, images_rel.replace("images", "labels"))
        out_crack = os.path.join(args.out, split, "crack")
        out_no = os.path.join(args.out, split, "no_crack")
        os.makedirs(out_crack, exist_ok=True)
        os.makedirs(out_no, exist_ok=True)

        for name in os.listdir(img_dir):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            stem = os.path.splitext(name)[0]
            label_path = os.path.join(lbl_dir, stem + ".txt")
            src = os.path.join(img_dir, name)
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                shutil.copy2(src, os.path.join(out_crack, name))
            else:
                shutil.copy2(src, os.path.join(out_no, name))

    print(f"已生成分类数据集于: {args.out}")

if __name__ == "__main__":
    main()

