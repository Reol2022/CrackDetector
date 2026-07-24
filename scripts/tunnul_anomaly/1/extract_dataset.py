import os
import csv
import shutil
from tqdm import tqdm

def extract_dataset(mapping_path, img_src_dir, txt_src_dir, dst_root):
    """
    根据映射文件，将有病害的图片和对应的 txt 文件复制到新数据集目录
    """
    images_dst = os.path.join(dst_root, 'images')
    labels_dst = os.path.join(dst_root, 'labels')
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)

    with open(mapping_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行（如果有）
        for row in tqdm(reader, desc="Copying"):
            if len(row) < 2:
                continue
            txt_name, img_name = row[0].strip(), row[1].strip()
            # 源文件路径
            img_src = os.path.join(img_src_dir, img_name)
            txt_src = os.path.join(txt_src_dir, txt_name)
            # 目标文件路径（保持相同文件名）
            img_dst = os.path.join(images_dst, img_name)
            txt_dst = os.path.join(labels_dst, txt_name)

            # 复制图片
            if os.path.exists(img_src):
                shutil.copy2(img_src, img_dst)
            else:
                print(f"警告：图片不存在 {img_src}")
            # 复制标注
            if os.path.exists(txt_src):
                shutil.copy2(txt_src, txt_dst)
            else:
                print(f"警告：标注不存在 {txt_src}")

    print(f"提取完成！新数据集位于：{dst_root}")
    print(f"images 数量：{len(os.listdir(images_dst))}")
    print(f"labels 数量：{len(os.listdir(labels_dst))}")

if __name__ == "__main__":
    # ========== 配置区域 ==========
    mapping_path = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\mapping.csv"
    img_src_dir = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\3_img"
    txt_src_dir = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\labels"
    dst_root = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\dataset"   # 新数据集根目录
    # =============================

    extract_dataset(mapping_path, img_src_dir, txt_src_dir, dst_root)