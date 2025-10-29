import os
import urllib.request
import zipfile
import shutil
import glob
from tqdm import tqdm


def download_dataset():
    """
    下载隧道裂缝数据集并解压到data目录
    使用的是SDNET2018数据集，包含混凝土、路面和墙体的裂缝图像
    """
    print("开始下载隧道裂缝数据集...")

    # 创建数据目录
    os.makedirs("data", exist_ok=True)

    # 下载数据集 (使用SDNET2018数据集)
    url = "https://digitalcommons.usu.edu/context/all_datasets/article/1047/type/native/viewcontent"
    output = "data/concrete_crack_images.zip"
    extract_dir = "data/SDNET2018_extracted"

    if not os.path.exists(output):
        print(f"正在从 {url} 下载数据集...")
        try:
            # 使用urllib下载文件，显示进度条
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
                print(f"\r下载进度: {percent}% [{downloaded} / {total_size}]", end="")

            urllib.request.urlretrieve(url, output, reporthook=report_progress)
            print("\n下载完成！")
        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载SDNET2018数据集并放置在data目录下")
            print("数据集下载链接: https://digitalcommons.usu.edu/all_datasets/48/")
            return

    # 解压数据集
    print("解压数据集...")
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    # 重新组织数据集结构
    print("重新组织数据集结构...")
    os.makedirs("data/train/crack", exist_ok=True)
    os.makedirs("data/train/no_crack", exist_ok=True)
    os.makedirs("data/val/crack", exist_ok=True)
    os.makedirs("data/val/no_crack", exist_ok=True)

    # SDNET2018数据集结构:
    # P/CP - 混凝土裂缝图像
    # P/UP - 混凝土无裂缝图像
    # W/CW - 墙体裂缝图像
    # W/UW - 墙体无裂缝图像
    # D/UD - 路面无裂缝图像 (没有路面裂缝图像)

    # 收集所有裂缝图像
    crack_images = []
    crack_paths = [
        os.path.join(extract_dir, "P", "CP", "*.jpg"),
        os.path.join(extract_dir, "W", "CW", "*.jpg")
    ]
    
    for path_pattern in crack_paths:
        crack_images.extend(glob.glob(path_pattern))
    
    print(f"找到 {len(crack_images)} 张裂缝图像")
    
    # 收集所有无裂缝图像
    no_crack_images = []
    no_crack_paths = [
        os.path.join(extract_dir, "P", "UP", "*.jpg"),
        os.path.join(extract_dir, "W", "UW", "*.jpg"),
        os.path.join(extract_dir, "D", "UD", "*.jpg")
    ]
    
    for path_pattern in no_crack_paths:
        no_crack_images.extend(glob.glob(path_pattern))
    
    print(f"找到 {len(no_crack_images)} 张无裂缝图像")

    # 移动裂缝图像
    print("处理裂缝图像...")
    for i, img_path in enumerate(tqdm(crack_images)):
        img_name = os.path.basename(img_path)
        if i < int(len(crack_images) * 0.8):  # 80%用于训练
            dst = os.path.join("data/train/crack", img_name)
        else:  # 20%用于验证
            dst = os.path.join("data/val/crack", img_name)
        shutil.copy(img_path, dst)

    # 移动无裂缝图像
    print("处理无裂缝图像...")
    for i, img_path in enumerate(tqdm(no_crack_images)):
        img_name = os.path.basename(img_path)
        if i < int(len(no_crack_images) * 0.8):  # 80%用于训练
            dst = os.path.join("data/train/no_crack", img_name)
        else:  # 20%用于验证
            dst = os.path.join("data/val/no_crack", img_name)
        shutil.copy(img_path, dst)

    print("数据集准备完成！")
    print(f"训练集裂缝图像: {len(os.listdir('data/train/crack'))}张")
    print(f"训练集无裂缝图像: {len(os.listdir('data/train/no_crack'))}张")
    print(f"验证集裂缝图像: {len(os.listdir('data/val/crack'))}张")
    print(f"验证集无裂缝图像: {len(os.listdir('data/val/no_crack'))}张")


if __name__ == "__main__":
    download_dataset()