import os
import shutil
import glob
from tqdm import tqdm

# ================= 配置区域 (请修改这里) =================

# 1. 目标大本营 (不要改)
DST_ROOT = r"data/tunnel_anomaly"

# 2. 【数据集 A】二分类数据集路径 (FY387 Dataset 1)
# 里面应该有 Normal 和 Defect (或类似名字) 的文件夹
CLS_NORMAL_DIR = r"E:/我的项目/研究生/CrackDetector/data/tunnel_anomaly/dataset/Image Classification Dataset/dataset-tongji-fy/background"   # 你的无裂缝文件夹
CLS_DEFECT_DIR = r"E:/我的项目/研究生/CrackDetector/data/tunnel_anomaly/dataset/Image Classification Dataset/dataset-tongji-fy/defects"   # 你的有裂缝文件夹

# 3. 【数据集 B】检测数据集路径 (FY387 Dataset 2 - Generalization Test)
# 这里面的图通常都是比较难的测试图
DET_IMG_DIR = r"E:/我的项目/研究生/CrackDetector/data/tunnel_anomaly/dataset/Generalization Test-Object Detection/JPEGImages"      # 你的检测图片文件夹

# =======================================================

def safe_copy(src_path, dst_folder, new_name):
    """安全复制，带进度显示"""
    os.makedirs(dst_folder, exist_ok=True)
    dst_path = os.path.join(dst_folder, new_name)
    shutil.copy2(src_path, dst_path)

def main():
    print("🚀 开始合并剩余数据集...")

    # --- 处理 Dataset A: 二分类数据集 (Classification) ---
    print("\n📦 正在处理 [二分类数据集]...")
    
    # 1. 把 Normal 搬到 train/good (这能极大缓解误检！)
    if os.path.exists(CLS_NORMAL_DIR):
        files = glob.glob(os.path.join(CLS_NORMAL_DIR, "*.*"))
        # 过滤非图片
        files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.bmp'))]
        
        print(f"   -> 发现 {len(files)} 张正常图 (含管道/缝)，正在注入训练集...")
        for idx, f in enumerate(tqdm(files)):
            ext = os.path.splitext(f)[1]
            # 命名格式: fy_cls_norm_0001.jpg
            new_name = f"fy_cls_norm_{idx:05d}{ext}"
            safe_copy(f, os.path.join(DST_ROOT, "train/good"), new_name)
    else:
        print(f"❌ 路径不存在跳过: {CLS_NORMAL_DIR}")

    # 2. 把 Defect 搬到 test/crack
    if os.path.exists(CLS_DEFECT_DIR):
        files = glob.glob(os.path.join(CLS_DEFECT_DIR, "*.*"))
        files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.bmp'))]
        
        print(f"   -> 发现 {len(files)} 张缺陷图，正在注入测试集...")
        for idx, f in enumerate(tqdm(files)):
            ext = os.path.splitext(f)[1]
            # 命名格式: fy_cls_crack_0001.jpg
            new_name = f"fy_cls_crack_{idx:05d}{ext}"
            safe_copy(f, os.path.join(DST_ROOT, "test/crack"), new_name)
            # 注意：二分类数据没有Mask，所以不复制到 ground_truth
    else:
        print(f"❌ 路径不存在跳过: {CLS_DEFECT_DIR}")


    # --- 处理 Dataset B: 检测数据集 (Detection) ---
    print("\n📦 正在处理 [检测数据集]...")
    # 策略：检测数据集通常很难，我们把它全部当作“缺陷/难样本”放入测试集
    # 如果检测数据集里混有正常图，你需要手动挑出来，或者全部当测试看结果
    
    if os.path.exists(DET_IMG_DIR):
        files = glob.glob(os.path.join(DET_IMG_DIR, "*.*"))
        files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.bmp'))]
        
        print(f"   -> 发现 {len(files)} 张检测图，全部放入测试集作为'高难度考题'...")
        for idx, f in enumerate(tqdm(files)):
            ext = os.path.splitext(f)[1]
            # 命名格式: fy_det_0001.jpg
            new_name = f"fy_det_{idx:05d}{ext}"
            safe_copy(f, os.path.join(DST_ROOT, "test/crack"), new_name)
            # 同样，因为没有像素级Mask(只有框)，不放入 ground_truth
            # PatchCore 会计算 Image-Level AUC (图片分对了没)，这足够了
    else:
        print(f"❌ 路径不存在跳过: {DET_IMG_DIR}")

    print("\n" + "="*30)
    print("✅ 所有数据集整合完毕！")
    print("现在你的 data/tunnel_anomaly 包含了：")
    print("1. TTD 数据 (带 Mask 真值)")
    print("2. 二分类数据 (丰富了背景多样性，防误检)")
    print("3. 检测数据 (增加了测试难度)")
    print("="*30)

if __name__ == "__main__":
    main()