import os
import glob

# 你的项目根目录 (根据你的报错信息修改)
ROOT_DIR = r"E:\我的项目\研究生\CrackDetector\data\tunnel_anomaly"

def check_dir(name, subpath):
    full_path = os.path.join(ROOT_DIR, subpath)
    print(f"\n🔍 正在检查: {name}")
    print(f"   -> 路径: {full_path}")
    
    if not os.path.exists(full_path):
        print("   ❌ 文件夹不存在！")
        return 0
    
    # 搜索常见图片格式
    files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.tif']:
        files.extend(glob.glob(os.path.join(full_path, ext)))
        # 还要试试大写后缀
        files.extend(glob.glob(os.path.join(full_path, ext.upper())))
    
    count = len(files)
    if count == 0:
        print("   ❌ 文件夹存在，但里面没有图片！(或者图片在更深层的子文件夹里)")
        # 尝试列出前几个文件看看是什么
        all_files = os.listdir(full_path)
        if all_files:
            print(f"   👀 发现的文件/文件夹: {all_files[:5]}")
    else:
        print(f"   ✅ 成功找到 {count} 张图片！")
        print(f"   -> 示例: {files[0]}")
    return count

print("="*30)
c1 = check_dir("正常训练集 (Normal)", "train/good")
c2 = check_dir("异常测试集 (Abnormal)", "test/crack")
print("="*30)

if c1 > 0 and c2 > 0:
    print("🎉 结论：路径没问题，Anomalib 应该能读到。")
else:
    print("💀 结论：路径或文件结构有严重问题，请立即修复！")