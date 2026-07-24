import os
import shutil
from tqdm import tqdm

# 源数据路径
sources = [
    {
        'name': 'CrackDetection',
        'img_dir': r'E:\MyProject\Datasets\tunnel_anomaly\dataset\CrackDetection.v2-v2.yolov8\img\images',
        'label_dir': r'E:\MyProject\Datasets\tunnel_anomaly\dataset\CrackDetection.v2-v2.yolov8\img\labels',
        'class_map': {'crack': 0}  # 只有裂缝
    },
    {
        'name': 'Generalization',
        'img_dir': r'E:\MyProject\Datasets\tunnel_anomaly\dataset\Generalization Test-Object Detection\JPEGImages',
        'label_dir': r'E:\MyProject\Datasets\tunnel_anomaly\dataset\Generalization Test-Object Detection\labels',
        'class_map': {'crack': 0, 'leakage': 1}
    },
    {
        'name': 'TACK',
        'img_dir': r'E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\dataset\images',
        'label_dir': r'E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\dataset\labels',
        'class_map': {'crack': 0, 'leakage': 1}
    }
]

# 目标训练集路径
train_img_dir = r'E:\MyProject\Datasets\tunnel_anomaly\new_dataset\images'
train_label_dir = r'E:\MyProject\Datasets\tunnel_anomaly\new_dataset\labels'
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)

# 为了避免文件名冲突，添加前缀
prefixes = ['crack_', 'gen_', 'tack_']

for idx, src in enumerate(sources):
    img_files = [f for f in os.listdir(src['img_dir']) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in tqdm(img_files, desc=f"Copying {src['name']}"):
        # 新文件名加前缀
        new_img_name = f"{prefixes[idx]}{img_file}"
        new_txt_name = new_img_name.replace(new_img_name.split('.')[-1], 'txt')

        # 复制图片
        shutil.copy(os.path.join(src['img_dir'], img_file),
                    os.path.join(train_img_dir, new_img_name))

        # 复制标注文件（假设txt与图片同名）
        txt_file = img_file.replace(img_file.split('.')[-1], 'txt')
        src_txt = os.path.join(src['label_dir'], txt_file)
        if os.path.exists(src_txt):
            shutil.copy(src_txt, os.path.join(train_label_dir, new_txt_name))
        else:
            print(f"警告：{src_txt} 不存在")

print("整合完成！")