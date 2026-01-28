import os

# 确保 configs 文件夹存在
os.makedirs("configs", exist_ok=True)

# ★★★ 3050 Ti 专用：ResNet18 + 224x224 + v1.0语法修复 ★★★
config_content = """# Anomalib v1.0 Laptop Config (Fixed & Slim)
seed_everything: 42

data:
  class_path: anomalib.data.Folder
  init_args:
    name: tunnel_anomaly
    # 👇 保持你之前的全英文路径 (请确认图片都在这里面)
    root: E:/Datasets/tunnel_anomaly
    
    normal_dir: train/good
    abnormal_dir: test/crack
    normal_test_dir: test/good 
    mask_dir: null
    extensions: null
    
    # ⚠️ 注意：image_size 在 v1.0 中不能写在这里了，已删除！
    
    # 👇 4GB 显存比较紧张，先用 4 试试，不行就改 1
    train_batch_size: 4
    eval_batch_size: 4
    num_workers: 0  # Windows 下多进程容易报错，本地跑设为 0 最稳
    
    val_split_mode: from_test
    val_split_ratio: 0.2

model:
  class_path: anomalib.models.Patchcore
  init_args:
    # 👇 核心：使用小模型 ResNet18，省显存
    backbone: resnet18
    pre_trained: true
    
    # 👇 核心：在这里定义输入尺寸 (v1.0 写法)
    input_size: [224, 224]
    
    coreset_sampling_ratio: 0.1
    num_neighbors: 9

trainer:
  accelerator: auto
  devices: 1
  max_epochs: 1
  default_root_dir: ./results
  callbacks:
    - class_path: anomalib.callbacks.ModelCheckpoint
      init_args:
        mode: max
        monitor: image_AUROC
        save_last: true
"""

with open("configs/config_laptop_final.yaml", "w", encoding="utf-8") as f:
    f.write(config_content)

print("✅ 本地配置已修复 (v1.0语法 + ResNet18)！")
print("🚀 请运行: anomalib train --config configs/config_laptop_final.yaml")