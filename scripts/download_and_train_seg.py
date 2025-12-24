import argparse
import os
import shutil
import sys
from ultralytics import YOLO

def download_and_train():
    """
    使用 Ultralytics 官方支持的 crack-seg 数据集进行自动下载和训练。
    """
    print("正在准备下载 crack-seg 数据集并开始训练...")
    print("数据集将自动下载到 ../datasets/crack-seg (相对于当前运行目录)")
    
    # 使用官方数据集配置名称，Ultralytics 会自动识别并下载
    # 也可以使用本地配置 configs/crack_seg_official.yaml
    data_config = "crack-seg.yaml" 
    
    try:
        model = YOLO("yolov8n-seg.pt")
        
        # 启动训练，这会自动触发下载
        results = model.train(
            data=data_config,
            epochs=10,        # 演示用，设为 10 轮
            imgsz=640,
            batch=16,
            name="train_crack_seg_official",
            project="runs/segment"
        )
        
        print("训练完成！")
        print(f"结果保存在: {results.save_dir}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("\n如果下载失败，请尝试手动下载：")
        print("链接: https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip")
        print("解压到: ../datasets/crack-seg")

if __name__ == "__main__":
    download_and_train()
