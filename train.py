import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm

from utils.dataset import get_data_loaders
from models.model import get_model


def train(args):
    # YOLOv8 分类训练分支（可选）
    if args.use_yolov8:
        try:
            from ultralytics import YOLO
            import glob
            import shutil
        except ImportError:
            raise ImportError("未找到 ultralytics，请先安装：pip install ultralytics")

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # 加载YOLOv8分类模型（支持本地权重或模型名称）
        model = YOLO(args.yolo_model)
        print(f"使用YOLOv8分类模型: {args.yolo_model}")
        # 训练：data 指向根目录，包含 train/ 与 val/ 子文件夹
        # 强制指定设备与数据加载并行度，避免默认回落到CPU
        results = model.train(
            data=args.data_dir,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.yolo_device,
            workers=args.workers
        )
        # 复制最佳权重到 checkpoints
        try:
            candidates = glob.glob(os.path.join('runs', 'classify', '*', 'weights', 'best.pt'))
            if candidates:
                latest = max(candidates, key=os.path.getmtime)
                dst = os.path.join(args.checkpoint_dir, 'best_yolov8.pt')
                shutil.copy2(latest, dst)
                print(f"已保存最佳YOLOv8权重至: {dst}")
            else:
                print("未找到YOLOv8训练权重文件（runs/classify/*/weights/best.pt）")
        except Exception as e:
            print(f"复制YOLOv8最佳权重失败: {e}")
        return

    # YOLOv8 检测训练分支（可选）
    if args.use_yolov8_detect:
        try:
            from ultralytics import YOLO
            import glob
            import shutil
        except ImportError:
            raise ImportError("未找到 ultralytics，请先安装：pip install ultralytics")

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # 加载YOLOv8检测模型（支持本地权重或模型名称）
        model = YOLO(args.detect_model)
        print(f"使用YOLOv8检测模型: {args.detect_model}")

        # 训练：data 指向 detect.yaml（或 Roboflow 的 data.yaml）
        results = model.train(
            data=args.detect_data,
            epochs=args.detect_epochs,
            imgsz=args.detect_imgsz,
            batch=args.batch_size,
            device=args.yolo_device,
            workers=args.workers,
            project=args.detect_project,
            name=args.detect_name
        )

        # 复制最佳权重到 checkpoints
        try:
            candidates = glob.glob(os.path.join('runs', 'detect', '*', 'weights', 'best.pt'))
            if candidates:
                latest = max(candidates, key=os.path.getmtime)
                dst = os.path.join(args.checkpoint_dir, 'best_yolov8_detect.pt')
                shutil.copy2(latest, dst)
                print(f"已保存最佳YOLOv8检测权重至: {dst}")
            else:
                print("未找到YOLOv8检测训练权重文件（runs/detect/*/weights/best.pt）")
        except Exception as e:
            print(f"复制YOLOv8检测最佳权重失败: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集（使用工具函数，避免目录拼接错误）
    train_loader, val_loader = get_data_loaders(data_dir=args.data_dir, batch_size=args.batch_size)

    # 获取模型
    model = get_model('CrackDetector', num_classes=2, pretrained=True)
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0

    for epoch in range(args.epochs):
        # 训练模式
        model.train()
        running_loss = 0.0
        train_corrects = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)

        # 验证模式
        model.eval()
        running_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))

        scheduler.step()

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'final_model.pth'))
    print("训练完成!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crack Detection Training')
    parser.add_argument('--data_dir', type=str, default='data', help='数据集根目录（包含 train/ 与 val/ 子目录）')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型保存路径')
    # YOLOv8 分类训练相关参数
    parser.add_argument('--use_yolov8', action='store_true', help='使用YOLOv8分类训练替代ResNet50')
    parser.add_argument('--yolo_model', type=str, default='yolov8n-cls.pt', help='YOLOv8分类模型：名称或本地权重路径')
    parser.add_argument('--imgsz', type=int, default=224, help='YOLOv8输入尺寸（默认224）')
    parser.add_argument('--yolo_device', type=str, default='0', help='YOLOv8设备：GPU编号如"0"，或"cpu"')
    parser.add_argument('--workers', type=int, default=4, help='数据加载并行度（workers），适当调小可降低CPU占用')
    # YOLOv8 检测训练相关参数
    parser.add_argument('--use_yolov8_detect', action='store_true', help='使用YOLOv8进行检测训练')
    parser.add_argument('--detect_data', type=str, default='data/detect.yaml', help='检测数据集配置yaml路径')
    parser.add_argument('--detect_model', type=str, default='yolov8n.pt', help='YOLOv8检测模型：名称或本地权重路径')
    parser.add_argument('--detect_imgsz', type=int, default=640, help='YOLOv8检测输入尺寸（默认640）')
    parser.add_argument('--detect_epochs', type=int, default=100, help='YOLOv8检测训练轮数（默认100）')
    parser.add_argument('--detect_project', type=str, default='runs/detect', help='YOLOv8检测训练日志目录')
    parser.add_argument('--detect_name', type=str, default='train_crack', help='YOLOv8检测训练任务名称')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train(args)
