import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import time

from models.model import get_model
from utils.dataset import get_data_loaders

def train(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)
    print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    model = get_model(model_type="classification", num_classes=2)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 创建TensorBoard日志
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 训练循环
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印统计信息
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'final_model.pth'))
    print("训练完成！")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="隧道裂缝检测模型训练")
    parser.add_argument('--data_dir', type=str, default='data', help='数据集目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard日志目录')
    
    args = parser.parse_args()
    train(args)