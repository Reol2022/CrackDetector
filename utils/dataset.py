import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CrackDataset(Dataset):
    """
    裂缝数据集类
    """
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir: 数据集根目录
            transform: 图像变换
            mode: 'train' 或 'val'
        """
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.mode = mode
        
        # 获取所有图像路径和标签
        self.crack_dir = os.path.join(self.root_dir, 'crack')
        self.no_crack_dir = os.path.join(self.root_dir, 'no_crack')
        
        self.crack_images = [(os.path.join(self.crack_dir, img), 1) 
                            for img in os.listdir(self.crack_dir) 
                            if img.endswith(('.jpg', '.jpeg', '.png'))]
        
        self.no_crack_images = [(os.path.join(self.no_crack_dir, img), 0) 
                               for img in os.listdir(self.no_crack_dir) 
                               if img.endswith(('.jpg', '.jpeg', '.png'))]
        
        self.images = self.crack_images + self.no_crack_images
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_data_loaders(data_dir='data', batch_size=32):
    """
    获取训练和验证数据加载器
    
    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        
    Returns:
        train_loader, val_loader
    """
    # 定义数据变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = CrackDataset(root_dir=data_dir, transform=train_transform, mode='train')
    val_dataset = CrackDataset(root_dir=data_dir, transform=val_transform, mode='val')
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader