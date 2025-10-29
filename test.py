import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms

from models.model import get_model

def preprocess_image(image_path):
    """
    预处理图像
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    
    return image_tensor, image

def predict(model, image_tensor, device):
    """
    使用模型进行预测
    """
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
    return preds.item(), probs.cpu().numpy()[0]

def visualize_result(image, prediction, probabilities):
    """
    可视化预测结果
    """
    class_names = ['无裂缝', '有裂缝']
    
    plt.figure(figsize=(10, 5))
    
    # 显示图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'预测: {class_names[prediction]}')
    plt.axis('off')
    
    # 显示概率条形图
    plt.subplot(1, 2, 2)
    plt.bar(class_names, probabilities)
    plt.ylim(0, 1)
    plt.title('预测概率')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def highlight_cracks(image_path, output_path=None):
    """
    使用OpenCV突出显示裂缝
    """
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 形态学操作去除小噪点
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在原图上绘制轮廓
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = get_model(model_type="classification", num_classes=2)
    
    # 加载模型权重
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功加载模型: {args.model_path}")
    else:
        print(f"模型文件不存在: {args.model_path}")
        return
    
    model = model.to(device)
    
    # 预处理图像
    image_tensor, original_image = preprocess_image(args.image_path)
    
    # 进行预测
    prediction, probabilities = predict(model, image_tensor, device)
    
    # 可视化结果
    visualize_result(original_image, prediction, probabilities)
    
    # 如果预测为裂缝，则突出显示裂缝
    if prediction == 1:
        print("检测到裂缝，正在突出显示...")
        highlighted_image = highlight_cracks(args.image_path, 'highlighted_cracks.png')
        print("已保存突出显示的裂缝图像: highlighted_cracks.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="隧道裂缝检测模型测试")
    parser.add_argument('--image_path', type=str, required=True, help='测试图像路径')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='模型权重路径')
    
    args = parser.parse_args()
    main(args)