import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from torchvision import transforms

from models.model import get_model

class CrackDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智慧隧道裂缝检测系统")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 创建界面
        self.create_widgets()
        
        # 当前图像
        self.current_image_path = None
        self.current_image = None
        
    def load_model(self):
        """加载模型"""
        try:
            model_path = "checkpoints/best_model.pth"
            if not os.path.exists(model_path):
                messagebox.showwarning("警告", "模型文件不存在，请先训练模型！")
                return
            
            self.model = get_model(model_type="classification", num_classes=2)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            print("模型加载成功！")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
    
    def create_widgets(self):
        """创建界面组件"""
        # 顶部标题
        title_frame = tk.Frame(self.root, bg="#4a7abc", height=60)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(title_frame, text="智慧隧道裂缝检测系统", font=("Arial", 18, "bold"), 
                              bg="#4a7abc", fg="white")
        title_label.pack(pady=10)
        
        # 主框架
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 左侧图像显示区域
        self.image_frame = tk.Frame(main_frame, bg="white", width=400, height=400)
        self.image_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, bg="white", text="请选择图像")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 右侧控制区域
        control_frame = tk.Frame(main_frame, bg="#f0f0f0", width=300)
        control_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH)
        
        # 按钮区域
        button_frame = tk.Frame(control_frame, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        self.select_button = tk.Button(button_frame, text="选择图像", command=self.select_image,
                                     bg="#4a7abc", fg="white", font=("Arial", 12), width=15)
        self.select_button.pack(pady=5)
        
        self.detect_button = tk.Button(button_frame, text="检测裂缝", command=self.detect_crack,
                                     bg="#4CAF50", fg="white", font=("Arial", 12), width=15)
        self.detect_button.pack(pady=5)
        self.detect_button.config(state=tk.DISABLED)
        
        self.highlight_button = tk.Button(button_frame, text="突出显示裂缝", command=self.highlight_crack,
                                       bg="#FF9800", fg="white", font=("Arial", 12), width=15)
        self.highlight_button.pack(pady=5)
        self.highlight_button.config(state=tk.DISABLED)
        
        # 结果显示区域
        result_frame = tk.LabelFrame(control_frame, text="检测结果", bg="#f0f0f0", font=("Arial", 12))
        result_frame.pack(fill=tk.X, pady=10, ipady=5)
        
        self.result_label = tk.Label(result_frame, text="等待检测...", bg="#f0f0f0", font=("Arial", 14))
        self.result_label.pack(pady=10)
        
        # 概率显示区域
        self.prob_frame = tk.Frame(control_frame, bg="#f0f0f0")
        self.prob_frame.pack(fill=tk.X, pady=10)
        
        self.prob_label_no_crack = tk.Label(self.prob_frame, text="无裂缝概率: 0%", bg="#f0f0f0", font=("Arial", 12))
        self.prob_label_no_crack.pack(anchor=tk.W)
        
        self.prob_label_crack = tk.Label(self.prob_frame, text="有裂缝概率: 0%", bg="#f0f0f0", font=("Arial", 12))
        self.prob_label_crack.pack(anchor=tk.W)
        
        # 状态栏
        self.status_bar = tk.Label(self.root, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_image(self):
        """选择图像"""
        file_path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.detect_button.config(state=tk.NORMAL)
            self.status_bar.config(text=f"已加载图像: {os.path.basename(file_path)}")
            self.result_label.config(text="等待检测...")
            self.prob_label_no_crack.config(text="无裂缝概率: 0%")
            self.prob_label_crack.config(text="有裂缝概率: 0%")
            self.highlight_button.config(state=tk.DISABLED)
    
    def display_image(self, image_path):
        """显示图像"""
        try:
            image = Image.open(image_path)
            self.current_image = image
            
            # 调整图像大小以适应显示区域
            image = self.resize_image(image, (380, 380))
            
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图像: {str(e)}")
    
    def resize_image(self, image, size):
        """调整图像大小，保持纵横比"""
        w, h = image.size
        ratio = min(size[0]/w, size[1]/h)
        new_size = (int(w*ratio), int(h*ratio))
        return image.resize(new_size, Image.LANCZOS)
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
        
        return image_tensor
    
    def detect_crack(self):
        """检测裂缝"""
        if not self.current_image_path or not self.model:
            messagebox.showwarning("警告", "请先选择图像并确保模型已加载！")
            return
        
        try:
            self.status_bar.config(text="正在检测...")
            
            # 预处理图像
            image_tensor = self.preprocess_image(self.current_image_path)
            
            # 进行预测
            with torch.no_grad():
                outputs = self.model(image_tensor.to(self.device))
                _, preds = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
            
            prediction = preds.item()
            probabilities = probs.cpu().numpy()[0]
            
            # 更新结果显示
            if prediction == 1:
                self.result_label.config(text="检测结果: 有裂缝", fg="red")
                self.highlight_button.config(state=tk.NORMAL)
            else:
                self.result_label.config(text="检测结果: 无裂缝", fg="green")
                self.highlight_button.config(state=tk.DISABLED)
            
            # 更新概率显示
            self.prob_label_no_crack.config(text=f"无裂缝概率: {probabilities[0]*100:.2f}%")
            self.prob_label_crack.config(text=f"有裂缝概率: {probabilities[1]*100:.2f}%")
            
            self.status_bar.config(text="检测完成")
        except Exception as e:
            messagebox.showerror("错误", f"检测过程中出错: {str(e)}")
            self.status_bar.config(text="检测失败")
    
    def highlight_crack(self):
        """突出显示裂缝"""
        if not self.current_image_path:
            return
        
        try:
            self.status_bar.config(text="正在处理...")
            
            # 读取图像
            image = cv2.imread(self.current_image_path)
            
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
            
            # 转换为PIL图像并显示
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            
            # 显示处理后的图像
            result_pil = self.resize_image(result_pil, (380, 380))
            photo = ImageTk.PhotoImage(result_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            self.status_bar.config(text="裂缝突出显示完成")
        except Exception as e:
            messagebox.showerror("错误", f"处理图像时出错: {str(e)}")
            self.status_bar.config(text="处理失败")

def main():
    root = tk.Tk()
    app = CrackDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()