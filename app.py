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

def resource_path(relpath: str) -> str:
    """在打包环境与开发环境下安全解析资源路径。
    优先查找：PyInstaller临时目录(_MEIPASS) → 当前工作目录 → 可执行文件所在目录/源码目录。
    """
    candidates = []
    base_meipass = getattr(sys, "_MEIPASS", None)
    if base_meipass:
        candidates.append(base_meipass)
    candidates.append(os.getcwd())
    if getattr(sys, "frozen", False):
        candidates.append(os.path.dirname(sys.executable))
    else:
        candidates.append(os.path.dirname(__file__))
    for base in candidates:
        p = os.path.join(base, relpath)
        if os.path.exists(p):
            return p
    return relpath

class CrackDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CrackDetector")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 默认模式：分类/检测/分割
        self.mode_var = tk.StringVar(value="分类")
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 创建界面
        self.create_widgets()
        
        # 当前图像
        self.current_image_path = None
        self.current_image = None
        
    def load_model(self):
        """加载模型（根据模式）"""
        try:
            mode = self.mode_var.get()
            if mode == "分类":
                # 优先使用 ResNet50（本地训练的分类权重），否则使用 YOLOv8 分类
                resnet_path = resource_path("checkpoints/best_model.pth")
                if os.path.exists(resnet_path):

                    
                    # 加载 ResNet50 分类模型
                    self.model = get_model(model_name="CrackDetector", model_type="classification", num_classes=2, pretrained=False)
                    # 将权重映射到当前设备并加载，然后把模型移到设备，避免 CPU/GPU 不一致
                    state = torch.load(resnet_path, map_location=self.device)
                    self.model.load_state_dict(state if isinstance(state, dict) else state)
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    print(f"已加载 ResNet50 分类模型: {resnet_path}")
                else:
                    cls_path = resource_path("checkpoints/best_yolov8.pt")
                    if os.path.exists(cls_path):
                        weights = cls_path
                    else:
                        fallback = resource_path("yolov8n-cls.pt")
                        weights = fallback
                        messagebox.showinfo("提示", f"未找到 {cls_path}，改用 {os.path.basename(fallback)} 进行分类推理。")
                    # 加载 YOLOv8 分类模型
                    self.model = get_model(model_name="yolov8", model_type="classification", weights=weights, device=self.device)
                    self.model.eval()
            elif mode == "检测":
                det_path = resource_path("checkpoints/best_yolov8_detect.pt")
                if os.path.exists(det_path):
                    weights = det_path
                else:
                    fallback = resource_path("yolov8n.pt")
                    weights = fallback
                    messagebox.showinfo("提示", f"未找到 {det_path}，改用 {os.path.basename(fallback)} 进行检测推理。")
                self.model = get_model(model_name="yolov8-detect", weights=weights, device=self.device)
                self.model.eval()
            elif mode == "分割":
                seg_path = resource_path("checkpoints/best_yolov8_seg.pt")
                if os.path.exists(seg_path):
                    weights = seg_path
                else:
                    fallback = resource_path("yolov8n-seg.pt")
                    weights = fallback
                    messagebox.showinfo("提示", f"未找到 {seg_path}，改用 {os.path.basename(fallback)} 进行分割推理。")
                self.model = get_model(model_name="yolov8-seg", weights=weights, device=self.device)
                self.model.eval()
            else:
                raise ValueError(f"不支持的模式: {mode}")
            print(f"模型加载成功（模式: {mode}）！")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
    
    def create_widgets(self):
        """创建界面组件"""
        # 顶部标题
        title_frame = tk.Frame(self.root, bg="#4a7abc", height=60)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(title_frame, text="裂缝检测工具", font=("Arial", 18, "bold"),
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

        # 模式选择
        mode_frame = tk.LabelFrame(control_frame, text="任务模式", bg="#f0f0f0", font=("Arial", 12))
        mode_frame.pack(fill=tk.X, pady=10, ipady=5)
        tk.Label(mode_frame, text="选择模式：", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        mode_options = ["分类", "检测", "分割"]
        mode_menu = tk.OptionMenu(mode_frame, self.mode_var, *mode_options, command=lambda _: self.on_mode_change())
        mode_menu.config(width=10)
        mode_menu.pack(side=tk.LEFT, padx=5)
        
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
        
        self.segment_button = tk.Button(button_frame, text="分割识别", command=self.segment_crack,
                                        bg="#9C27B0", fg="white", font=("Arial", 12), width=15)
        self.segment_button.pack(pady=5)
        self.segment_button.config(state=tk.DISABLED)
        
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
            if self.mode_var.get() == "分割":
                self.segment_button.config(state=tk.NORMAL)
            else:
                self.segment_button.config(state=tk.DISABLED)
    
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
    
    def preprocess_image(self, image):
        """预处理图像"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 与训练时保持一致的尺寸调整
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)
    
    def detect_crack(self):
        """检测裂缝/位置/掩膜（按模式）"""
        if not self.current_image_path or not self.model:
            messagebox.showwarning("警告", "请先选择图像并确保模型已加载！")
            return

        try:
            self.status_bar.config(text="正在检测...")

            # 加载图像
            image = Image.open(self.current_image_path).convert('RGB')
            mode = self.mode_var.get()

            if mode == "分类":
                if hasattr(self.model, 'predict'):
                    prediction, probabilities = self.model.predict(image)
                    if probabilities is None:
                        probabilities = np.array([0.0, 0.0])
                else:
                    image_tensor = self.preprocess_image(image)
                    image_tensor = image_tensor.unsqueeze(0)
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

                if isinstance(probabilities, np.ndarray) and probabilities.size >= 2:
                    self.prob_label_no_crack.config(text=f"无裂缝概率: {probabilities[0]*100:.2f}%")
                    self.prob_label_crack.config(text=f"有裂缝概率: {probabilities[1]*100:.2f}%")
                else:
                    self.prob_label_no_crack.config(text="无裂缝概率: -")
                    self.prob_label_crack.config(text="有裂缝概率: -")

            elif mode == "检测":
                if not hasattr(self.model, 'predict'):
                    messagebox.showerror("错误", "当前模式需要 YOLOv8 检测模型。")
                    return
                r = self.model.predict(image)
                im = np.array(image.convert('RGB'))
                has_box = False
                if getattr(r, 'boxes', None) is not None:
                    for box in r.boxes:
                        has_box = True
                        b = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy()) if hasattr(box, 'conf') else 0.0
                        cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else 0
                        name = r.names.get(cls_id, 'crack') if hasattr(r, 'names') else 'crack'
                        x1, y1, x2, y2 = map(int, b)
                        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(im, f"{name} {conf:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                result_pil = Image.fromarray(im)
                result_pil = self.resize_image(result_pil, (380, 380))
                photo = ImageTk.PhotoImage(result_pil)
                self.image_label.config(image=photo)
                self.image_label.image = photo
                if has_box:
                    self.result_label.config(text="检测结果: 有裂缝(检测)", fg="red")
                else:
                    self.result_label.config(text="检测结果: 无裂缝(检测)", fg="green")
                self.prob_label_no_crack.config(text="无裂缝概率: -")
                self.prob_label_crack.config(text="有裂缝概率: -")
                self.highlight_button.config(state=tk.DISABLED)

            elif mode == "分割":
                if not hasattr(self.model, 'predict'):
                    messagebox.showerror("错误", "当前模式需要 YOLOv8 分割模型。")
                    return
                r = self.model.predict(image)
                plotted = r.plot()
                plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                result_pil = Image.fromarray(plotted_rgb)
                result_pil = self.resize_image(result_pil, (380, 380))
                photo = ImageTk.PhotoImage(result_pil)
                self.image_label.config(image=photo)
                self.image_label.image = photo
                has_mask = getattr(r, 'masks', None) is not None and r.masks is not None
                if has_mask:
                    self.result_label.config(text="检测结果: 有裂缝(分割)", fg="red")
                else:
                    self.result_label.config(text="检测结果: 无裂缝(分割)", fg="green")
                self.prob_label_no_crack.config(text="无裂缝概率: -")
                self.prob_label_crack.config(text="有裂缝概率: -")
                self.highlight_button.config(state=tk.DISABLED)
            else:
                messagebox.showerror("错误", f"未知模式: {mode}")
                return

            self.status_bar.config(text="检测完成")
        except Exception as e:
            messagebox.showerror("错误", f"检测过程中出错: {str(e)}")
            self.status_bar.config(text="检测失败")

    def on_mode_change(self):
        """模式变更后重新加载模型"""
        try:
            self.load_model()
            self.result_label.config(text="等待检测...", fg="black")
            self.prob_label_no_crack.config(text="无裂缝概率: 0%")
            self.prob_label_crack.config(text="有裂缝概率: 0%")
            self.highlight_button.config(state=tk.DISABLED)
            self.status_bar.config(text="模式已切换，请选择图像并检测")
            if self.mode_var.get() == "分割":
                self.segment_button.config(state=tk.NORMAL if self.current_image_path else tk.DISABLED)
            else:
                self.segment_button.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("错误", f"切换模式失败: {str(e)}")
    
    def segment_crack(self):
        if not self.current_image_path or not self.model:
            messagebox.showwarning("警告", "请先选择图像并确保模型已加载！")
            return
        if self.mode_var.get() != "分割":
            messagebox.showwarning("警告", "当前模式不是分割，请切换到分割模式。")
            return
        try:
            self.status_bar.config(text="正在分割识别...")
            image = Image.open(self.current_image_path).convert('RGB')
            if not hasattr(self.model, 'predict'):
                messagebox.showerror("错误", "当前模式需要 YOLOv8 分割模型。")
                return
            r = self.model.predict(image)
            plotted = r.plot()
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(plotted_rgb)
            result_pil = self.resize_image(result_pil, (380, 380))
            photo = ImageTk.PhotoImage(result_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            has_mask = getattr(r, 'masks', None) is not None and r.masks is not None
            if has_mask:
                self.result_label.config(text="检测结果: 有裂缝(分割)", fg="red")
            else:
                self.result_label.config(text="检测结果: 无裂缝(分割)", fg="green")
            self.prob_label_no_crack.config(text="无裂缝概率: -")
            self.prob_label_crack.config(text="有裂缝概率: -")
            self.status_bar.config(text="分割识别完成")
        except Exception as e:
            messagebox.showerror("错误", f"分割识别过程中出错: {str(e)}")
            self.status_bar.config(text="分割识别失败")
    
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
