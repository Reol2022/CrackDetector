try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class YOLOv8Detector:
    """
    YOLOv8 检测封装：predict(image) 返回单图结果对象（含 boxes）。
    与原 models/model.py 中的 YOLOv8Detector 等价，便于拆分与复用。
    """
    def __init__(self, weights_path=None, device=None):
        if YOLO is None:
            raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics")
        self.device = device
        self.weights_path = weights_path or 'yolov8n.pt'
        self.model = YOLO(self.weights_path)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def predict(self, image):
        results = self.model.predict(image, device=self.device, verbose=False)
        return results[0]

