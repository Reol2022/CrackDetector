try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics") from e

class YOLOv8Segmenter:
    """YOLOv8 分割模型封装。
    predict(image) 返回单图结果对象，包含 masks。
    """
    def __init__(self, weights_path=None, device=None):
        self.device = device
        self.weights_path = weights_path or 'yolov8n-seg.pt'
        self.model = YOLO(self.weights_path)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def predict(self, image):
        results = self.model.predict(image, device=self.device, verbose=False)
        return results[0]
