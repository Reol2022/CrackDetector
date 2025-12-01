# 桌面版打包与资源路径

## PyInstaller 打包（Windows）

单文件：

```sh
pyinstaller -F detect.py --name CrackDetector --add-data "configs;configs" --add-data "checkpoints;checkpoints" --collect-submodules ultralytics
```

目录模式（便于资源管理）：

```sh
pyinstaller -D detect.py --name CrackDetector --add-data "configs;configs" --add-data "checkpoints;checkpoints" --collect-submodules ultralytics
```

## 资源路径适配

为兼容打包后的临时目录，建议使用 `resource_path()`：

```python
import os, sys

def resource_path(rel_path: str) -> str:
    base = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base, rel_path)

# 用法示例
weights = resource_path("checkpoints/best_yolov8_detect.pt")
data_yaml = resource_path("configs/detect.yaml")
```

## 验证与分发

- 在打包目录下运行，验证检测与视频推理
- 将 `checkpoints/` 与 `configs/` 一并分发或内嵌到 exe（目录模式更灵活）

