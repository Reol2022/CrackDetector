# 桌面版打包与资源路径（Windows）

## 💻 桌面版（EXE）

将 GUI 打包为 Windows 可执行文件，开箱即用。

- 一键打包（目录模式推荐）：
  - `./scripts/build_exe.ps1`
- 单文件模式（体积较大）：
  - `./scripts/build_exe.ps1 -OneFile`
- 或使用规范文件：
  - `pyinstaller --noconfirm --clean crackdetector.spec`

运行方式：
- 目录模式：`dist/CrackDetector/CrackDetector.exe`
- 单文件模式：`dist/CrackDetector.exe`

常见问题与排查：
- `ultralytics` 数据缺失或模块未收集
  - 现已在 `crackdetector.spec` 中使用 `collect_data('ultralytics')` 与隐藏导入；若自行命令行打包，请添加：
    - `--hidden-import ultralytics --hidden-import torch --hidden-import torchvision --hidden-import cv2 --hidden-import PIL --hidden-import numpy`
- `torch` 体积过大（单文件 2GB+）
  - 安装 CPU-only 版本再打包：
    - `python -m pip uninstall -y torch torchvision torchaudio`
    - `python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
  - 尽量使用目录模式打包，外置权重与配置到与 exe 同目录（`checkpoints/`、`configs/`），避免内嵌增大体积。


## 打包方式

- 依赖：`pip install pyinstaller`
- 推荐：使用 `scripts/build_exe.ps1` 一键打包 GUI（`app.py`）。

方式 A：一键脚本

```powershell
# 目录模式（推荐，资源管理更直观）
./scripts/build_exe.ps1

# 单文件模式（将全部打包为一个 exe）
./scripts/build_exe.ps1 -OneFile
```

方式 B：使用 spec 文件（目录模式）

```powershell
pyinstaller --noconfirm --clean crackdetector.spec
```

方式 C：直接命令（单文件示例）

```powershell
pyinstaller --noconfirm --clean `
  --name CrackDetector `
  --onefile --windowed `
  --add-data "configs/detect.yaml;configs" `
  --add-data "configs/classify.yaml;configs" `
  --add-data "configs/seg.yaml;configs" `
  --add-data "checkpoints/best_yolov8.pt;checkpoints" `
  --add-data "checkpoints/best_yolov8_detect.pt;checkpoints" `
  --add-data "checkpoints/best_yolov8_seg.pt;checkpoints" `
  --add-data "yolov8n.pt;." `
  --add-data "yolov8n-cls.pt;." `
  --add-data "yolov8n-seg.pt;." `
  --hidden-import ultralytics --hidden-import torch --hidden-import torchvision --hidden-import cv2 --hidden-import PIL --hidden-import numpy `
  app.py
```

## 资源路径适配

`app.py` 已内置资源路径适配函数，支持打包临时目录与运行目录：

```python
def resource_path(relpath: str) -> str:
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
```

使用示例：

```python
weights = resource_path("checkpoints/best_yolov8_detect.pt")
data_yaml = resource_path("configs/detect.yaml")
```

## 输出与运行

- 目录模式：在 `dist/CrackDetector/CrackDetector.exe` 运行；可直接放置或替换 `checkpoints/*.pt`、`configs/*.yaml`
- 单文件模式：在 `dist/CrackDetector.exe` 运行；建议将权重文件放在与 exe 同一目录，或通过 `--add-data` 内嵌

## 分发建议

- 将 `CrackDetector.exe` 与 `checkpoints/`、`configs/` 一并打包分发（或使用单文件模式内嵌）
- 如果用户需要自行替换权重，只需把新权重文件放到 exe 同目录或 `dist/CrackDetector/checkpoints/`
