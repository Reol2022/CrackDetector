# 问答与排查指南（FAQ）

汇总在不同使用情景下常见的问题与解决方法，覆盖安装、推理、依赖与环境等方面。

## 安装与环境

- 无法识别命令（如 `crackdetector-detect` 未找到）
  - 原因：未正确安装或环境变量未更新
  - 解决：确保在虚拟环境或全局执行 `pip install -e .` 或 `pip install .`；Windows 需保证 `Python\Scripts` 在 PATH 中，或重启终端

- 未安装 Ultralytics
  - 现象：`ImportError: No module named ultralytics`
  - 解决：执行 `pip install ultralytics`

- 未安装/版本不匹配的 PyTorch
  - 现象：分类或 YOLO 推理时报 `ImportError` 或 CUDA 相关错误
  - 解决：安装与平台匹配的 PyTorch（CPU 版更轻量）：
    - CPU 版：`pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
    - GPU 版：参考官方指引选择 CUDA 对应版本

- OpenCV/Pillow/Numpy 导入错误
  - 解决：`pip install opencv-python pillow numpy`

- 虚拟环境未激活导致依赖混乱
  - 解决：Windows PowerShell：`python -m venv venv; venv\Scripts\Activate.ps1`

## 路径与文件

- 权重路径不一致或不存在
  - 现象：`FileNotFoundError: checkpoints/best_yolov8_detect.pt`
  - 解决：
    - 检测 CLI/API 传入 `--weights` 或将权重放置在 `checkpoints/` 下
    - 分类传入 `--weights` 或将 `best_model.pth` 放置在 `checkpoints/`

- 路径包含空格或中文导致读取失败
  - 解决：用引号包裹路径参数（例如 `"E:\我的项目\..."`），或将资源移动到 ASCII 路径

- YAML/配置路径错误
  - 现象：找不到 `configs/detect.yaml`
  - 解决：确保使用仓库内配置或自行提供绝对路径；pip 包不内置配置文件

## 推理与性能

- GPU 不可用或 CUDA 报错
  - 解决：传入 `--device cpu` 使用 CPU 推理；若需 GPU，安装匹配 CUDA 的 PyTorch 并确保驱动正常

- 半精度（`--half`）推理失败
  - 原因：CPU 或不支持的 GPU
  - 解决：移除 `--half` 或切换到支持的 GPU

- 推理输出未保存或找不到结果
  - 现象：未生成 `runs/detect/...`
  - 解决：检测 CLI/API 默认保存到 `runs/detect/<name>`；确认有写权限或显式设置 `--project`

- 结果偏差或置信度异常
  - 可能原因：训练与推理时预处理不一致、权重不匹配
  - 解决：检查训练集归一化与尺寸，确保权重与任务一致

## 分类权重兼容性

- `RuntimeError: size mismatch` 或 `Missing key(s) in state_dict`
  - 原因：权重头部类别数不同或保存格式不同（包裹在 `state_dict`）
  - 解决：
    - 使用相同类别数训练的权重
    - 代码中已兼容 `state["state_dict"]`；若仍报错，请重新导出正确的 `state_dict`

## 版本与依赖冲突

- Ultralytics 版本不兼容
  - 现象：API 变更导致 `predict` 参数报错
  - 解决：升级到兼容版本：`pip install -U ultralytics`

- Torch/Torchvision 版本不匹配
  - 现象：导入时报版本不兼容错误
  - 解决：参照 PyTorch 官网选择匹配版本；或统一安装 CPU 版（版本自动匹配）

## 其他建议

- 大体积与启动慢（与 exe 打包相关）
  - 建议：优先使用 pip 包（本指南），按需安装依赖与外置权重，避免单文件 exe 膨胀

- 提交问题时请附带信息
  - Python 版本、操作系统、包版本（`pip freeze | findstr crackdetector ultralytics torch`）、报错堆栈、复现步骤

---

如果这里未覆盖你的问题，请在 Issue 中附详细信息，我会协助定位与修复。
