# 贡献指南

## 分支与命名

- `main`: 稳定版本（ResNet50 分类）
- `yolov8`: 开发分支（检测/分割）
- `feature/<name>`: 特性开发分支，例如 `feature/video-detect`

## PR 流程

1. Fork 仓库并创建分支
2. 遵循代码风格（PEP8，保持一致的命名与文档）
3. 提交前自测（训练脚本、推理脚本能运行）
4. 发起 PR，说明变更点与动机

## 代码规范

- 模型放在 `models/`，配置放在 `configs/`，脚本在 `scripts/`
- 不在推理路径中硬编码绝对路径，使用相对路径或 `resource_path()`
- README 与 docs 同步更新使用方法与配置示例

