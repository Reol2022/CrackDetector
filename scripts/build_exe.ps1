#requires -version 5.1
param(
    [switch]$OneFile
)

$ErrorActionPreference = 'Stop'

Write-Host "[CrackDetector] 设置打包环境..." -ForegroundColor Cyan

python -V | Out-Null

try {
    pip show pyinstaller | Out-Null
} catch {
    Write-Host "安装 PyInstaller..." -ForegroundColor Yellow
    pip install pyinstaller
}

Write-Host "开始打包 GUI 为 EXE..." -ForegroundColor Cyan

if ($OneFile) {
    # 单文件模式：生成 dist/CrackDetector.exe（仅打包存在的资源）
    $addDataArgs = @()
    function AddDataIfExists($src, $dst) {
        if (Test-Path $src) {
            $script:addDataArgs += @('--add-data', "$src;$dst")
        }
    }

    AddDataIfExists "configs/detect.yaml" "configs"
    AddDataIfExists "configs/classify.yaml" "configs"
    AddDataIfExists "configs/seg.yaml" "configs"

    AddDataIfExists "checkpoints/best_yolov8.pt" "checkpoints"
    AddDataIfExists "checkpoints/best_yolov8_detect.pt" "checkpoints"
    AddDataIfExists "checkpoints/best_yolov8_seg.pt" "checkpoints"

    AddDataIfExists "yolov8n.pt" "."
    AddDataIfExists "yolov8n-cls.pt" "."
    AddDataIfExists "yolov8n-seg.pt" "."

    $argsList = @(
        '--noconfirm','--clean',
        '--name','CrackDetector',
        '--onefile','--windowed'
    ) + $addDataArgs + @(
        '--hidden-import','ultralytics',
        '--hidden-import','torch',
        '--hidden-import','torchvision',
        '--hidden-import','cv2',
        '--hidden-import','PIL',
        '--hidden-import','numpy',
        'app.py'
    )

    & pyinstaller @argsList
} else {
    # 文件夹模式：使用 spec，生成 dist/CrackDetector/
    pyinstaller --noconfirm --clean crackdetector.spec
}

Write-Host "打包完成。输出位于: dist" -ForegroundColor Green
