# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('configs/detect.yaml', 'configs'), ('configs/classify.yaml', 'configs'), ('configs/seg.yaml', 'configs'), ('checkpoints/best_yolov8.pt', 'checkpoints'), ('checkpoints/best_yolov8_detect.pt', 'checkpoints'), ('yolov8n.pt', '.'), ('yolov8n-cls.pt', '.')],
    hiddenimports=['ultralytics', 'torch', 'torchvision', 'cv2', 'PIL', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='CrackDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
