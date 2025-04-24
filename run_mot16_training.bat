@echo off
echo Task-Aware Video Compression Training with MOT16 Dataset
echo ======================================================

REM Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python and ensure it's in your PATH.
    exit /b 1
)

REM Run the preparation and training script
echo Starting training preparation...
python scripts/prepare_mot16_training.py ^
    --mot_root "D:/NCS/propose/dataset/MOT16" ^
    --output_root "D:/NCS/propose/dataset/processed" ^
    --epochs 50 ^
    --batch_size 4 ^
    --lr 1e-4 ^
    --qp 30 ^
    --task_weight 1.0 ^
    --recon_weight 1.0 ^
    --bitrate_weight 0.1 ^
    --output_dir "./checkpoints/mot16" ^
    --num_workers 4

if %errorlevel% neq 0 (
    echo Error: Training preparation or execution failed.
    exit /b 1
)

echo Training process completed successfully.
exit /b 0 