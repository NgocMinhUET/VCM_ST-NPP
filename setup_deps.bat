@echo off
echo Setting up environment for VCM_ST-NPP...

:: Check if pip is available
where pip >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: pip is not installed or not in PATH
    exit /b 1
)

:: Install required packages
echo Installing required packages...
pip install -r requirements.txt

:: Check if sklearn is available after installation
python -c "import sklearn" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Warning: scikit-learn installation may have failed. Trying alternative install...
    pip install scikit-learn
)

:: Check for GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo Setup complete! You can now run the training script.
echo Example usage:
echo python train.py --dataset path\to\dataset --task_type tracking --seq_length 5 --epochs 50 --batch_size 4 --lr 1e-4 --qp 30 