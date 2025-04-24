@echo off
echo ======================================================
echo VCM-ST-NPP Setup Script for Windows
echo ======================================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    goto :error
)

echo Checking pip installation...
pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: pip not found. Please make sure pip is installed with Python.
    goto :error
)

echo Checking if virtual environment exists...
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate virtual environment.
    goto :error
)

echo Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Failed to upgrade pip. Continuing with installation...
)

echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install dependencies.
    goto :error
)

echo Checking for CUDA...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

echo Checking for FFmpeg...
where ffmpeg >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Warning: FFmpeg not found in PATH. Some video processing features may not work.
    echo Please install FFmpeg from https://ffmpeg.org/download.html
) else (
    echo FFmpeg found in PATH.
)

echo Creating necessary directories...
if not exist checkpoints mkdir checkpoints
if not exist results mkdir results
if not exist data mkdir data

echo Checking for model checkpoints...
if not exist checkpoints\detection_model.pth (
    echo Dummy detection model checkpoint not found. Creating...
    python -c "import torch; model = {'state_dict': {}, 'metadata': {'task': 'detection'}}; torch.save(model, 'checkpoints/detection_model.pth')"
)
if not exist checkpoints\segmentation_model.pth (
    echo Dummy segmentation model checkpoint not found. Creating...
    python -c "import torch; model = {'state_dict': {}, 'metadata': {'task': 'segmentation'}}; torch.save(model, 'checkpoints/segmentation_model.pth')"
)
if not exist checkpoints\tracking_model.pth (
    echo Dummy tracking model checkpoint not found. Creating...
    python -c "import torch; model = {'state_dict': {}, 'metadata': {'task': 'tracking'}}; torch.save(model, 'checkpoints/tracking_model.pth')"
)

echo Verifying model loading...
python -c "from models.combined_model import CombinedModel; model = CombinedModel(task_type='detection'); print('Model successfully created')"
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Failed to load model. Please check if all dependencies are installed correctly.
) else (
    echo Model loading verification successful.
)

echo ======================================================
echo VCM-ST-NPP Setup Complete
echo ======================================================
echo.
echo To activate the environment, run:
echo     venv\Scripts\activate
echo.
echo To train a model, run:
echo     python train.py --dataset dummy --task_type detection --dry-run
echo.
echo To evaluate a model, run:
echo     python evaluate.py --dataset dummy --task detection --checkpoint checkpoints/detection_model.pth
echo ======================================================

goto :eof

:error
echo.
echo Setup failed. Please check the error messages above.
exit /b 1 