@echo off
echo Creating Python virtual environment...
python -m venv vcm_env

echo Activating virtual environment...
call vcm_env\Scripts\activate

echo Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete! Activate the environment with: vcm_env\Scripts\activate

echo.
echo NOTE: For Windows, please install ffmpeg manually:
echo 1. Download from https://ffmpeg.org/download.html
echo 2. Extract the files
echo 3. Add the bin folder to your PATH environment variable 