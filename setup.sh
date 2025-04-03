#!/bin/bash

# Create a Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv vcm_env

# Activate the virtual environment
echo "Activating virtual environment..."
source vcm_env/bin/activate

# Install dependencies
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install ffmpeg if not already installed
if ! command -v ffmpeg &> /dev/null
then
    echo "Installing ffmpeg..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ffmpeg
    else
        echo "Please install ffmpeg manually from https://ffmpeg.org/download.html"
    fi
else
    echo "ffmpeg is already installed"
fi

echo "Setup complete! Activate the environment with: source vcm_env/bin/activate" 