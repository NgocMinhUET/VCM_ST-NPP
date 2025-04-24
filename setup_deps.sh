#!/bin/bash

echo "Setting up environment for VCM_ST-NPP..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Check if sklearn is available after installation
python -c "import sklearn" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Warning: scikit-learn installation may have failed. Trying alternative install..."
    pip install scikit-learn
fi

# Check for GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Setup complete! You can now run the training script."
echo "Example usage:"
echo "python train.py --dataset /path/to/dataset --task_type tracking --seq_length 5 --epochs 50 --batch_size 4 --lr 1e-4 --qp 30" 