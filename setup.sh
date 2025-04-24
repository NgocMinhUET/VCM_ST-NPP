#!/bin/bash
# Script cài đặt cho Task-Aware Video Preprocessing

echo "=== Đang cài đặt Task-Aware Video Preprocessing ==="

# Kiểm tra xem conda đã được cài đặt chưa
if ! command -v conda &> /dev/null; then
    echo "Conda không được tìm thấy. Vui lòng cài đặt Miniconda hoặc Anaconda trước."
    exit 1
fi

# Tạo môi trường conda mới
echo "=== Tạo môi trường conda 'tavp' ==="
conda create -n tavp python=3.8 -y

# Kích hoạt môi trường
echo "=== Kích hoạt môi trường tavp ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tavp

# Cài đặt các gói phụ thuộc
echo "=== Cài đặt các gói phụ thuộc ==="
pip install -r requirements.txt

# Cài đặt PyTorch với CUDA nếu có GPU
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU được phát hiện, cài đặt PyTorch với CUDA ==="
    conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -y
else
    echo "=== Không phát hiện GPU, cài đặt PyTorch CPU ==="
    conda install pytorch torchvision cpuonly -c pytorch -y
fi

# Tạo thư mục cần thiết
echo "=== Tạo thư mục cần thiết ==="
mkdir -p checkpoints logs results data

# Kiểm tra xem FFmpeg đã được cài đặt chưa
if ! command -v ffmpeg &> /dev/null; then
    echo "CẢNH BÁO: FFmpeg không được tìm thấy. Cần cài đặt FFmpeg để xử lý video."
    
    # Cài đặt FFmpeg dựa trên hệ điều hành
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Phát hiện Linux. Đang cài đặt FFmpeg..."
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Phát hiện macOS. Cài đặt FFmpeg bằng Homebrew..."
        brew install ffmpeg
    else
        echo "Không thể tự động cài đặt FFmpeg trên hệ điều hành này."
        echo "Vui lòng tải FFmpeg từ https://ffmpeg.org/download.html"
    fi
else
    echo "FFmpeg đã được cài đặt."
fi

echo ""
echo "=== Cài đặt hoàn tất ==="
echo "Để kích hoạt môi trường, chạy: conda activate tavp"
echo "Để huấn luyện mô hình: python train.py --data_dir path/to/data --dataset_type mot --task_type tracking --checkpoint_dir results/tracking --use_quantization"
echo "" 