#!/bin/bash

# Find the latest model versions
STNPP_MODEL=$(ls -t trained_models/joint/stnpp_joint_best_v*.pt | head -1)
QAL_MODEL=$(ls -t trained_models/joint/qal_joint_best_v*.pt | head -1)

echo "Using models:"
echo "STNPP: $STNPP_MODEL"
echo "QAL: $QAL_MODEL"

# Check if model exists
if [ ! -f "$STNPP_MODEL" ]; then
    echo "Error: Model file not found: $STNPP_MODEL"
    exit 1
fi

# Define video path
VIDEO_PATH="datasets/MOTChallenge/MOT16/test/MOT16-01/video.mp4"

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    echo "Checking for img1 directory..."
    
    # Check if img1 directory exists
    IMG_DIR="datasets/MOTChallenge/MOT16/test/MOT16-01/img1"
    if [ ! -d "$IMG_DIR" ]; then
        echo "Error: Neither video file nor img1 directory found"
        exit 1
    fi
    
    echo "Found img1 directory. Converting images to video..."
    # Create video from images
    ffmpeg -y -framerate 30 -i "$IMG_DIR/%06d.jpg" -c:v libx264 -pix_fmt yuv420p "$VIDEO_PATH"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create video from images"
        exit 1
    fi
    echo "Successfully created video file"
fi

# Create output directory
mkdir -p results/compression_test

# Step 1: Compare compression methods
echo "Step 1: Comparing video compression methods..."
python compare_compression_methods.py \
    --input_video "$VIDEO_PATH" \
    --model_path "$STNPP_MODEL" \
    --output_dir results/compression_test

# Check if compression comparison was successful
if [ $? -ne 0 ]; then
    echo "Error: Compression comparison failed."
    exit 1
fi

echo "Compression comparison completed successfully."