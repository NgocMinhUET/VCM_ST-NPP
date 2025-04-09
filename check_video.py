#!/usr/bin/env python3
"""
Script to check if the downloaded videos can be opened and display basic information.
"""

import os
import cv2
import argparse
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Check video files in a directory')
    parser.add_argument('--dir', type=str, default='test_data',
                        help='Directory containing video files')
    return parser.parse_args()

def check_video(filepath):
    """
    Open a video file and display basic information.
    
    Args:
        filepath: Path to the video file
    
    Returns:
        bool: True if the video was opened successfully, False otherwise
    """
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {filepath}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read a few frames to ensure the video is valid
    frames_read = 0
    while cap.isOpened() and frames_read < 10:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
    
    cap.release()
    
    print(f"Video: {os.path.basename(filepath)}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Frame count: {frame_count}")
    print(f"  - Duration: {frame_count/fps:.2f} seconds")
    print(f"  - Frames read successfully: {frames_read}")
    print(f"  - Status: {'OK' if frames_read > 0 else 'Error - Could not read frames'}")
    print()
    
    return frames_read > 0

def main():
    args = parse_args()
    
    # Find all video files in the directory
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend(glob(os.path.join(args.dir, f'*{ext}')))
    
    if not video_files:
        print(f"No video files found in directory: {args.dir}")
        return
    
    print(f"Found {len(video_files)} video files in {args.dir}:")
    
    # Check each video file
    success_count = 0
    for video_file in video_files:
        success = check_video(video_file)
        if success:
            success_count += 1
    
    print(f"Summary: {success_count}/{len(video_files)} videos are valid and can be opened.")

if __name__ == "__main__":
    main() 