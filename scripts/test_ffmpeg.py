#!/usr/bin/env python3
"""
Test script to verify that FFmpeg is properly accessible
and the proxy network can encode/decode frames.
"""

import os
import sys
import subprocess
import numpy as np
import torch
import cv2
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def test_ffmpeg_path():
    """Test if ffmpeg executable can be found and executed."""
    print("Testing FFmpeg path...")
    
    # Find ffmpeg executable
    ffmpeg_path = os.path.join(parent_dir, 'ffmpeg', 'bin', 'ffmpeg.exe')
    
    # Verify that ffmpeg exists
    if not os.path.exists(ffmpeg_path):
        print(f"ERROR: FFmpeg executable not found at {ffmpeg_path}.")
        print("Please run setup_ffmpeg.bat first.")
        return False
    
    print(f"FFmpeg found at: {ffmpeg_path}")
    
    # Test executing ffmpeg
    try:
        result = subprocess.run([ffmpeg_path, "-version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               check=True)
        print("FFmpeg version output:")
        print(result.stdout.decode()[:200] + "..." if len(result.stdout) > 200 else result.stdout.decode())
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to execute FFmpeg: {e}")
        if hasattr(e, 'stderr'):
            print(f"STDERR: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error when executing FFmpeg: {e}")
        return False

def test_simple_encode_decode():
    """Test a simple encoding and decoding operation with FFmpeg."""
    print("\nTesting simple encode/decode operation...")
    
    # Create a temporary directory
    temp_dir = os.path.join(parent_dir, 'temp_test')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a simple test frame
    width, height = 320, 240
    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some color gradient
    for y in range(height):
        for x in range(width):
            test_frame[y, x, 0] = x * 255 // width  # Red
            test_frame[y, x, 1] = y * 255 // height  # Green
            test_frame[y, x, 2] = 128  # Blue
    
    # Save the test frame
    frame_path = os.path.join(temp_dir, "test_frame.png")
    cv2.imwrite(frame_path, test_frame)
    
    # Find ffmpeg executable
    ffmpeg_path = os.path.join(parent_dir, 'ffmpeg', 'bin', 'ffmpeg.exe')
    
    # Encode the frame
    input_path = frame_path
    output_path = os.path.join(temp_dir, "test_encoded.mp4")
    
    try:
        # Encode command
        cmd = [
            ffmpeg_path, "-y",
            "-i", input_path,
            "-c:v", "libx265",
            "-preset", "medium",
            "-x265-params", "qp=27",
            output_path
        ]
        print(f"Running encode command: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # Check if the encoded file exists
        if not os.path.exists(output_path):
            print(f"ERROR: Encoded file was not created at {output_path}")
            return False
        
        # Get encoded file size
        encoded_size = os.path.getsize(output_path)
        print(f"Encoded file size: {encoded_size} bytes")
        
        # Decode back
        decoded_path = os.path.join(temp_dir, "test_decoded.png")
        cmd = [
            ffmpeg_path, "-y",
            "-i", output_path,
            decoded_path
        ]
        print(f"Running decode command: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # Check if the decoded file exists
        if not os.path.exists(decoded_path):
            print(f"ERROR: Decoded file was not created at {decoded_path}")
            return False
        
        print(f"Successfully encoded and decoded test frame")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"ERROR: FFmpeg command failed: {e}")
        print(f"STDOUT: {e.stdout.decode() if hasattr(e, 'stdout') else 'N/A'}")
        print(f"STDERR: {e.stderr.decode() if hasattr(e, 'stderr') else 'N/A'}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error during encoding/decoding: {e}")
        return False
    finally:
        # Clean up
        try:
            for file in [frame_path, output_path, os.path.join(temp_dir, "test_decoded.png")]:
                if os.path.exists(file):
                    os.remove(file)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("FFmpeg Test Script")
    print("=" * 60)
    
    # Test FFmpeg path
    if not test_ffmpeg_path():
        print("\nFFmpeg path test FAILED!")
        return False
    
    # Test simple encode/decode
    if not test_simple_encode_decode():
        print("\nSimple encode/decode test FAILED!")
        return False
    
    print("\n" + "=" * 60)
    print("All tests PASSED! FFmpeg is properly configured.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 