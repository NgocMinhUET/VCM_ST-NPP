"""
Codec utilities for task-aware video compression.

This module provides utility functions for encoding and decoding video
using external codecs like FFmpeg.
"""

import os
import subprocess
import platform
from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path
import logging

def compress_with_ffmpeg(
    input_path: str, 
    output_path: str, 
    qp: int,
    overwrite: bool = False,
    ffmpeg_path: Optional[str] = None,
    additional_args: Optional[List[str]] = None,
    verbose: bool = False
) -> bool:
    """
    Compress a video file using FFmpeg with libx265 codec.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save compressed output
        qp: Quantization parameter (0-51, lower is higher quality)
        overwrite: Whether to overwrite output file if it exists
        ffmpeg_path: Optional custom path to ffmpeg executable
        additional_args: Optional list of additional arguments to pass to ffmpeg
        verbose: Whether to print verbose output
        
    Returns:
        True if compression successful, False otherwise
    """
    try:
        # Validate inputs
        if not os.path.exists(input_path):
            print(f"Error: Input file does not exist: {input_path}")
            return False
        
        if os.path.exists(output_path) and not overwrite:
            print(f"Error: Output file already exists: {output_path}")
            return False
        
        if not 0 <= qp <= 51:
            print(f"Warning: QP value {qp} outside normal range (0-51), results may be unexpected")
        
        # Create directory for output file if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Determine ffmpeg command
        if ffmpeg_path is None:
            ffmpeg_cmd = "ffmpeg"
        else:
            ffmpeg_cmd = ffmpeg_path
        
        # Build command
        cmd = [
            ffmpeg_cmd,
            "-i", input_path,
            "-c:v", "libx265",
            "-qp", str(qp),
        ]
        
        # Add additional arguments if provided
        if additional_args:
            cmd.extend(additional_args)
        
        # Add output path and overwrite flag
        if overwrite:
            cmd.extend(["-y", output_path])
        else:
            cmd.append(output_path)
        
        # Print command if verbose
        if verbose:
            print(f"Running command: {' '.join(cmd)}")
        
        # Run the ffmpeg command
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
            universal_newlines=True
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        # Check if process was successful
        if process.returncode != 0:
            print(f"Error compressing video: {stderr}")
            return False
        
        # Check if output file was created and has size > 0
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print("Error: Output file was not created or is empty")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error during compression: {str(e)}")
        return False


def decompress_with_ffmpeg(
    input_path: str, 
    output_path: str,
    ffmpeg_path: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = False
) -> bool:
    """
    Decompress a video file using FFmpeg.
    
    Args:
        input_path: Path to compressed video file
        output_path: Path to save decompressed output
        ffmpeg_path: Optional custom path to ffmpeg executable
        overwrite: Whether to overwrite output file if it exists
        verbose: Whether to print verbose output
        
    Returns:
        True if decompression successful, False otherwise
    """
    try:
        # Validate inputs
        if not os.path.exists(input_path):
            print(f"Error: Input file does not exist: {input_path}")
            return False
        
        if os.path.exists(output_path) and not overwrite:
            print(f"Error: Output file already exists: {output_path}")
            return False
        
        # Create directory for output file if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Determine ffmpeg command
        if ffmpeg_path is None:
            ffmpeg_cmd = "ffmpeg"
            else:
            ffmpeg_cmd = ffmpeg_path
        
        # Build command
        cmd = [
            ffmpeg_cmd,
            "-i", input_path,
            "-c:v", "rawvideo",  # Use raw video output
        ]
        
        # Add output path and overwrite flag
        if overwrite:
            cmd.extend(["-y", output_path])
        else:
            cmd.append(output_path)
        
        # Print command if verbose
        if verbose:
            print(f"Running command: {' '.join(cmd)}")
        
        # Run the ffmpeg command
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
            universal_newlines=True
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        # Check if process was successful
        if process.returncode != 0:
            print(f"Error decompressing video: {stderr}")
            return False
        
        # Check if output file was created and has size > 0
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print("Error: Output file was not created or is empty")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error during decompression: {str(e)}")
        return False


def calculate_bitrate(file_path: str, ffprobe_path: Optional[str] = None) -> Optional[float]:
    """
    Calculate the bitrate of a video file in bits per second.
    
    Args:
        file_path: Path to the video file
        ffprobe_path: Optional custom path to ffprobe executable
        
    Returns:
        Bitrate in bits per second or None if calculation failed
    """
    try:
        # Validate input
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return None
        
        # Determine ffprobe command
        if ffprobe_path is None:
            ffprobe_cmd = "ffprobe"
        else:
            ffprobe_cmd = ffprobe_path
        
        # Build command to get bitrate information
    cmd = [
            ffprobe_cmd,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=bit_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        
        # Run the ffprobe command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Check if process was successful
        if result.returncode != 0:
            print(f"Error calculating bitrate: {result.stderr}")
            return None
        
        # Parse bitrate from output
        bitrate_str = result.stdout.strip()
        
        # If bitrate is not directly available, calculate from duration and size
        if not bitrate_str or bitrate_str == "N/A":
            # Get file size in bits
            file_size_bits = os.path.getsize(file_path) * 8
            
            # Get duration
            duration_cmd = [
                ffprobe_cmd,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ]
            
            duration_result = subprocess.run(
                duration_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if duration_result.returncode != 0:
                print(f"Error calculating duration: {duration_result.stderr}")
                return None
            
            duration_str = duration_result.stdout.strip()
            try:
                duration = float(duration_str)
                bitrate = file_size_bits / duration
                return bitrate
            except (ValueError, ZeroDivisionError):
                print("Error parsing duration or calculating bitrate")
                return None
        else:
            try:
                return float(bitrate_str)
            except ValueError:
                print(f"Error parsing bitrate: {bitrate_str}")
                return None
        
    except Exception as e:
        print(f"Error calculating bitrate: {str(e)}")
        return None


def check_ffmpeg_installed() -> bool:
    """
    Check if FFmpeg is installed and available in the system path.
        
    Returns:
        True if FFmpeg is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return result.returncode == 0
    except Exception:
        return False


def get_video_info(file_path: str, ffprobe_path: Optional[str] = None) -> Optional[Dict]:
    """
    Get information about a video file using ffprobe.
    
    Args:
        file_path: Path to the video file
        ffprobe_path: Optional custom path to ffprobe executable
        
    Returns:
        Dictionary with video information or None if retrieval failed
    """
    try:
        # Validate input
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return None
        
        # Determine ffprobe command
        if ffprobe_path is None:
            ffprobe_cmd = "ffprobe"
        else:
            ffprobe_cmd = ffprobe_path
        
        # Build command to get video information in JSON format
        cmd = [
            ffprobe_cmd,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path
        ]
        
        # Run the ffprobe command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Check if process was successful
        if result.returncode != 0:
            print(f"Error getting video info: {result.stderr}")
            return None
        
        # Parse JSON output
        import json
        info = json.loads(result.stdout)
        
        # Extract relevant information
        video_info = {}
        
        # Get format information
        if "format" in info:
            format_info = info["format"]
            video_info["format"] = format_info.get("format_name")
            video_info["duration"] = float(format_info.get("duration", 0))
            video_info["size"] = int(format_info.get("size", 0))
            video_info["bit_rate"] = int(format_info.get("bit_rate", 0))
        
        # Get video stream information
        if "streams" in info:
            for stream in info["streams"]:
                if stream.get("codec_type") == "video":
                    video_info["codec"] = stream.get("codec_name")
                    video_info["width"] = stream.get("width")
                    video_info["height"] = stream.get("height")
                    video_info["fps"] = eval(stream.get("r_frame_rate", "0/1"))
                    video_info["frames"] = int(stream.get("nb_frames", 0))
                    break
        
        return video_info
        
    except Exception as e:
        print(f"Error getting video info: {str(e)}")
        return None


# Test the functions
if __name__ == "__main__":
    # Check if FFmpeg is installed
    if check_ffmpeg_installed():
        print("FFmpeg is installed")
        
        # Test compress function with a sample file
        input_file = "sample.mp4"
        output_file = "compressed.mp4"
        
        if os.path.exists(input_file):
            print(f"Compressing {input_file} to {output_file} with QP=30")
            success = compress_with_ffmpeg(input_file, output_file, qp=30, overwrite=True, verbose=True)
            
            if success:
                print("Compression successful")
                
                # Get video info
                info = get_video_info(output_file)
                if info:
                    print(f"Video info: {info}")
                
                # Calculate bitrate
                bitrate = calculate_bitrate(output_file)
                if bitrate:
                    print(f"Bitrate: {bitrate/1000:.2f} kbps")
            else:
                print("Compression failed")
        else:
            print(f"Sample file {input_file} not found")
    else:
        print("FFmpeg is not installed") 