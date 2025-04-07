"""
Common utility functions to be used across modules.
This file centralizes frequently used functionality to promote code reuse and reduce duplication.
"""

import os
import torch
import numpy as np
import random
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List, Callable


def setup_logging(log_dir: str, log_name: str = "main.log", console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up logging configuration with console and file outputs.
    
    Args:
        log_dir: Directory to save log files
        log_name: Name of the log file
        console_level: Logging level for console output
        file_level: Logging level for file output
        
    Returns:
        Configured logger instance
    """
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("st_npp")
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Determine the best available device (CUDA or CPU).
    
    Returns:
        PyTorch device object
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_directory_structure(dirs: List[str]) -> None:
    """
    Create multiple directories at once.
    
    Args:
        dirs: List of directories to create
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


class Timer:
    """Simple timer class for tracking execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        self.end_time = time.time()
        return self.elapsed_time()
    
    def elapsed_time(self) -> float:
        """
        Get elapsed time without stopping the timer.
        
        Returns:
            Elapsed time in seconds or -1 if timer wasn't started
        """
        if self.start_time is None:
            return -1
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time


def load_file_list(directory: str, extensions: List[str], recursive: bool = False) -> List[str]:
    """
    Load list of files with specific extensions from a directory.
    
    Args:
        directory: Directory path to search
        extensions: List of file extensions to include (e.g., ['.jpg', '.png'])
        recursive: Whether to search subdirectories recursively
        
    Returns:
        List of file paths
    """
    path = Path(directory)
    files = []
    
    # Make sure extensions start with a dot
    exts = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    # Search pattern based on recursive flag
    pattern = '**/*' if recursive else '*'
    
    # Find all matching files
    for ext in exts:
        files.extend([str(f) for f in path.glob(f"{pattern}{ext}")])
    
    return sorted(files)


def calculate_psnr(original: torch.Tensor, compressed: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original: Original image tensor
        compressed: Compressed/reconstructed image tensor
        max_val: Maximum value of the signal
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def recursive_to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively move data structures containing tensors to the specified device.
    
    Args:
        data: Input data structure (can be tensor, list, tuple, dict)
        device: Target device
        
    Returns:
        Data structure with tensors moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [recursive_to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device) for k, v in data.items()}
    else:
        return data


def convert_size_bytes(size_bytes: int) -> str:
    """
    Convert size in bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_name[i]}"


def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.
    
    Args:
        n: Number to check
        
    Returns:
        True if n is a power of two, False otherwise
    """
    return n > 0 and (n & (n - 1)) == 0


def calculate_bitrate(encoded_data: bytes, frames: int, resolution: Tuple[int, int], fps: float = 30.0) -> float:
    """
    Calculate bitrate in bits per second (bps) from encoded data.
    
    Args:
        encoded_data: Encoded data in bytes
        frames: Number of frames
        resolution: Frame resolution as (height, width)
        fps: Frames per second
        
    Returns:
        Bitrate in bits per second
    """
    total_bits = len(encoded_data) * 8
    duration_seconds = frames / fps
    return total_bits / duration_seconds


def calculate_bpp(encoded_data: bytes, resolution: Tuple[int, int], frames: int) -> float:
    """
    Calculate bits per pixel (bpp) from encoded data.
    
    Args:
        encoded_data: Encoded data in bytes
        resolution: Frame resolution as (height, width)
        frames: Number of frames
        
    Returns:
        Bits per pixel
    """
    total_bits = len(encoded_data) * 8
    total_pixels = resolution[0] * resolution[1] * frames
    return total_bits / total_pixels 