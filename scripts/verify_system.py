#!/usr/bin/env python3
"""
System Verification Script for ST-NPP Project

This script verifies the system setup for the ST-NPP project, including:
1. Python version
2. PyTorch installation and CUDA availability
3. FFmpeg installation with codec support
4. Required Python packages
5. Dataset directories
6. Disk space

Run this script after installation to ensure your system is properly configured.
"""

import os
import sys
import platform
import shutil
import subprocess
import importlib
from pathlib import Path
import pkg_resources
import argparse


def print_section(title):
    """Print a section header with formatting."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_status(name, status, message=""):
    """Print a status message with color."""
    status_str = "[ \033[92mOK\033[0m ]" if status else "[\033[91mFAIL\033[0m]"
    print(f"{status_str} {name:<30} {message}")
    return status


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    py_version = platform.python_version()
    major, minor, _ = py_version.split('.')
    status = int(major) >= 3 and int(minor) >= 8
    
    message = f"Found Python {py_version}"
    if not status:
        message += " (Required: Python 3.8+)"
    
    return print_status("Python Version", status, message)


def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch
        torch_version = torch.__version__
        
        has_cuda = torch.cuda.is_available()
        cuda_message = "with CUDA" if has_cuda else "without CUDA (GPU acceleration unavailable)"
        
        status = True
        message = f"Found PyTorch {torch_version} {cuda_message}"
        
        if has_cuda:
            cuda_version = torch.version.cuda
            message += f" (CUDA {cuda_version})"
            
            # Check GPU memory
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert to GB
                message += f"\n                               GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)"
        
        return print_status("PyTorch", status, message)
        
    except ImportError:
        return print_status("PyTorch", False, "Not installed (Required)")


def check_ffmpeg():
    """Check if FFmpeg is installed and supports required codecs."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                encoding="utf-8")
        
        if result.returncode != 0:
            return print_status("FFmpeg", False, "Installed but returned an error")
        
        # Get version line
        version_line = result.stdout.split('\n')[0]
        
        # Check codec support
        encoders_result = subprocess.run(["ffmpeg", "-encoders"],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        encoding="utf-8")
        
        has_h264 = "libx264" in encoders_result.stdout
        has_h265 = "libx265" in encoders_result.stdout or "hevc" in encoders_result.stdout.lower()
        has_vp9 = "libvpx-vp9" in encoders_result.stdout
        
        codecs_status = []
        if has_h264:
            codecs_status.append("H.264 ✓")
        else:
            codecs_status.append("H.264 ✗")
            
        if has_h265:
            codecs_status.append("H.265/HEVC ✓")
        else:
            codecs_status.append("H.265/HEVC ✗")
            
        if has_vp9:
            codecs_status.append("VP9 ✓")
        else:
            codecs_status.append("VP9 ✗")
        
        message = f"{version_line}\n                               Codecs: {', '.join(codecs_status)}"
        
        # Overall status - require at least H.264 and H.265
        status = has_h264 and has_h265
        if not status:
            message += "\n                               Missing required codec support for H.264 and/or H.265"
        
        return print_status("FFmpeg", status, message)
        
    except FileNotFoundError:
        return print_status("FFmpeg", False, "Not installed or not in PATH")


def check_required_packages():
    """Check if all required packages are installed."""
    required_packages = [
        "numpy",
        "opencv-python",
        "tqdm",
        "matplotlib",
        "tensorboard",
        "scipy",
        "pillow",
        "motmetrics",
    ]
    
    all_installed = True
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
        except pkg_resources.DistributionNotFound:
            all_installed = False
            missing_packages.append(package)
    
    message = "All required packages installed"
    if not all_installed:
        message = f"Missing packages: {', '.join(missing_packages)}"
    
    return print_status("Required Packages", all_installed, message)


def check_directories():
    """Check if required directories exist and create them if needed."""
    required_dirs = [
        "datasets",
        "datasets/MOTChallenge",
        "datasets/coco_video",
        "datasets/kitti_semantic",
        "trained_models",
        "trained_models/proxy",
        "trained_models/stnpp",
        "trained_models/qal",
        "trained_models/joint",
        "results",
        "logs",
    ]
    
    created_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            created_dirs.append(directory)
    
    message = "All required directories exist"
    if created_dirs:
        message = f"Created directories: {', '.join(created_dirs)}"
    
    return print_status("Project Directories", True, message)


def check_disk_space():
    """Check available disk space."""
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    try:
        total, used, free = shutil.disk_usage(project_dir)
        
        # Convert to GB
        total_gb = total / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        
        status = free_gb >= 50  # Require at least 50GB free
        
        message = f"Total: {total_gb:.1f} GB, Free: {free_gb:.1f} GB"
        if not status:
            message += " (Recommended: at least 50GB free space for datasets and models)"
        
        return print_status("Disk Space", status, message)
    
    except Exception as e:
        return print_status("Disk Space", False, f"Error checking disk space: {e}")


def check_gpu_compute_capability():
    """Check GPU compute capability if CUDA is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return print_status("GPU Compute Capability", False, "CUDA not available")
        
        compute_capability = {}
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            major, minor = torch.cuda.get_device_capability(i)
            compute_capability[i] = (major, minor)
        
        # Require compute capability 3.5 or higher for modern PyTorch
        all_supported = all(major >= 3 and (major > 3 or minor >= 5) for major, minor in compute_capability.values())
        
        message = ", ".join([f"GPU {i}: {device_name} (Compute {major}.{minor})" 
                            for i, ((major, minor), device_name) in 
                            enumerate(zip(compute_capability.values(), 
                                        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))])
        
        if not all_supported:
            message += "\nSome GPUs have compute capability below 3.5, which may not be fully supported"
        
        return print_status("GPU Compute Capability", all_supported, message)
    
    except ImportError:
        return print_status("GPU Compute Capability", False, "PyTorch not installed")
    except Exception as e:
        return print_status("GPU Compute Capability", False, f"Error checking compute capability: {e}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify system setup for ST-NPP project")
    parser.add_argument("--create-dirs", action="store_true", help="Create missing directories")
    
    return parser.parse_args()


def main():
    """Main function to run all checks."""
    args = parse_args()
    
    print_section("ST-NPP System Verification")
    
    print("Checking system configuration...\n")
    
    python_status = check_python_version()
    pytorch_status = check_pytorch()
    ffmpeg_status = check_ffmpeg()
    packages_status = check_required_packages()
    
    if args.create_dirs:
        dirs_status = check_directories()
    else:
        dirs_status = True  # Skip directory check if not requested
    
    space_status = check_disk_space()
    gpu_status = check_gpu_compute_capability()
    
    # Summarize results
    print_section("Summary")
    
    all_checks_passed = all([
        python_status,
        pytorch_status,
        ffmpeg_status,
        packages_status,
        dirs_status,
        space_status,
        # gpu_status is not critical
    ])
    
    if all_checks_passed:
        print("\033[92mAll critical checks passed! Your system is ready for ST-NPP.\033[0m")
    else:
        print("\033[91mSome checks failed. Please address the issues before proceeding.\033[0m")
    
    # Recommendations
    print("\nRecommendations:")
    
    if not pytorch_status or not torch.cuda.is_available():
        print("- Install PyTorch with CUDA support for GPU acceleration")
        print("  Visit: https://pytorch.org/get-started/locally/")
    
    if not ffmpeg_status:
        print("- Install FFmpeg with libx264 and libx265 support")
        print("  Visit: https://ffmpeg.org/download.html")
        print("  Or run: scripts/setup_ffmpeg.sh (Linux/macOS) or scripts/setup_ffmpeg.bat (Windows)")
    
    if not packages_status:
        print("- Install missing Python packages:")
        print("  pip install -r requirements.txt")
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main()) 