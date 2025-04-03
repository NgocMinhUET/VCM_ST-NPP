#!/usr/bin/env python3
"""
System Verification Script

This script performs a comprehensive check of the system to ensure all components
required for the video compression project are properly set up and functioning.
It verifies:
1. Python version and required packages
2. CUDA availability
3. FFmpeg installation
4. Model files
5. Sample videos
6. End-to-end compression test
"""

import os
import sys
import platform
import subprocess
import importlib
import argparse
import shutil
import time
import hashlib
from pathlib import Path
import pkg_resources
import tempfile

# Add project root to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import project-specific modules only when needed
def import_project_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        return None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify system setup for video compression project")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues automatically")
    parser.add_argument("--run_test", action="store_true", help="Run a quick compression test")
    parser.add_argument("--skip_deps", action="store_true", help="Skip dependency checks")
    parser.add_argument("--skip_cuda", action="store_true", help="Skip CUDA checks")
    parser.add_argument("--skip_ffmpeg", action="store_true", help="Skip FFmpeg checks")
    parser.add_argument("--skip_models", action="store_true", help="Skip model file checks")
    parser.add_argument("--skip_videos", action="store_true", help="Skip sample video checks")
    return parser.parse_args()


def print_status(category, status, message, is_error=False, is_warning=False, verbose=False):
    """Print a formatted status message."""
    if is_error:
        icon = "❌"
        color = "\033[91m"  # Red
    elif is_warning:
        icon = "⚠️"
        color = "\033[93m"  # Yellow
    else:
        icon = "✅"
        color = "\033[92m"  # Green
    
    reset = "\033[0m"
    
    # Only color output if on a terminal
    if not sys.stdout.isatty():
        color = ""
        reset = ""
    
    print(f"{color}{icon} {category}: {status}{reset}")
    if message and (verbose or is_error or is_warning):
        print(f"   {message}")


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    current_version = sys.version_info
    required_version = (3, 8)
    
    if current_version >= required_version:
        version_str = f"{current_version.major}.{current_version.minor}.{current_version.micro}"
        return True, f"Python {version_str}", f"Required: >= 3.8, Found: {version_str}"
    else:
        version_str = f"{current_version.major}.{current_version.minor}.{current_version.micro}"
        return False, f"Python {version_str} (too old)", f"Required: >= 3.8, Found: {version_str}"


def check_dependencies():
    """Check if all required packages are installed."""
    req_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements.txt")
    missing = []
    outdated = []
    
    if not os.path.exists(req_file):
        return False, "Missing requirements.txt", "File not found: requirements.txt"
    
    with open(req_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    for req in requirements:
        req_name = req.split('>=')[0] if '>=' in req else req.split('==')[0] if '==' in req else req
        
        try:
            pkg = pkg_resources.get_distribution(req_name)
            # Check version requirement if specified
            if '>=' in req:
                min_version = req.split('>=')[1]
                if pkg.version < min_version:
                    outdated.append(f"{req_name} (found: {pkg.version}, required: >={min_version})")
            elif '==' in req:
                exact_version = req.split('==')[1]
                if pkg.version != exact_version:
                    outdated.append(f"{req_name} (found: {pkg.version}, required: =={exact_version})")
        except pkg_resources.DistributionNotFound:
            missing.append(req_name)
    
    if missing or outdated:
        status = f"{len(requirements) - len(missing) - len(outdated)}/{len(requirements)} packages OK"
        if missing and outdated:
            message = f"Missing: {', '.join(missing)}; Outdated: {', '.join(outdated)}"
        elif missing:
            message = f"Missing: {', '.join(missing)}"
        else:
            message = f"Outdated: {', '.join(outdated)}"
        return False, status, message
    else:
        return True, f"All {len(requirements)} packages installed", "All dependencies meet requirements"


def check_cuda():
    """Check if CUDA is available and which version."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            
            return True, "CUDA available", f"CUDA {cuda_version}, Devices: {device_count} ({device_name})"
        else:
            return False, "CUDA not available", "PyTorch installed but CUDA unavailable. CPU will be used (slower)."
    
    except ImportError:
        return False, "PyTorch not installed", "Cannot check CUDA - PyTorch package missing"


def check_ffmpeg():
    """Check if FFmpeg is installed and configure if necessary."""
    try:
        # Check FFmpeg version
        result = subprocess.run(["ffmpeg", "-version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[0]
            
            # Check for H.265/HEVC and VP9 support
            encoders_result = subprocess.run(["ffmpeg", "-encoders"],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           text=True)
            
            has_hevc = "hevc" in encoders_result.stdout.lower() or "libx265" in encoders_result.stdout.lower()
            has_vp9 = "vp9" in encoders_result.stdout.lower() or "libvpx-vp9" in encoders_result.stdout.lower()
            
            if has_hevc and has_vp9:
                return True, "FFmpeg installed", f"{version_line} with HEVC/VP9 support"
            else:
                missing_codecs = []
                if not has_hevc:
                    missing_codecs.append("HEVC/H.265")
                if not has_vp9:
                    missing_codecs.append("VP9")
                
                return False, "Codec support limited", f"{version_line} but missing codecs: {', '.join(missing_codecs)}"
        else:
            return False, "FFmpeg test failed", f"Error running FFmpeg: {result.stderr}"
    
    except FileNotFoundError:
        return False, "FFmpeg not found", "FFmpeg is not installed or not in PATH"


def setup_ffmpeg(fix=False):
    """Guide user to set up FFmpeg."""
    is_windows = platform.system() == "Windows"
    script_path = os.path.join("scripts", "setup_ffmpeg.bat" if is_windows else "setup_ffmpeg.sh")
    abs_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), script_path)
    
    if not os.path.exists(abs_script_path):
        return False, f"Setup script not found: {script_path}"
    
    # If fix flag is set, attempt to run the setup script
    if fix:
        try:
            print(f"Running FFmpeg setup script: {script_path}")
            
            if is_windows:
                result = subprocess.run([abs_script_path], 
                                      shell=True,
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE,
                                      text=True)
            else:
                # Make sure the script is executable
                os.chmod(abs_script_path, 0o755)
                result = subprocess.run([abs_script_path],
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE,
                                      text=True)
            
            if result.returncode == 0:
                return True, "FFmpeg setup successful"
            else:
                return False, f"FFmpeg setup failed: {result.stderr}"
        
        except Exception as e:
            return False, f"Error running FFmpeg setup: {str(e)}"
    
    # Otherwise, just provide instructions
    if is_windows:
        return False, f"To install FFmpeg, run: {script_path}"
    else:
        return False, f"To install FFmpeg, run: bash {script_path}"


def check_model_files():
    """Check if model files exist and are valid."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dirs = [
        os.path.join(project_dir, "trained_models", "improved_autoencoder"),
        os.path.join(project_dir, "models"),
    ]
    
    model_files = []
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            model_files.extend([
                os.path.join(model_dir, f) 
                for f in os.listdir(model_dir) 
                if f.endswith(('.pt', '.pth', '.ckpt', '.h5', '.model'))
            ])
    
    if model_files:
        size_info = []
        for model_file in model_files[:3]:  # Show info for up to 3 models
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            size_info.append(f"{os.path.basename(model_file)} ({size_mb:.1f} MB)")
        
        more_text = f" + {len(model_files) - 3} more" if len(model_files) > 3 else ""
        return True, f"Found {len(model_files)} model files", f"{', '.join(size_info)}{more_text}"
    else:
        return False, "No model files found", "No .pt, .pth, .ckpt, .h5 or .model files found"


def check_sample_videos():
    """Check for sample videos in the data directory."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dirs = [
        os.path.join(project_dir, "data", "sample_videos"),
        os.path.join(project_dir, "datasets", "sample_videos"),
    ]
    
    video_files = []
    for sample_dir in sample_dirs:
        if os.path.exists(sample_dir):
            video_files.extend([
                os.path.join(sample_dir, f) 
                for f in os.listdir(sample_dir) 
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            ])
    
    # Also check if MOT16 dataset exists
    mot_dirs = [
        os.path.join(project_dir, "data", "MOT16"),
        os.path.join(project_dir, "datasets", "MOT16"),
        os.path.join("D:/NCS/propose/dataset/MOT16"),  # Check the external path mentioned in the code
    ]
    
    mot_exists = any(os.path.exists(d) for d in mot_dirs)
    
    if video_files:
        status = f"Found {len(video_files)} sample videos"
        if mot_exists:
            status += " + MOT16 dataset"
        
        video_info = []
        for video_file in video_files[:3]:  # Show info for up to 3 videos
            size_mb = os.path.getsize(video_file) / (1024 * 1024)
            video_info.append(f"{os.path.basename(video_file)} ({size_mb:.1f} MB)")
        
        more_text = f" + {len(video_files) - 3} more" if len(video_files) > 3 else ""
        message = f"{', '.join(video_info)}{more_text}"
        
        return len(video_files) >= 3, status, message
    else:
        message = "No sample videos found"
        if mot_exists:
            message += " but MOT16 dataset exists"
        return False, "No sample videos", message


def setup_sample_videos(fix=False):
    """Guide user to set up sample videos."""
    script_path = os.path.join("scripts", "download_sample_videos.py")
    abs_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), script_path)
    
    if not os.path.exists(abs_script_path):
        return False, f"Setup script not found: {script_path}"
    
    # If fix flag is set, attempt to run the setup script
    if fix:
        try:
            print(f"Running sample video download script: {script_path}")
            
            result = subprocess.run([sys.executable, abs_script_path],
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True)
            
            if result.returncode == 0:
                return True, "Sample videos downloaded successfully"
            else:
                return False, f"Sample video download failed: {result.stderr}"
        
        except Exception as e:
            return False, f"Error downloading sample videos: {str(e)}"
    
    # Otherwise, just provide instructions
    return False, f"To download sample videos, run: python {script_path}"


def run_quick_compression_test():
    """Run a quick end-to-end compression test."""
    print("Running quick compression test...")
    success = True
    error_msg = ""
    
    # Create a tiny test video file in memory
    try:
        import numpy as np
        import cv2
        import torch
        
        # Check if the improved_autoencoder module exists
        module = import_project_module("improved_autoencoder")
        if module is None:
            return False, "Failed to import improved_autoencoder module"
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a small test video (10 frames)
            frames = []
            for i in range(10):
                # Create a simple gradient frame
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
                frame[:, :, 0] = i * 25  # Varying blue channel
                frame[:, :, 1] = np.arange(0, 128).reshape(-1, 1)  # Horizontal gradient
                frame[:, :, 2] = np.arange(0, 128).reshape(1, -1)  # Vertical gradient
                frames.append(frame)
            
            # Save frames as a test video
            test_video = os.path.join(tmp_dir, "test_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(test_video, fourcc, 30, (128, 128))
            for frame in frames:
                out.write(frame)
            out.release()
            
            # Get the model
            model_dirs = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "trained_models", "improved_autoencoder")
            ]
            
            model_file = None
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    model_files = [
                        os.path.join(model_dir, f) 
                        for f in os.listdir(model_dir) 
                        if f.endswith('.pt') and ("best" in f or "latest" in f)
                    ]
                    if model_files:
                        model_file = model_files[0]
                        break
            
            if model_file is None:
                return False, "No model file found for testing"
            
            # Load the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = module.ImprovedAutoencoder(input_channels=3, latent_channels=128, time_reduction=4)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
            model.eval()
            
            # Process a small batch of frames
            frames_tensor = torch.stack([
                torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                for frame in frames
            ]).to(device)
            
            # Test forward pass
            with torch.no_grad():
                t1 = time.time()
                latent = model.encode(frames_tensor.unsqueeze(0))
                reconstructed = model.decode(latent).squeeze(0)
                t2 = time.time()
            
            # Calculate metrics
            psnr = -10 * torch.log10(((frames_tensor - reconstructed) ** 2).mean())
            
            # Write the reconstructed frames to a video
            output_video = os.path.join(tmp_dir, "reconstructed_video.mp4")
            out = cv2.VideoWriter(output_video, fourcc, 30, (128, 128))
            for frame in reconstructed.cpu().permute(0, 2, 3, 1).numpy():
                out.write((frame * 255).astype(np.uint8))
            out.release()
            
            # Test if we can also compress with FFmpeg
            try:
                ffmpeg_output = os.path.join(tmp_dir, "ffmpeg_output.mp4")
                subprocess.run([
                    "ffmpeg", "-y", "-i", test_video, 
                    "-c:v", "libx265", "-crf", "28", 
                    ffmpeg_output
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                ffmpeg_success = os.path.exists(ffmpeg_output) and os.path.getsize(ffmpeg_output) > 0
                ffmpeg_status = "succeed" if ffmpeg_success else "failed"
            except Exception as e:
                ffmpeg_success = False
                ffmpeg_status = f"failed with error: {str(e)}"
            
            return True, (f"Compression test successful: PSNR={psnr:.2f}dB, "
                          f"Processing time: {(t2-t1)*1000:.2f}ms, "
                          f"FFmpeg test: {ffmpeg_status}")
    
    except ImportError as e:
        return False, f"Dependency error: {str(e)}"
    except Exception as e:
        return False, f"Test failed: {str(e)}"


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "="*80)
    print(" System Verification for Video Compression Project ".center(80, "="))
    print("="*80 + "\n")
    
    all_passed = True
    issue_count = 0
    checks = []
    
    # Basic checks
    if not args.skip_deps:
        # Check Python version
        success, status, message = check_python_version()
        print_status("Python Version", status, message, is_error=not success, verbose=args.verbose)
        checks.append(("Python Version", success))
        if not success:
            all_passed = False
            issue_count += 1
        
        # Check dependencies
        success, status, message = check_dependencies()
        print_status("Dependencies", status, message, is_error=not success, verbose=args.verbose)
        checks.append(("Dependencies", success))
        if not success:
            all_passed = False
            issue_count += 1
    
    # CUDA check
    if not args.skip_cuda:
        success, status, message = check_cuda()
        print_status("CUDA", status, message, is_error=not success, verbose=args.verbose)
        checks.append(("CUDA", success))
        # CUDA is optional, so don't affect all_passed
    
    # FFmpeg check
    if not args.skip_ffmpeg:
        success, status, message = check_ffmpeg()
        print_status("FFmpeg", status, message, is_error=not success, verbose=args.verbose)
        checks.append(("FFmpeg", success))
        if not success:
            all_passed = False
            issue_count += 1
            
            # Try to fix FFmpeg if --fix flag is set
            if args.fix:
                fix_success, fix_message = setup_ffmpeg(fix=True)
                if fix_success:
                    # Re-check FFmpeg
                    success, status, message = check_ffmpeg()
                    print_status("FFmpeg (Fixed)", status, message, is_error=not success, verbose=args.verbose)
                    checks.append(("FFmpeg (Fixed)", success))
                    if success:
                        all_passed = True
                        issue_count -= 1
                else:
                    print_status("FFmpeg Fix", "Failed", fix_message, is_error=True, verbose=args.verbose)
            else:
                fix_success, fix_message = setup_ffmpeg(fix=False)
                print_status("FFmpeg Fix", "Not Attempted", fix_message, is_warning=True, verbose=args.verbose)
    
    # Model files check
    if not args.skip_models:
        success, status, message = check_model_files()
        print_status("Model Files", status, message, is_warning=not success, verbose=args.verbose)
        checks.append(("Model Files", success))
        # Models might be generated later, so don't affect all_passed
    
    # Sample videos check
    if not args.skip_videos:
        success, status, message = check_sample_videos()
        print_status("Sample Videos", status, message, is_warning=not success, verbose=args.verbose)
        checks.append(("Sample Videos", success))
        if not success:
            # Only warning, don't affect all_passed
            
            # Try to fix sample videos if --fix flag is set
            if args.fix:
                fix_success, fix_message = setup_sample_videos(fix=True)
                if fix_success:
                    # Re-check sample videos
                    success, status, message = check_sample_videos()
                    print_status("Sample Videos (Fixed)", status, message, is_warning=not success, verbose=args.verbose)
                    checks.append(("Sample Videos (Fixed)", success))
                else:
                    print_status("Sample Videos Fix", "Failed", fix_message, is_warning=True, verbose=args.verbose)
            else:
                fix_success, fix_message = setup_sample_videos(fix=False)
                print_status("Sample Videos Fix", "Not Attempted", fix_message, is_warning=True, verbose=args.verbose)
    
    # Run a quick compression test if requested
    if args.run_test and all_passed:
        success, message = run_quick_compression_test()
        print_status("Compression Test", "Passed" if success else "Failed", message, is_error=not success, verbose=True)
        checks.append(("Compression Test", success))
        if not success:
            all_passed = False
            issue_count += 1
    
    # Print summary
    print("\n" + "-"*80)
    print(" Verification Summary ".center(80, "-"))
    print("-"*80)
    
    passed_count = sum(1 for _, success in checks if success)
    print(f"Checks passed: {passed_count}/{len(checks)}")
    
    if issue_count > 0:
        print(f"Critical issues found: {issue_count}")
        print("Please fix the issues marked with ❌ before running the pipeline.")
    
    if all_passed:
        print("\n✅ All critical checks passed. The system is ready for video compression tasks.")
        sys.exit(0)
    else:
        print("\n❌ Some critical checks failed. Please fix the issues before proceeding.")
        
        # Provide guidance based on failed checks
        failed_checks = [name for name, success in checks if not success]
        if "FFmpeg" in failed_checks:
            is_windows = platform.system() == "Windows"
            script = "scripts/setup_ffmpeg.bat" if is_windows else "scripts/setup_ffmpeg.sh"
            print(f"- To fix FFmpeg issues, run: {script}")
        
        if "Dependencies" in failed_checks:
            print("- To fix dependency issues, run: pip install -r requirements.txt")
        
        if "Sample Videos" in failed_checks:
            print("- To download sample videos, run: python scripts/download_sample_videos.py")
        
        print("\nYou can run this script with the --fix flag to attempt automatic fixes.")
        sys.exit(1)


if __name__ == "__main__":
    main() 