#!/usr/bin/env python3
"""
Script to download sample videos for testing the video compression system.
Downloads videos from public datasets and places them in the data/sample_videos directory.
"""

import os
import sys
import argparse
import requests
from tqdm import tqdm
import hashlib
import shutil
import zipfile
import tarfile
import subprocess

# Define sample video sources with metadata
SAMPLE_VIDEOS = [
    {
        "name": "pedestrians.mp4",
        "url": "https://media.githubusercontent.com/media/opencv/opencv_extra/master/testdata/highgui/video/pedestrians.mp4",
        "description": "Pedestrians walking on a street, good for testing object detection and tracking",
        "size_mb": 2.2,
        "md5": "ee6a9b4605fd805306a5c5e3d231fbfa",
        "license": "BSD License (OpenCV)"
    },
    {
        "name": "highway.mp4",
        "url": "https://media.githubusercontent.com/media/intelligentrobots/datasets/main/videos/highway.mp4",
        "description": "Highway traffic scene with multiple vehicles",
        "size_mb": 4.7,
        "md5": "0b7d880e419f4c1a3eb68ddfde0e9610",
        "license": "Public Domain"
    },
    {
        "name": "crowd.mp4",
        "url": "https://media.githubusercontent.com/media/intelligentrobots/datasets/main/videos/crowd.mp4",
        "description": "Crowded scene with many people, challenging for tracking",
        "size_mb": 5.2,
        "md5": "7b49c71828a7c4a3356b15ddbecabf8f",
        "license": "Public Domain"
    }
]

# URLs for additional video datasets
MOT_DATASET_URL = "https://motchallenge.net/data/MOT16.zip"
DAVIS_DATASET_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download sample videos for testing")
    parser.add_argument("--output_dir", type=str, default="data/sample_videos",
                        help="Directory to save sample videos")
    parser.add_argument("--dataset", choices=["basic", "mot", "davis", "all"],
                        default="basic", help="Dataset to download")
    parser.add_argument("--verify", action="store_true",
                        help="Verify integrity of downloaded files")
    parser.add_argument("--force", action="store_true",
                        help="Force download even if files already exist")
    return parser.parse_args()


def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url, output_path, expected_md5=None):
    """Download a file with progress bar."""
    if os.path.exists(output_path) and expected_md5:
        if calculate_md5(output_path) == expected_md5:
            print(f"File already exists and MD5 matches: {output_path}")
            return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(output_path, 'wb') as f, tqdm(
                desc=os.path.basename(output_path),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        # Verify MD5 if provided
        if expected_md5:
            actual_md5 = calculate_md5(output_path)
            if actual_md5 != expected_md5:
                print(f"Warning: MD5 mismatch for {output_path}")
                print(f"Expected: {expected_md5}")
                print(f"Actual: {actual_md5}")
                return False
        
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def extract_archive(archive_path, output_dir):
    """Extract a zip or tar archive."""
    print(f"Extracting {archive_path} to {output_dir}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(output_dir)
    else:
        print(f"Unsupported archive format: {archive_path}")
        return False
    
    return True


def download_sample_videos(output_dir, force=False):
    """Download sample videos from the predefined sources."""
    os.makedirs(output_dir, exist_ok=True)
    
    successful_downloads = []
    
    for video in SAMPLE_VIDEOS:
        output_path = os.path.join(output_dir, video["name"])
        
        if os.path.exists(output_path) and not force:
            print(f"File already exists: {output_path}")
            successful_downloads.append(video["name"])
            continue
        
        print(f"Downloading {video['name']} ({video['size_mb']:.1f} MB)...")
        if download_file(video["url"], output_path, video["md5"]):
            successful_downloads.append(video["name"])
    
    print(f"\nSuccessfully downloaded {len(successful_downloads)} of {len(SAMPLE_VIDEOS)} sample videos.")
    
    return successful_downloads


def download_mot_dataset(output_dir, force=False):
    """Download and extract MOT16 dataset."""
    mot_dir = os.path.join(output_dir, "MOT16")
    
    if os.path.exists(mot_dir) and not force:
        print(f"MOT16 dataset already exists at {mot_dir}")
        return True
    
    zip_path = os.path.join(output_dir, "MOT16.zip")
    
    print(f"Downloading MOT16 dataset (~1.9 GB)...")
    if not download_file(MOT_DATASET_URL, zip_path):
        return False
    
    # Extract the dataset
    if extract_archive(zip_path, output_dir):
        # Remove the zip file after extraction
        os.remove(zip_path)
        print(f"MOT16 dataset extracted to {mot_dir}")
        return True
    
    return False


def download_davis_dataset(output_dir, force=False):
    """Download and extract DAVIS dataset."""
    davis_dir = os.path.join(output_dir, "DAVIS")
    
    if os.path.exists(davis_dir) and not force:
        print(f"DAVIS dataset already exists at {davis_dir}")
        return True
    
    zip_path = os.path.join(output_dir, "DAVIS-2017-trainval-480p.zip")
    
    print(f"Downloading DAVIS dataset (~340 MB)...")
    if not download_file(DAVIS_DATASET_URL, zip_path):
        return False
    
    # Extract the dataset
    if extract_archive(zip_path, output_dir):
        # Remove the zip file after extraction
        os.remove(zip_path)
        print(f"DAVIS dataset extracted to {davis_dir}")
        return True
    
    return False


def verify_ffmpeg():
    """Verify FFmpeg is installed."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if result.returncode == 0:
            print("FFmpeg is installed:")
            print(result.stdout.splitlines()[0])
            return True
        else:
            print("FFmpeg check failed.")
            return False
    except FileNotFoundError:
        print("FFmpeg is not installed or not in PATH.")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    # Verify FFmpeg installation
    if not verify_ffmpeg():
        print("FFmpeg is required for video processing.")
        print("Please install FFmpeg or run the setup scripts:")
        print("  - Windows: scripts/setup_ffmpeg.bat")
        print("  - Linux/macOS: scripts/setup_ffmpeg.sh")
        sys.exit(1)
    
    # Create output directories
    data_dir = os.path.dirname(args.output_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download selected datasets
    if args.dataset in ["basic", "all"]:
        download_sample_videos(args.output_dir, args.force)
    
    if args.dataset in ["mot", "all"]:
        download_mot_dataset(data_dir, args.force)
    
    if args.dataset in ["davis", "all"]:
        download_davis_dataset(data_dir, args.force)
    
    # Verify all files if requested
    if args.verify:
        for video in SAMPLE_VIDEOS:
            output_path = os.path.join(args.output_dir, video["name"])
            if os.path.exists(output_path):
                print(f"Verifying {video['name']}...")
                actual_md5 = calculate_md5(output_path)
                if actual_md5 == video["md5"]:
                    print(f"✓ MD5 checksum matches for {video['name']}")
                else:
                    print(f"✗ MD5 mismatch for {video['name']}")
                    print(f"  Expected: {video['md5']}")
                    print(f"  Actual: {actual_md5}")
    
    # Print summary
    print("\nSample Video Summary:")
    for video in SAMPLE_VIDEOS:
        path = os.path.join(args.output_dir, video["name"])
        if os.path.exists(path):
            print(f"✓ {video['name']} - {video['description']}")
        else:
            print(f"✗ {video['name']} - Not downloaded")
    
    if args.dataset in ["mot", "all"]:
        mot_dir = os.path.join(data_dir, "MOT16")
        if os.path.exists(mot_dir):
            print(f"✓ MOT16 dataset - Available at {mot_dir}")
        else:
            print("✗ MOT16 dataset - Not downloaded")
    
    if args.dataset in ["davis", "all"]:
        davis_dir = os.path.join(data_dir, "DAVIS")
        if os.path.exists(davis_dir):
            print(f"✓ DAVIS dataset - Available at {davis_dir}")
        else:
            print("✗ DAVIS dataset - Not downloaded")


if __name__ == "__main__":
    main() 