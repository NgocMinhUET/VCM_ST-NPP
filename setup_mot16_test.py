#!/usr/bin/env python3
"""
Script to download and prepare a small portion of the MOT16 dataset for testing.
"""

import os
import sys
import urllib.request
import zipfile
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Download a small portion of MOT16 dataset')
    parser.add_argument('--output_dir', type=str, default='./test_data/mot16_test',
                        help='Directory to save the dataset')
    parser.add_argument('--sequence', type=str, default='MOT16-02',
                        help='MOT16 sequence to download (e.g., MOT16-02, MOT16-04)')
    return parser.parse_args()

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_and_extract_sequence(sequence_name, output_dir):
    """Download and extract a specific MOT16 sequence."""
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for the specific sequence
    # Note: This assumes the MOT16 sequences are accessible via individual URLs
    # If not, you would need to download the full dataset and extract just the needed sequence
    url = f"https://motchallenge.net/data/MOT16/{sequence_name}.zip"
    zip_path = os.path.join(output_dir, f"{sequence_name}.zip")
    
    print(f"Downloading {sequence_name}...")
    try:
        download_url(url, zip_path)
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        print("Will try alternative approach...")
        # Alternative: Download the MOT16 demo (which is much smaller)
        url = "https://motchallenge.net/data/MOT16_demo.zip"
        print(f"Downloading MOT16 demo...")
        download_url(url, os.path.join(output_dir, "MOT16_demo.zip"))
        zip_path = os.path.join(output_dir, "MOT16_demo.zip")
    
    print(f"Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Clean up zip file
    os.remove(zip_path)
    
    # Find the extracted sequence directory
    for root, dirs, files in os.walk(output_dir):
        for dir_name in dirs:
            if sequence_name in dir_name:
                sequence_path = os.path.join(root, dir_name)
                print(f"Found sequence at: {sequence_path}")
                
                # Check if the sequence has an 'img1' subdirectory (which contains the frames)
                img_dir = os.path.join(sequence_path, 'img1')
                if os.path.exists(img_dir) and os.path.isdir(img_dir):
                    print(f"Found image directory: {img_dir}")
                    return sequence_path
    
    print(f"Could not find {sequence_name} in the extracted files.")
    return None

def create_test_dataset(sequence_path, output_dir):
    """Create a small test dataset with a limited number of frames."""
    if sequence_path is None or not os.path.exists(sequence_path):
        print("Sequence path is invalid.")
        return
    
    # Get the image directory
    img_dir = os.path.join(sequence_path, 'img1')
    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        return
    
    # Create output directories
    test_img_dir = os.path.join(output_dir, os.path.basename(sequence_path), 'img1')
    os.makedirs(test_img_dir, exist_ok=True)
    
    # Copy a subset of frames (e.g., first 100)
    image_files = sorted(os.listdir(img_dir))
    max_frames = min(100, len(image_files))
    
    print(f"Copying {max_frames} frames to {test_img_dir}...")
    for i in range(max_frames):
        src = os.path.join(img_dir, image_files[i])
        dst = os.path.join(test_img_dir, image_files[i])
        shutil.copy2(src, dst)
    
    print(f"Created test dataset with {max_frames} frames at {output_dir}")

def main():
    args = parse_args()
    
    # Download and extract the sequence
    sequence_path = download_and_extract_sequence(args.sequence, args.output_dir)
    
    # Create a small test dataset
    if sequence_path:
        create_test_dataset(sequence_path, args.output_dir)
    else:
        print("Failed to download sequence. Check if the URL is correct or try another sequence.")

if __name__ == "__main__":
    main() 