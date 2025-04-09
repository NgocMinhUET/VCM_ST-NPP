#!/usr/bin/env python3
"""
Script to download a sample video for testing the STNPP model.
"""

import os
import sys
import requests
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Download a sample video for testing')
    parser.add_argument('--output_dir', type=str, default='test_data',
                        help='Directory to save the downloaded video')
    return parser.parse_args()

def download_file(url, filename):
    """
    Download a file from a URL with a progress bar.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    progress_bar = tqdm(
        total=total_size, 
        unit='iB', 
        unit_scale=True,
        desc=f"Downloading {filename}"
    )
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Sample video URLs
    sample_videos = [
        {
            'name': 'sample_video.mp4',
            'url': 'https://media.istockphoto.com/id/1198271727/video/traffic-at-night-in-city.mp4?s=mp4-640x640-is&k=20&c=gJqLkYY7jW-eEcluWOvLXKCqvrDxNNh0wp-G2RhW4JQ='
        },
        {
            'name': 'sample_video2.mp4',
            'url': 'https://media.istockphoto.com/id/1352182773/video/4k-clip-of-people-walking-in-city.mp4?s=mp4-640x640-is&k=20&c=iOJIYwOaxu5ZCrckiydNb5-zUByE3HQH3c6mpVGJl-Y='
        }
    ]
    
    # Download each video
    for video in sample_videos:
        output_path = os.path.join(args.output_dir, video['name'])
        print(f"Downloading {video['name']} to {output_path}...")
        download_file(video['url'], output_path)
        print(f"Downloaded {video['name']} successfully.")
    
    print(f"All videos downloaded to {args.output_dir}.")

if __name__ == "__main__":
    main() 