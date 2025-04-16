import os
import subprocess
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Encode videos using x265')
    parser.add_argument('--input', type=str, required=True, help='Input video file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--crf', type=int, default=23, help='CRF value (0-51, lower means better quality)')
    parser.add_argument('--preset', type=str, default='medium', 
                      choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                      help='x265 preset')
    return parser.parse_args()

def encode_video(input_path, output_path, crf, preset):
    """
    Encode a video using x265
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Construct x265 command
    cmd = [
        'x265',
        '--input', input_path,
        '--output', output_path,
        '--crf', str(crf),
        '--preset', preset
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully encoded: {input_path} -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error encoding {input_path}: {e}")
        return False
    return True

def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if input_path.is_file():
        # Single file encoding
        output_path = output_dir / f"{input_path.stem}_x265{input_path.suffix}"
        encode_video(str(input_path), str(output_path), args.crf, args.preset)
    elif input_path.is_dir():
        # Directory encoding
        for video_file in input_path.glob('*.mp4'):  # Add more extensions if needed
            output_path = output_dir / f"{video_file.stem}_x265{video_file.suffix}"
            encode_video(str(video_file), str(output_path), args.crf, args.preset)
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == '__main__':
    main() 