import os
import subprocess
import numpy as np
import cv2
import tempfile
import shutil
from pathlib import Path
import tensorflow as tf

def save_frames_as_yuv(frames, output_path, format='420'):
    """
    Save frames as YUV file.
    
    Args:
        frames: List or numpy array of frames in RGB format
        output_path: Path to save the YUV file
        format: YUV format ('420', '422', or '444')
        
    Returns:
        Path to the saved YUV file
    """
    if isinstance(frames, list):
        frames = np.stack(frames)
        
    height, width = frames.shape[1:3]
    
    # Create YUV file
    with open(output_path, 'wb') as f:
        for frame in frames:
            # Convert from RGB to YUV
            frame_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
            
            # Write Y plane
            f.write(frame_yuv[:, :, 0].tobytes())
            
            # Subsampling for U and V planes based on format
            if format == '420':
                # Downsample U and V planes to half resolution in both dimensions
                u = cv2.resize(frame_yuv[:, :, 1], (width // 2, height // 2))
                v = cv2.resize(frame_yuv[:, :, 2], (width // 2, height // 2))
            elif format == '422':
                # Downsample U and V planes to half resolution horizontally
                u = cv2.resize(frame_yuv[:, :, 1], (width // 2, height))
                v = cv2.resize(frame_yuv[:, :, 2], (width // 2, height))
            elif format == '444':
                # No downsampling
                u = frame_yuv[:, :, 1]
                v = frame_yuv[:, :, 2]
            else:
                raise ValueError(f"Unsupported YUV format: {format}")
                
            # Write U and V planes
            f.write(u.tobytes())
            f.write(v.tobytes())
    
    return output_path

def encode_with_hevc(input_path, output_path, qp=23, preset='medium', yuv_format='420', resolution=None):
    """
    Encode video with HEVC codec using ffmpeg.
    
    Args:
        input_path: Path to the input YUV file
        output_path: Path to save the encoded file
        qp: Quantization Parameter
        preset: Encoding preset ('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow')
        yuv_format: YUV format ('420', '422', or '444')
        resolution: Resolution of the input video as 'WxH' (e.g., '1920x1080')
        
    Returns:
        Path to the encoded file
    """
    if resolution is None:
        raise ValueError("Resolution must be provided for raw YUV input")
        
    # Construct ffmpeg command
    cmd = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pixel_format', f"yuv{yuv_format}p",
        '-video_size', resolution,
        '-i', input_path,
        '-c:v', 'libx265',
        '-preset', preset,
        '-x265-params', f"qp={qp}",
        '-f', 'mp4',
        output_path
    ]
    
    # Run the command
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error encoding video: {e}")
        print(f"STDERR: {e.stderr.decode()}")
        raise
        
    return output_path

def decode_with_hevc(input_path, output_path, yuv_format='420', resolution=None):
    """
    Decode HEVC video to YUV using ffmpeg.
    
    Args:
        input_path: Path to the input encoded file
        output_path: Path to save the decoded YUV file
        yuv_format: YUV format ('420', '422', or '444')
        resolution: Resolution of the video as 'WxH' (e.g., '1920x1080')
        
    Returns:
        Path to the decoded YUV file
    """
    # Construct ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-f', 'rawvideo',
        '-pixel_format', f"yuv{yuv_format}p"
    ]
    
    # Add resolution if provided
    if resolution is not None:
        cmd.extend(['-s', resolution])
        
    cmd.append(output_path)
    
    # Run the command
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error decoding video: {e}")
        print(f"STDERR: {e.stderr.decode()}")
        raise
        
    return output_path

def load_yuv_frames(yuv_path, width, height, n_frames=None, format='420'):
    """
    Load frames from YUV file.
    
    Args:
        yuv_path: Path to the YUV file
        width: Frame width
        height: Frame height
        n_frames: Number of frames to load (None for all)
        format: YUV format ('420', '422', or '444')
        
    Returns:
        List of frames in RGB format
    """
    # Calculate frame size in bytes based on YUV format
    if format == '420':
        # Y plane: width * height, U and V planes: (width/2) * (height/2) each
        frame_size = width * height + 2 * (width // 2) * (height // 2)
    elif format == '422':
        # Y plane: width * height, U and V planes: (width/2) * height each
        frame_size = width * height + 2 * (width // 2) * height
    elif format == '444':
        # Y, U, and V planes: width * height each
        frame_size = width * height * 3
    else:
        raise ValueError(f"Unsupported YUV format: {format}")
    
    # Open YUV file and read frames
    frames = []
    with open(yuv_path, 'rb') as f:
        while True:
            # Read a frame
            yuv_data = f.read(frame_size)
            if not yuv_data or (n_frames is not None and len(frames) >= n_frames):
                break
                
            # Parse YUV data based on format
            if format == '420':
                y_size = width * height
                u_size = v_size = (width // 2) * (height // 2)
                
                y = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape(height, width)
                u = np.frombuffer(yuv_data[y_size:y_size+u_size], dtype=np.uint8).reshape(height//2, width//2)
                v = np.frombuffer(yuv_data[y_size+u_size:], dtype=np.uint8).reshape(height//2, width//2)
                
                # Upsample U and V planes to match Y plane size
                u = cv2.resize(u, (width, height))
                v = cv2.resize(v, (width, height))
                
            elif format == '422':
                y_size = width * height
                u_size = v_size = (width // 2) * height
                
                y = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape(height, width)
                u = np.frombuffer(yuv_data[y_size:y_size+u_size], dtype=np.uint8).reshape(height, width//2)
                v = np.frombuffer(yuv_data[y_size+u_size:], dtype=np.uint8).reshape(height, width//2)
                
                # Upsample U and V planes to match Y plane size
                u = cv2.resize(u, (width, height))
                v = cv2.resize(v, (width, height))
                
            elif format == '444':
                y_size = u_size = v_size = width * height
                
                y = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape(height, width)
                u = np.frombuffer(yuv_data[y_size:y_size+u_size], dtype=np.uint8).reshape(height, width)
                v = np.frombuffer(yuv_data[y_size+u_size:], dtype=np.uint8).reshape(height, width)
            
            # Stack YUV planes and convert to RGB
            yuv_frame = np.stack([y, u, v], axis=2)
            rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)
            
            frames.append(rgb_frame)
    
    return frames

def run_hevc_pipeline(frames, qp=23, preset='medium', yuv_format='420'):
    """
    Run the complete HEVC encoding and decoding pipeline on a sequence of frames.
    
    Args:
        frames: List or numpy array of frames in RGB format
        qp: Quantization Parameter
        preset: Encoding preset
        yuv_format: YUV format
        
    Returns:
        List of decoded frames in RGB format
    """
    if isinstance(frames, list):
        frames = np.stack(frames)
        
    height, width = frames.shape[1:3]
    resolution = f"{width}x{height}"
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define file paths
        temp_dir = Path(temp_dir)
        yuv_path = temp_dir / "input.yuv"
        encoded_path = temp_dir / "encoded.mp4"
        decoded_path = temp_dir / "decoded.yuv"
        
        # Save frames as YUV
        save_frames_as_yuv(frames, str(yuv_path), format=yuv_format)
        
        # Encode with HEVC
        encode_with_hevc(
            str(yuv_path), 
            str(encoded_path), 
            qp=qp, 
            preset=preset, 
            yuv_format=yuv_format, 
            resolution=resolution
        )
        
        # Decode back to YUV
        decode_with_hevc(
            str(encoded_path), 
            str(decoded_path), 
            yuv_format=yuv_format, 
            resolution=resolution
        )
        
        # Load decoded frames
        decoded_frames = load_yuv_frames(
            str(decoded_path), 
            width, 
            height, 
            n_frames=len(frames), 
            format=yuv_format
        )
    
    return decoded_frames

def calculate_bpp(encoded_file_path, frames):
    """
    Calculate bits per pixel (bpp) for an encoded video.
    
    Args:
        encoded_file_path: Path to the encoded video file
        frames: List or numpy array of the original frames
        
    Returns:
        Bits per pixel value
    """
    if isinstance(frames, list):
        frames = np.stack(frames)
        
    # Get file size in bits
    file_size_bits = os.path.getsize(encoded_file_path) * 8
    
    # Calculate total number of pixels
    num_frames = len(frames)
    height, width = frames.shape[1:3]
    total_pixels = num_frames * height * width
    
    # Calculate bits per pixel
    bpp = file_size_bits / total_pixels
    
    return bpp 