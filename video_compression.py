import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Import mô hình autoencoder
from train_mot_simplified import SimpleAutoencoder

def parse_args():
    parser = argparse.ArgumentParser(description="Compress and decompress video using trained autoencoder")
    
    # Input parameters
    parser.add_argument("--input_video", type=str, required=True,
                        help="Path to input video file")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="trained_models/mot16_model/autoencoder_best.pt",
                        help="Path to trained autoencoder model")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames to process at once")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results/compressed_videos",
                        help="Directory to save compressed and reconstructed videos")
    
    return parser.parse_args()

def extract_frames(video_path):
    """Tách frames từ video."""
    print(f"Opening video file: {video_path}")
    # Check if file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chuyển BGR sang RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize nếu cần (để phù hợp với mô hình)
        frame = cv2.resize(frame, (224, 224))
        
        # Chuẩn hóa về [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    return frames, fps, width, height, frame_count

def create_video(frames, output_path, fps, width, height):
    """Tạo video từ frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Chuyển từ float [0, 1] sang uint8 [0, 255]
        frame = (frame * 255).astype(np.uint8)
        
        # Chuyển từ RGB sang BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize lại kích thước gốc nếu cần
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
            
        out.write(frame)
    
    out.release()

def compress_video(model, frames, time_steps, device):
    """Nén video sử dụng autoencoder."""
    total_frames = len(frames)
    compressed_sequences = []
    reconstructed_frames = []
    
    # Xử lý theo chuỗi frames
    for i in tqdm(range(0, total_frames, time_steps), desc="Compressing video"):
        # Lấy một chuỗi frames
        if i + time_steps > total_frames:
            # Xử lý frame cuối, lặp lại nếu không đủ
            sequence = frames[i:] + frames[i:i+time_steps-len(frames[i:])]
        else:
            sequence = frames[i:i+time_steps]
        
        # Chuyển thành tensor
        sequence_tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
        sequence_tensor = sequence_tensor.to(device)
        
        # Nén và tái tạo
        with torch.no_grad():
            compressed = model.encoder(sequence_tensor)
            reconstructed = model.decoder(compressed)
        
        # Lưu dữ liệu nén và tái tạo
        compressed_sequences.append(compressed.cpu().numpy())
        
        # Chuyển lại về dạng frames
        reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
        
        # Chỉ lấy đúng số lượng frames cần thiết, tránh lặp lại
        if i + time_steps > total_frames:
            reconstructed_frames.extend(reconstructed_np[:total_frames-i])
        else:
            reconstructed_frames.extend(reconstructed_np)
    
    return compressed_sequences, reconstructed_frames

def calculate_compression_ratio(original_frames, compressed_sequences):
    """Tính tỷ lệ nén dữ liệu."""
    # Kích thước dữ liệu gốc (bytes)
    original_size = np.prod(np.array(original_frames).shape) * np.float32().itemsize
    
    # Kích thước dữ liệu nén (bytes)
    compressed_size = sum(np.prod(seq.shape) * np.float32().itemsize for seq in compressed_sequences)
    
    # Tỷ lệ nén
    compression_ratio = original_size / compressed_size if compressed_size < original_size else original_size / compressed_size
    
    # In ra thông tin chi tiết để gỡ lỗi
    print(f"Original frames shape: {np.array(original_frames).shape}")
    print(f"Original frames size: {original_size} bytes")
    
    for i, seq in enumerate(compressed_sequences):
        print(f"Compressed sequence {i} shape: {seq.shape}")
    
    print(f"Total compressed size: {compressed_size} bytes")
    
    # Nếu compressed_size lớn hơn original_size, model không nén hiệu quả
    if compressed_size > original_size:
        print("Warning: The compressed size is larger than the original size.")
        print("This indicates that the model is not effectively compressing the data.")
        print("Consider adjusting the model architecture or training parameters for better compression.")
    
    return compression_ratio, original_size, compressed_size

def main():
    args = parse_args()
    
    # Normalize file paths to handle Windows/Unix path separators
    input_video = os.path.normpath(args.input_video)
    
    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tải mô hình
    print(f"Loading model from {args.model_path}...")
    model = SimpleAutoencoder().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Tạo tên file đầu ra
    video_name = Path(input_video).stem
    output_reconstructed = os.path.join(args.output_dir, f"{video_name}_reconstructed.mp4")
    output_comparison = os.path.join(args.output_dir, f"{video_name}_comparison.mp4")
    
    # Tách frames từ video
    print(f"Extracting frames from {input_video}...")
    frames, fps, width, height, frame_count = extract_frames(input_video)
    print(f"Extracted {len(frames)} frames ({width}x{height} at {fps} fps)")
    
    # Nén và tái tạo video
    print("Compressing and reconstructing video...")
    compressed_sequences, reconstructed_frames = compress_video(model, frames, args.time_steps, device)
    
    # Tính tỷ lệ nén
    compression_ratio, original_size_bytes, compressed_size_bytes = calculate_compression_ratio(frames, compressed_sequences)
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Original size: {original_size_bytes/1024/1024:.2f} MB")
    print(f"Compressed size: {compressed_size_bytes/1024/1024:.2f} MB")
    
    # Tạo video đã tái tạo
    print(f"Creating reconstructed video: {output_reconstructed}")
    create_video(reconstructed_frames, output_reconstructed, fps, width, height)
    
    # Tạo video so sánh (gốc bên trái, tái tạo bên phải)
    print(f"Creating comparison video: {output_comparison}")
    comparison_frames = []
    for orig, recon in zip(frames, reconstructed_frames):
        # Resize về kích thước gốc
        if orig.shape[0] != height or orig.shape[1] != width:
            orig = cv2.resize(orig, (width, height))
        if recon.shape[0] != height or recon.shape[1] != width:
            recon = cv2.resize(recon, (width, height))
            
        # Tạo hình ảnh so sánh
        comparison = np.hstack([orig, recon])
        comparison_frames.append(comparison)
    
    create_video(comparison_frames, output_comparison, fps, width*2, height)
    
    # Lưu kết quả vào file
    results_file = os.path.join(args.output_dir, f"{video_name}_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Video compression results for {input_video}\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
        f.write(f"Original size: {original_size_bytes/1024/1024:.2f} MB\n")
        f.write(f"Compressed size: {compressed_size_bytes/1024/1024:.2f} MB\n")
        f.write(f"Frames: {frame_count}\n")
        f.write(f"Resolution: {width}x{height}\n")
        f.write(f"FPS: {fps}\n")
    
    print(f"\nCompression completed. Results saved to {args.output_dir}")
    print(f"Reconstructed video: {output_reconstructed}")
    print(f"Comparison video: {output_comparison}")
    print(f"Results file: {results_file}")

if __name__ == "__main__":
    main() 