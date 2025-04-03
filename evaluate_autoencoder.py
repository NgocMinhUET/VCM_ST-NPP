import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import cv2

# Import dataset và model
from mot_dataset import MOTImageSequenceDataset
from train_mot_simplified import SimpleAutoencoder

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained autoencoder on MOT16 dataset")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, default="D:/NCS/propose/dataset/MOT16",
                        help="Path to the MOT16 dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (train or test)")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames in each sequence")
    parser.add_argument("--frame_stride", type=int, default=4,
                        help="Stride for frame sampling")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="trained_models/mot16_model/autoencoder_best.pt",
                        help="Path to the trained model checkpoint")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results/autoencoder_evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of sample sequences to visualize")
    
    return parser.parse_args()

def visualize_reconstruction(original, reconstruction, sequence_idx, output_dir):
    """Trực quan hóa kết quả nén và tái tạo cho một chuỗi frame."""
    
    # Tạo thư mục lưu kết quả
    output_path = os.path.join(output_dir, f"sequence_{sequence_idx}")
    os.makedirs(output_path, exist_ok=True)
    
    # Lấy frames từ tensor (B, C, T, H, W)
    original = original.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
    reconstruction = reconstruction.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
    
    # Chuyển từ tensor sang numpy và lưu hình ảnh
    for t in range(original.shape[1]):
        # Lấy frame gốc và tái tạo
        orig_frame = original[0, t].cpu().numpy() * 255
        recon_frame = reconstruction[0, t].cpu().numpy() * 255
        
        # Chuyển về uint8 để lưu hình ảnh
        orig_frame = orig_frame.astype(np.uint8)
        recon_frame = recon_frame.astype(np.uint8)
        
        # Tạo hình ảnh so sánh
        comparison = np.hstack((orig_frame, recon_frame))
        
        # Lưu hình ảnh
        cv2.imwrite(os.path.join(output_path, f"frame_{t:03d}.jpg"), 
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # Tạo hình ảnh tổng hợp cho một số frame đại diện
    plt.figure(figsize=(12, 8))
    select_frames = np.linspace(0, original.shape[1]-1, 4, dtype=int)
    
    for i, t in enumerate(select_frames):
        # Original
        plt.subplot(2, 4, i+1)
        plt.imshow(original[0, t].cpu().numpy())
        plt.title(f"Original frame {t}")
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(2, 4, i+5)
        plt.imshow(reconstruction[0, t].cpu().numpy())
        plt.title(f"Reconstructed frame {t}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "summary.png"))
    plt.close()
    
    # Tạo video so sánh
    video_path = os.path.join(output_path, "comparison.mp4")
    height, width = comparison.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    
    for t in range(original.shape[1]):
        orig_frame = original[0, t].cpu().numpy() * 255
        recon_frame = reconstruction[0, t].cpu().numpy() * 255
        
        orig_frame = orig_frame.astype(np.uint8)
        recon_frame = recon_frame.astype(np.uint8)
        
        comparison = np.hstack((orig_frame, recon_frame))
        video.write(cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    video.release()
    
    return output_path

def calculate_metrics(original, reconstruction):
    """Tính toán các chỉ số đánh giá: MSE, PSNR, SSIM."""
    mse = nn.MSELoss()(original, reconstruction).item()
    
    # PSNR (Peak Signal-to-Noise Ratio)
    max_pixel = 1.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    
    # Tính BPP (bits per pixel) - tỷ lệ nén
    # Tính kích thước của latent representation (bit/pixel)
    # Giả sử 32 bit cho mỗi giá trị float (thông thường)
    original_size = np.prod(original.shape) * 32  # bits
    # Giả sử latent size là 1/8 kích thước original (qua 2 lần max pooling với stride 2 theo H và W)
    latent_size = original_size / 8  # bits
    
    # Số lượng pixel trong ảnh gốc
    total_pixels = np.prod(original.shape[2:]) * original.shape[1]  # H*W*T
    
    # BPP = bits / pixels
    bpp = latent_size / total_pixels
    
    return {
        "MSE": mse,
        "PSNR": psnr,
        "BPP": bpp
    }

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tạo thư mục lưu kết quả
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tải dataset
    print("Loading MOT16 dataset...")
    dataset = MOTImageSequenceDataset(
        dataset_path=args.dataset_path,
        time_steps=args.time_steps,
        split=args.split,
        frame_stride=args.frame_stride
    )
    
    # Tạo dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # Tải model đã huấn luyện
    print(f"Loading model from {args.model_path}...")
    model = SimpleAutoencoder(input_channels=3).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Đánh giá mô hình
    metrics_all = {"MSE": [], "PSNR": [], "BPP": []}
    samples_visualized = 0
    
    with torch.no_grad():
        for batch_idx, frames in enumerate(tqdm(dataloader, desc="Evaluating model")):
            # Chỉ đánh giá số lượng mẫu được chỉ định
            if batch_idx >= args.num_samples and samples_visualized >= args.num_samples:
                break
                
            # Chuyển dữ liệu sang device
            frames = frames.to(device)  # (B, T, C, H, W)
            frames = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # Chạy mô hình
            reconstructed = model(frames)
            
            # Tính toán các metrics
            metrics = calculate_metrics(frames, reconstructed)
            for key, value in metrics.items():
                metrics_all[key].append(value)
            
            # Trực quan hóa kết quả
            if samples_visualized < args.num_samples:
                visualize_reconstruction(frames, reconstructed, batch_idx, args.output_dir)
                samples_visualized += 1
    
    # In kết quả đánh giá
    print("\nEvaluation results:")
    for key, values in metrics_all.items():
        print(f"Average {key}: {np.mean(values):.6f}")
    
    # Lưu kết quả đánh giá
    metrics_file = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Evaluation results:\n")
        for key, values in metrics_all.items():
            f.write(f"Average {key}: {np.mean(values):.6f}\n")
        
        # Add MOTA information (not calculated here - would need tracking evaluation)
        f.write("\nNote: MOTA (Multiple Object Tracking Accuracy) is not calculated in this simple evaluation.\n")
        f.write("To calculate MOTA, you need to run object tracking evaluation on the reconstructed videos.\n")
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Generated {samples_visualized} sample visualizations")

if __name__ == "__main__":
    main() 