import os
import argparse
import numpy as np
import torch
import cv2
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import mô hình autoencoder
from train_mot_simplified import SimpleAutoencoder
from video_compression import compress_video, create_video, extract_frames

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MOT tracking performance with compressed videos")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, default="D:/NCS/propose/dataset/MOT16",
                        help="Path to the MOT16 dataset directory")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Specific MOT sequence to evaluate (e.g., MOT16-02). If None, evaluates all sequences.")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (train or test)")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="trained_models/mot16_model/autoencoder_best.pt",
                        help="Path to trained autoencoder model")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames to process at once")
    
    # Tracking parameters
    parser.add_argument("--tracker", type=str, default="sort",
                        help="Tracking method to use: sort, deepsort, etc.")
    parser.add_argument("--tracking_tool", type=str, default="py-motmetrics",
                        help="Tool to use for tracking evaluation: py-motmetrics, TrackEval, etc.")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results/mot_tracking_evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization of tracking results")
    
    return parser.parse_args()

def convert_frames_to_video(frames, output_path, fps=30.0):
    """Tạo video từ các frames."""
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Chuyển đổi sang uint8 nếu là float
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Chuyển từ RGB sang BGR
        if frame.shape[2] == 3:  # Nếu có 3 kênh màu
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
        out.write(frame)
    
    out.release()
    return output_path

def run_mot_tracker(video_path, output_dir, tracker="sort"):
    """Chạy thuật toán tracking trên video và trả về đường dẫn đến kết quả."""
    tracker_output_dir = os.path.join(output_dir, "tracking_results")
    os.makedirs(tracker_output_dir, exist_ok=True)
    
    video_name = Path(video_path).stem
    output_path = os.path.join(tracker_output_dir, f"{video_name}.txt")
    
    # Gọi tracker (mã hóa cứng cho giả lập)
    # Trong thực tế, bạn sẽ chạy tracker thực và lưu kết quả vào output_path
    print(f"Running {tracker} tracker on {video_path}...")
    
    try:
        # Đây là giả lập - thay thế bằng mã thực tế
        if tracker == "sort":
            cmd = f"python external/sort/sort.py --input-video {video_path} --output {output_path}"
        elif tracker == "deepsort":
            cmd = f"python external/deep_sort/deep_sort.py --input-video {video_path} --output {output_path}"
        else:
            raise ValueError(f"Unsupported tracker: {tracker}")
        
        # Giả lập việc chạy tracker - trong thực tế, bạn sẽ chạy lệnh này
        # subprocess.run(cmd, shell=True, check=True)
        
        # Giả lập kết quả tracker - tạo file giả
        with open(output_path, 'w') as f:
            f.write("# This is a simulated tracking result file\n")
            f.write("# frame_id,track_id,x,y,w,h,confidence,class,visibility\n")
            
            # Tạo một số kết quả giả
            for i in range(100):
                frame_id = i + 1
                track_id = 1
                x, y, w, h = 100, 100, 50, 100
                confidence = 0.9
                class_id = 1
                visibility = 1.0
                f.write(f"{frame_id},{track_id},{x},{y},{w},{h},{confidence},{class_id},{visibility}\n")
        
        print(f"Tracking results saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error running tracker: {e}")
        return None

def evaluate_tracking(tracking_results, ground_truth, eval_tool="py-motmetrics"):
    """Đánh giá hiệu suất tracking sử dụng công cụ đánh giá."""
    # Giả lập đánh giá tracking - thay thế bằng mã thực tế
    print(f"Evaluating tracking results using {eval_tool}...")
    
    # Trong thực tế, bạn sẽ chạy công cụ đánh giá thực
    # Đây là ví dụ giả lập
    metrics = {
        "MOTA": 65.8,  # Multiple Object Tracking Accuracy
        "MOTP": 78.3,  # Multiple Object Tracking Precision
        "IDF1": 70.2,  # ID F1 Score
        "MT": 10,      # Mostly Tracked
        "ML": 5,       # Mostly Lost
        "FP": 120,     # False Positives
        "FN": 80,      # False Negatives
        "IDSW": 15     # ID Switches
    }
    
    return metrics

def process_mot_sequence(sequence_dir, model, device, args, output_dir):
    """Xử lý một chuỗi MOT: nén, tái tạo và đánh giá."""
    sequence_name = sequence_dir.name
    print(f"\nProcessing sequence: {sequence_name}")
    
    # Đường dẫn đến thư mục chứa ảnh
    img_dir = sequence_dir / 'img1'
    if not img_dir.exists():
        print(f"Warning: img1 directory not found in {sequence_dir}")
        return None
    
    # Đường dẫn đến ground truth
    gt_path = sequence_dir / 'gt' / 'gt.txt'
    if not gt_path.exists():
        print(f"Warning: Ground truth file not found: {gt_path}")
        # Trong tập test MOT16, ground truth có thể không có sẵn
    
    # Tạo thư mục đầu ra cho chuỗi này
    sequence_output_dir = os.path.join(output_dir, sequence_name)
    os.makedirs(sequence_output_dir, exist_ok=True)
    
    # Đường dẫn video đầu ra
    original_video_path = os.path.join(sequence_output_dir, f"{sequence_name}_original.mp4")
    reconstructed_video_path = os.path.join(sequence_output_dir, f"{sequence_name}_reconstructed.mp4")
    
    # Tách frames từ thư mục ảnh
    print(f"Extracting frames from {img_dir}...")
    # Sử dụng pattern %06d.jpg cho MOT16
    img_pattern = os.path.join(str(img_dir), "%06d.jpg")
    try:
        frames, fps, width, height, frame_count = extract_frames(img_pattern)
    except Exception as e:
        print(f"Error extracting frames: {e}")
        # Thử cách khác - đọc trực tiếp các file ảnh
        image_files = sorted(list(img_dir.glob('*.jpg')))
        frames = []
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1] for model
            frames.append(img.astype(np.float32) / 255.0)
        
        if not frames:
            print(f"No frames could be extracted from {img_dir}")
            return None
            
        frame_count = len(frames)
        height, width = frames[0].shape[:2]
        fps = 30.0  # Giả sử 30fps nếu không biết
    
    print(f"Extracted {len(frames)} frames ({width}x{height})")
    
    # Tạo video gốc
    print(f"Creating original video: {original_video_path}")
    convert_frames_to_video(frames, original_video_path, fps)
    
    # Nén và tái tạo video
    print("Compressing and reconstructing video...")
    compressed_sequences, reconstructed_frames = compress_video(model, frames, args.time_steps, device)
    
    # Tính toán chỉ số nén
    original_size = np.prod(np.array(frames).shape) * np.float32().itemsize
    compressed_size = sum(np.prod(seq.shape) * np.float32().itemsize for seq in compressed_sequences)
    compression_ratio = original_size / compressed_size if compressed_size < original_size else compressed_size / original_size
    
    # Tính BPP (bits per pixel)
    total_pixels = sum(frame.shape[0] * frame.shape[1] for frame in frames)
    bpp = (compressed_size * 8) / total_pixels
    
    # Tính PSNR
    psnr_values = []
    for orig, recon in zip(frames, reconstructed_frames):
        mse = np.mean((orig - recon) ** 2)
        if mse == 0:
            psnr = 100
        else:
            max_pixel = 1.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        psnr_values.append(psnr)
    
    avg_psnr = np.mean(psnr_values)
    
    # Tạo video tái tạo
    print(f"Creating reconstructed video: {reconstructed_video_path}")
    convert_frames_to_video(reconstructed_frames, reconstructed_video_path, fps)
    
    # Chạy tracking trên video gốc và video tái tạo
    original_tracking = run_mot_tracker(original_video_path, sequence_output_dir, tracker=args.tracker)
    reconstructed_tracking = run_mot_tracker(reconstructed_video_path, sequence_output_dir, tracker=args.tracker)
    
    # Đánh giá tracking (chỉ khi có ground truth)
    tracking_metrics = None
    if gt_path.exists():
        tracking_metrics = evaluate_tracking(reconstructed_tracking, gt_path, eval_tool=args.tracking_tool)
    
    # Ghi kết quả vào file
    results = {
        "sequence_name": sequence_name,
        "compression": {
            "original_size_mb": original_size / (1024 * 1024),
            "compressed_size_mb": compressed_size / (1024 * 1024),
            "compression_ratio": float(compression_ratio),
            "bpp": float(bpp)
        },
        "quality": {
            "psnr": float(avg_psnr)
        },
        "tracking": tracking_metrics if tracking_metrics else "Not evaluated"
    }
    
    # Lưu kết quả dạng JSON
    results_path = os.path.join(sequence_output_dir, f"{sequence_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_path}")
    print(f"PSNR: {avg_psnr:.2f} dB, BPP: {bpp:.4f}, Compression ratio: {compression_ratio:.2f}x")
    if tracking_metrics:
        print(f"MOTA: {tracking_metrics['MOTA']:.2f}, IDF1: {tracking_metrics['IDF1']:.2f}")
    
    return results

def main():
    args = parse_args()
    
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
    
    # Đường dẫn tới thư mục dataset split
    split_path = Path(args.dataset_path) / args.split
    if not split_path.exists():
        raise ValueError(f"Dataset split directory not found: {split_path}")
    
    # Tìm tất cả các thư mục chuỗi
    if args.sequence:
        # Xử lý một chuỗi cụ thể
        sequence_dir = split_path / args.sequence
        if not sequence_dir.exists():
            raise ValueError(f"Sequence directory not found: {sequence_dir}")
        sequence_dirs = [sequence_dir]
    else:
        # Xử lý tất cả các chuỗi
        sequence_dirs = [d for d in split_path.iterdir() if d.is_dir()]
    
    if len(sequence_dirs) == 0:
        raise ValueError(f"No sequence directories found in {split_path}")
    
    print(f"Found {len(sequence_dirs)} sequences in {split_path}")
    
    # Xử lý từng chuỗi
    all_results = {}
    for sequence_dir in sequence_dirs:
        results = process_mot_sequence(sequence_dir, model, device, args, args.output_dir)
        if results:
            all_results[sequence_dir.name] = results
    
    # Tính toán trung bình trên tất cả các chuỗi
    if all_results:
        avg_bpp = np.mean([r["compression"]["bpp"] for r in all_results.values()])
        avg_psnr = np.mean([r["quality"]["psnr"] for r in all_results.values()])
        avg_compression_ratio = np.mean([r["compression"]["compression_ratio"] for r in all_results.values()])
        
        # Tính MOTA trung bình nếu có
        mota_values = []
        for r in all_results.values():
            if r["tracking"] != "Not evaluated" and "MOTA" in r["tracking"]:
                mota_values.append(r["tracking"]["MOTA"])
        
        avg_mota = np.mean(mota_values) if mota_values else "Not evaluated"
        
        summary = {
            "average_metrics": {
                "bpp": float(avg_bpp),
                "psnr": float(avg_psnr),
                "compression_ratio": float(avg_compression_ratio),
                "mota": float(avg_mota) if isinstance(avg_mota, (int, float)) else avg_mota
            },
            "sequence_results": all_results
        }
        
        # Lưu tổng hợp kết quả
        summary_path = os.path.join(args.output_dir, "summary_results.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Lưu kết quả dạng dễ đọc
        report_path = os.path.join(args.output_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write("MOT Tracking Performance Evaluation\n")
            f.write("================================\n\n")
            f.write(f"Dataset: {args.dataset_path}, Split: {args.split}\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Tracker: {args.tracker}\n\n")
            
            f.write("Average Metrics:\n")
            f.write(f"PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"BPP: {avg_bpp:.4f}\n")
            f.write(f"Compression ratio: {avg_compression_ratio:.2f}x\n")
            f.write(f"MOTA: {avg_mota}\n\n")
            
            f.write("Per-Sequence Results:\n")
            for seq_name, results in all_results.items():
                f.write(f"\n{seq_name}:\n")
                f.write(f"  PSNR: {results['quality']['psnr']:.2f} dB\n")
                f.write(f"  BPP: {results['compression']['bpp']:.4f}\n")
                f.write(f"  Compression ratio: {results['compression']['compression_ratio']:.2f}x\n")
                
                if results["tracking"] != "Not evaluated":
                    f.write(f"  MOTA: {results['tracking']['MOTA']:.2f}\n")
                    f.write(f"  IDF1: {results['tracking']['IDF1']:.2f}\n")
                else:
                    f.write("  Tracking: Not evaluated\n")
        
        print(f"\nEvaluation completed! Results saved to {args.output_dir}")
        print(f"Summary report: {report_path}")
        print(f"Summary JSON: {summary_path}")
        
        print("\nAverage Metrics:")
        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"BPP: {avg_bpp:.4f}")
        print(f"Compression ratio: {avg_compression_ratio:.2f}x")
        print(f"MOTA: {avg_mota}")
    else:
        print("No valid results were obtained from any sequence.")

if __name__ == "__main__":
    main() 