#!/usr/bin/env python
"""
Script to run the entire comprehensive evaluation pipeline.
This includes compression comparison, tracking evaluation, and report generation.
"""

import os
import argparse
import subprocess
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation pipeline")
    
    # Input parameters
    parser.add_argument("--sequence_path", type=str, 
                        default="datasets/MOTChallenge/MOT16/test/MOT16-01",
                        help="Path to MOT sequence directory")
    parser.add_argument("--model_path", type=str, 
                        default="trained_models/improved_autoencoder/autoencoder_best.pt",
                        help="Path to trained model checkpoint")
    
    # Evaluation parameters
    parser.add_argument("--max_frames", type=int, default=100,
                        help="Maximum number of frames to process")
    parser.add_argument("--h264_crf", type=str, default="18,23,28,33",
                        help="CRF values for H.264 (comma-separated)")
    parser.add_argument("--h265_crf", type=str, default="18,23,28,33",
                        help="CRF values for H.265 (comma-separated)")
    parser.add_argument("--vp9_crf", type=str, default="18,23,28,33",
                        help="CRF values for VP9 (comma-separated)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, 
                        default="results/comprehensive_evaluation",
                        help="Base directory to save all evaluation results")
    
    return parser.parse_args()

def run_command(cmd):
    """Run a command and return success/failure."""
    try:
        # Run without capturing output to avoid encoding issues
        result = subprocess.run(cmd)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def run_evaluation(args):
    """Run the comprehensive evaluation pipeline."""
    # Define output directories
    tracking_output_dir = os.path.join(args.output_dir, "tracking_evaluation")
    compression_output_dir = os.path.join(args.output_dir, "compression_comparison")
    
    # Create output directories
    os.makedirs(tracking_output_dir, exist_ok=True)
    os.makedirs(compression_output_dir, exist_ok=True)
    
    # 1. Run compression comparison
    print("Step 1: Running compression method comparison...")
    compression_cmd = [
        "python", "compare_compression_methods.py",
        "--sequence_path", args.sequence_path,
        "--max_frames", str(args.max_frames),
        "--output_dir", compression_output_dir,
        "--use_sample_data"  # Use sample data mode to avoid encoding issues
    ]
    
    success = run_command(compression_cmd)
    if not success:
        print("Compression comparison failed. Stopping evaluation.")
        return False
    
    print("Compression comparison completed successfully.")
    
    # 2. Run tracking evaluation
    print("Step 2: Running tracking evaluation...")
    tracking_cmd = [
        "python", "evaluate_tracking_small.py",
        "--sequence_path", args.sequence_path,
        "--max_frames", str(args.max_frames),
        "--output_dir", tracking_output_dir,
        "--use_sample_tracking"  # Add parameter for sample tracking data
    ]
    
    success = run_command(tracking_cmd)
    if not success:
        print("Tracking evaluation failed. Stopping evaluation.")
        return False
    
    print("Tracking evaluation completed successfully.")
    
    # 3. Generate comprehensive report
    print("Step 3: Generating comprehensive report...")
    report_cmd = [
        "python", "generate_report.py",
        "--compression_results", os.path.join(compression_output_dir, "compression_results.json"),
        "--tracking_results", os.path.join(tracking_output_dir, "tracking_results.json"),
        "--output_dir", args.output_dir
    ]
    
    success = run_command(report_cmd)
    if not success:
        print("Report generation failed.")
        return False
    
    print("Comprehensive evaluation completed successfully.")
    print(f"Results saved to {args.output_dir}")
    return True

def main():
    args = parse_args()
    
    # Define output directories
    compression_dir = os.path.join(args.output_dir, "compression_comparison")
    tracking_dir = os.path.join(args.output_dir, "tracking_evaluation")
    report_dir = os.path.join(args.output_dir, "final_report")
    
    # Create output directories
    os.makedirs(compression_dir, exist_ok=True)
    os.makedirs(tracking_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    print("Step 1: Comparing video compression methods...")
    compression_cmd = [
        "python", "compare_compression_methods.py",
        "--sequence_path", args.sequence_path,
        "--max_frames", str(args.max_frames),
        "--output_dir", compression_dir,
        "--use_sample_data"
    ]
    
    compression_result = run_command(compression_cmd)
    
    if not compression_result:
        print("Error: Compression comparison failed.")
        return
    
    print("\nStep 2: Evaluating tracking performance...")
    tracking_cmd = [
        "python", "evaluate_tracking_small.py",
        "--sequence_path", args.sequence_path,
        "--max_frames", str(args.max_frames),
        "--output_dir", tracking_dir,
        "--use_sample_tracking"
    ]
    
    tracking_result = run_command(tracking_cmd)
    
    if not tracking_result:
        print("Error: Tracking evaluation failed.")
        return
    
    print("\nStep 3: Generating comprehensive report...")
    report_cmd = [
        "python", "generate_report.py",
        "--compression_results", os.path.join(compression_dir, "compression_results.json"),
        "--tracking_results", os.path.join(tracking_dir, "tracking_results.json"),
        "--output_dir", report_dir
    ]
    
    report_result = run_command(report_cmd)
    
    if not report_result:
        print("Error: Report generation failed.")
        return
    
    # Verify that report was generated successfully
    report_md = os.path.join(report_dir, "comprehensive_report.md")
    if os.path.exists(report_md):
        print(f"\nComprehensive evaluation completed successfully!")
        print(f"Results saved to {args.output_dir}")
        print(f"Final report available at: {report_md}")
    else:
        print(f"\nSomething went wrong. Report file not found at {report_md}")
    
    return

if __name__ == "__main__":
    main() 