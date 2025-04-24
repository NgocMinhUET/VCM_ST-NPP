#!/usr/bin/env python
"""
Script to run a comprehensive evaluation pipeline for task-aware video compression.
This includes compression comparison, tracking evaluation, and report generation.
"""

import os
import argparse
import subprocess
import time
import json
from pathlib import Path
import sys
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation pipeline")
    
    # Input parameters
    parser.add_argument("--sequence_path", type=str, required=True,
                        help="Path to input video sequence directory")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Maximum number of frames to process (0 = all frames)")
    
    # Compression parameters
    parser.add_argument("--h264_crf", type=str, default="18,23,28",
                        help="Comma-separated CRF values for H.264")
    parser.add_argument("--h265_crf", type=str, default="18,23,28",
                        help="Comma-separated CRF values for H.265")
    parser.add_argument("--vp9_crf", type=str, default="18,23,28",
                        help="Comma-separated CRF values for VP9")
    parser.add_argument("--our_qp", type=str, default="1,2,3",
                        help="Comma-separated QP values for our method")
    
    # Task parameters
    parser.add_argument("--tasks", type=str, default="detection,segmentation,tracking",
                        help="Comma-separated list of tasks to evaluate")
                        
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results/comprehensive_eval",
                        help="Directory to save evaluation results")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip evaluations that already have results")
    
    return parser.parse_args()

def run_command(cmd):
    """Run a command and return success (True) or failure (False)."""
    print(f"Running command: {cmd}")
    try:
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output to console
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        # Get return code
        return_code = process.poll()
        if return_code != 0:
            print(f"Command failed with return code {return_code}")
            print("Error output:")
            print(process.stderr.read())
            return False
        return True
        
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def run_evaluation(args):
    """Run the complete evaluation pipeline."""
    # Parse comma-separated values
    h264_crf_values = [int(crf) for crf in args.h264_crf.split(",")]
    h265_crf_values = [int(crf) for crf in args.h265_crf.split(",")]
    vp9_crf_values = [int(crf) for crf in args.vp9_crf.split(",")]
    our_qp_values = [int(qp) for qp in args.our_qp.split(",")]
    tasks = args.tasks.split(",")
    
    sequence_name = os.path.basename(args.sequence_path.rstrip("/"))
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"{sequence_name}_{experiment_timestamp}")
    
    # Create experiment directories
    os.makedirs(experiment_dir, exist_ok=True)
    compression_dir = os.path.join(experiment_dir, "compression")
    os.makedirs(compression_dir, exist_ok=True)
    task_dir = os.path.join(experiment_dir, "tasks")
    os.makedirs(task_dir, exist_ok=True)
    report_dir = os.path.join(experiment_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Save experiment configuration
    config = vars(args)
    config["experiment_dir"] = experiment_dir
    config["timestamp"] = experiment_timestamp
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Step 1: Run compression comparison
    print("\n" + "="*80)
    print("STEP 1: RUNNING COMPRESSION COMPARISON")
    print("="*80)
    
    compression_cmd = (
        f"python compare_compression_methods.py "
        f"--input_video {args.sequence_path} "
        f"--model_path {args.model_path} "
        f"--output_dir {compression_dir} "
    )
    
    if args.max_frames > 0:
        compression_cmd += f"--max_frames {args.max_frames} "
    
    # Add CRF and QP values
    compression_cmd += f"--h264_crf {args.h264_crf} "
    compression_cmd += f"--h265_crf {args.h265_crf} "
    compression_cmd += f"--vp9_crf {args.vp9_crf} "
    compression_cmd += f"--our_qp {args.our_qp} "
    
    if not run_command(compression_cmd):
        print("Compression comparison failed. Continuing to next steps...")
    
    # Step 2: Run task evaluations for each compression method and setting
    print("\n" + "="*80)
    print("STEP 2: RUNNING TASK EVALUATIONS")
    print("="*80)
    
    compression_results_file = os.path.join(compression_dir, "compression_results.json")
    if not os.path.exists(compression_results_file):
        print(f"Warning: Compression results file {compression_results_file} not found.")
        print("Task evaluations may not have all the necessary compressed videos.")
    else:
        with open(compression_results_file, "r") as f:
            compression_results = json.load(f)
            
        # Create directories for each task
        for task in tasks:
            task_specific_dir = os.path.join(task_dir, task)
            os.makedirs(task_specific_dir, exist_ok=True)
            
            print(f"\nEvaluating {task} performance...")
            
            # For each compression method and quality setting
            compression_methods = ["h264", "h265", "vp9", "ours"]
            quality_settings = {
                "h264": h264_crf_values,
                "h265": h265_crf_values,
                "vp9": vp9_crf_values,
                "ours": our_qp_values
            }
            
            for method in compression_methods:
                for quality in quality_settings[method]:
                    # Find the compressed video
                    video_key = f"{method}_crf{quality}" if method != "ours" else f"{method}_qp{quality}"
                    
                    if video_key in compression_results["compressed_videos"]:
                        compressed_video = compression_results["compressed_videos"][video_key]
                        
                        # Run task evaluation
                        task_eval_cmd = (
                            f"python evaluate_task_performance.py "
                            f"--task {task} "
                            f"--input_video {compressed_video} "
                            f"--ground_truth {args.sequence_path} "
                            f"--output_dir {os.path.join(task_specific_dir, video_key)} "
                        )
                        
                        if args.max_frames > 0:
                            task_eval_cmd += f"--max_frames {args.max_frames} "
                        
                        if not run_command(task_eval_cmd):
                            print(f"Task evaluation for {task} on {video_key} failed. Continuing...")
                    else:
                        print(f"Warning: No compressed video found for {video_key}")
    
    # Step 3: Generate comprehensive report
    print("\n" + "="*80)
    print("STEP 3: GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    report_cmd = (
        f"python generate_comprehensive_report.py "
        f"--experiment_dir {experiment_dir} "
        f"--output_dir {report_dir} "
        f"--tasks {args.tasks} "
    )
    
    if not run_command(report_cmd):
        print("Report generation failed.")
    
    print("\n" + "="*80)
    print(f"EVALUATION COMPLETE. Results saved to {experiment_dir}")
    print("="*80)
    
    return experiment_dir

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"Sequence: {args.sequence_path}")
    print(f"Model: {args.model_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    experiment_dir = run_evaluation(args)
    
    print(f"Evaluation results saved to {experiment_dir}")
    print("Done!")

if __name__ == "__main__":
    main() 