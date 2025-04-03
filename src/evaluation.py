import os
import numpy as np
import tensorflow as tf
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import tempfile
from scipy.interpolate import interp1d

# Import our pipeline modules
from src.pipeline import VCMPreprocessingPipeline
from utils.codec_utils import run_hevc_pipeline, calculate_bpp
from utils.video_utils import load_video, preprocess_frames, create_sliding_windows

class EvaluationModule:
    """
    Comprehensive evaluation module for the VCM preprocessing pipeline.
    
    This module evaluates the pipeline on various computer vision tasks:
    - Object Detection (COCO Video)
    - Semantic Segmentation (KITTI Semantic)
    - Object Tracking (MOTChallenge)
    
    It provides metrics for both compression efficiency and task performance.
    """
    
    def __init__(self, pipeline_model=None, detection_model=None, segmentation_model=None, tracking_model=None):
        """
        Initialize the evaluation module.
        
        Args:
            pipeline_model: Trained VCM preprocessing pipeline model
            detection_model: Object detection model for evaluation
            segmentation_model: Semantic segmentation model for evaluation
            tracking_model: Object tracking model for evaluation
        """
        self.pipeline_model = pipeline_model
        self.detection_model = detection_model
        self.segmentation_model = segmentation_model
        self.tracking_model = tracking_model
        
    def load_datasets(self, detection_dataset_path=None, segmentation_dataset_path=None, tracking_dataset_path=None):
        """
        Load evaluation datasets.
        
        Args:
            detection_dataset_path: Path to COCO Video dataset
            segmentation_dataset_path: Path to KITTI Semantic dataset
            tracking_dataset_path: Path to MOTChallenge dataset
            
        Returns:
            Dictionary containing loaded datasets
        """
        self.datasets = {}
        
        # Load detection dataset (COCO Video)
        if detection_dataset_path is not None:
            print(f"Loading detection dataset from {detection_dataset_path}...")
            # Implement dataset loading based on COCO Video format
            # This is a placeholder for actual dataset loading code
            self.datasets['detection'] = {'path': detection_dataset_path}
            
        # Load segmentation dataset (KITTI Semantic)
        if segmentation_dataset_path is not None:
            print(f"Loading segmentation dataset from {segmentation_dataset_path}...")
            # Implement dataset loading based on KITTI Semantic format
            # This is a placeholder for actual dataset loading code
            self.datasets['segmentation'] = {'path': segmentation_dataset_path}
            
        # Load tracking dataset (MOTChallenge)
        if tracking_dataset_path is not None:
            print(f"Loading tracking dataset from {tracking_dataset_path}...")
            # Implement dataset loading based on MOTChallenge format
            # This is a placeholder for actual dataset loading code
            self.datasets['tracking'] = {'path': tracking_dataset_path}
            
        return self.datasets
    
    def evaluate_bdrate(self, original_video, preprocessed_video, qp_range=(22, 27, 32, 37), reference_codec='hevc'):
        """
        Evaluate BD-Rate (Bjøntegaard Delta Rate) to measure bitrate savings.
        
        Args:
            original_video: Original video frames
            preprocessed_video: Preprocessed video frames
            qp_range: List of QP values to evaluate
            reference_codec: Reference codec ('hevc', 'vvc', etc.)
            
        Returns:
            BD-Rate value (percentage of bitrate savings)
        """
        # Encode original and preprocessed videos at different QP values
        original_results = []
        preprocessed_results = []
        
        for qp in qp_range:
            # Encode and decode original video
            decoded_original = run_hevc_pipeline(original_video, qp=qp)
            original_bpp = calculate_bpp(f"temp_original_{qp}.mp4", original_video)
            original_psnr = calculate_psnr(original_video, decoded_original)
            original_results.append((original_bpp, original_psnr))
            
            # Encode and decode preprocessed video
            decoded_preprocessed = run_hevc_pipeline(preprocessed_video, qp=qp)
            preprocessed_bpp = calculate_bpp(f"temp_preprocessed_{qp}.mp4", preprocessed_video)
            preprocessed_psnr = calculate_psnr(preprocessed_video, decoded_preprocessed)
            preprocessed_results.append((preprocessed_bpp, preprocessed_psnr))
        
        # Calculate BD-Rate
        bdrate = calculate_bdrate(original_results, preprocessed_results)
        
        return bdrate
    
    def evaluate_detection(self, original_video, preprocessed_video, qp=27):
        """
        Evaluate object detection performance.
        
        Args:
            original_video: Original video frames
            preprocessed_video: Preprocessed video frames
            qp: Quantization Parameter
            
        Returns:
            Dictionary with detection metrics (mAP)
        """
        if self.detection_model is None:
            raise ValueError("Detection model must be provided for evaluation")
            
        # Encode and decode videos
        decoded_original = run_hevc_pipeline(original_video, qp=qp)
        decoded_preprocessed = run_hevc_pipeline(preprocessed_video, qp=qp)
        
        # Run detection on decoded videos
        detections_original = run_detection(self.detection_model, decoded_original)
        detections_preprocessed = run_detection(self.detection_model, decoded_preprocessed)
        
        # Calculate mAP (mean Average Precision)
        map_original = calculate_map(detections_original, ground_truth=None)  # Ground truth would be provided in actual implementation
        map_preprocessed = calculate_map(detections_preprocessed, ground_truth=None)
        
        # Calculate bitrate
        bpp_original = calculate_bpp(f"temp_original_{qp}.mp4", original_video)
        bpp_preprocessed = calculate_bpp(f"temp_preprocessed_{qp}.mp4", preprocessed_video)
        
        return {
            'mAP_original': map_original,
            'mAP_preprocessed': map_preprocessed,
            'bpp_original': bpp_original,
            'bpp_preprocessed': bpp_preprocessed,
            'bitrate_savings': (bpp_original - bpp_preprocessed) / bpp_original * 100
        }
    
    def evaluate_segmentation(self, original_video, preprocessed_video, qp=27):
        """
        Evaluate semantic segmentation performance.
        
        Args:
            original_video: Original video frames
            preprocessed_video: Preprocessed video frames
            qp: Quantization Parameter
            
        Returns:
            Dictionary with segmentation metrics (mIoU)
        """
        if self.segmentation_model is None:
            raise ValueError("Segmentation model must be provided for evaluation")
            
        # Encode and decode videos
        decoded_original = run_hevc_pipeline(original_video, qp=qp)
        decoded_preprocessed = run_hevc_pipeline(preprocessed_video, qp=qp)
        
        # Run segmentation on decoded videos
        segmentations_original = run_segmentation(self.segmentation_model, decoded_original)
        segmentations_preprocessed = run_segmentation(self.segmentation_model, decoded_preprocessed)
        
        # Calculate mIoU (mean Intersection over Union)
        miou_original = calculate_miou(segmentations_original, ground_truth=None)  # Ground truth would be provided in actual implementation
        miou_preprocessed = calculate_miou(segmentations_preprocessed, ground_truth=None)
        
        # Calculate bitrate
        bpp_original = calculate_bpp(f"temp_original_{qp}.mp4", original_video)
        bpp_preprocessed = calculate_bpp(f"temp_preprocessed_{qp}.mp4", preprocessed_video)
        
        return {
            'mIoU_original': miou_original,
            'mIoU_preprocessed': miou_preprocessed,
            'bpp_original': bpp_original,
            'bpp_preprocessed': bpp_preprocessed,
            'bitrate_savings': (bpp_original - bpp_preprocessed) / bpp_original * 100
        }
    
    def evaluate_tracking(self, original_video, preprocessed_video, qp=27):
        """
        Evaluate object tracking performance.
        
        Args:
            original_video: Original video frames
            preprocessed_video: Preprocessed video frames
            qp: Quantization Parameter
            
        Returns:
            Dictionary with tracking metrics (MOTA, IDF1)
        """
        if self.tracking_model is None:
            raise ValueError("Tracking model must be provided for evaluation")
            
        # Encode and decode videos
        decoded_original = run_hevc_pipeline(original_video, qp=qp)
        decoded_preprocessed = run_hevc_pipeline(preprocessed_video, qp=qp)
        
        # Run tracking on decoded videos
        tracking_original = run_tracking(self.tracking_model, decoded_original)
        tracking_preprocessed = run_tracking(self.tracking_model, decoded_preprocessed)
        
        # Calculate tracking metrics
        mota_original, idf1_original = calculate_tracking_metrics(tracking_original, ground_truth=None)  # Ground truth would be provided in actual implementation
        mota_preprocessed, idf1_preprocessed = calculate_tracking_metrics(tracking_preprocessed, ground_truth=None)
        
        # Calculate bitrate
        bpp_original = calculate_bpp(f"temp_original_{qp}.mp4", original_video)
        bpp_preprocessed = calculate_bpp(f"temp_preprocessed_{qp}.mp4", preprocessed_video)
        
        return {
            'MOTA_original': mota_original,
            'MOTA_preprocessed': mota_preprocessed,
            'IDF1_original': idf1_original,
            'IDF1_preprocessed': idf1_preprocessed,
            'bpp_original': bpp_original,
            'bpp_preprocessed': bpp_preprocessed,
            'bitrate_savings': (bpp_original - bpp_preprocessed) / bpp_original * 100
        }
    
    def run_comprehensive_evaluation(self, video_path, time_steps=16, qp_range=(22, 27, 32, 37)):
        """
        Run a comprehensive evaluation on a video across different tasks and QP values.
        
        Args:
            video_path: Path to the video file
            time_steps: Number of time steps for temporal processing
            qp_range: List of QP values to evaluate
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        # Initialize results dictionary
        results = {
            'bdrate': {},
            'detection': {},
            'segmentation': {},
            'tracking': {}
        }
        
        # Load and preprocess video
        print(f"Loading video from {video_path}...")
        frames = load_video(video_path)
        if len(frames) < time_steps:
            raise ValueError(f"Video has only {len(frames)} frames, but {time_steps} are required.")
            
        processed_frames = preprocess_frames(frames)
        video_segments = create_sliding_windows(processed_frames, window_size=time_steps)
        
        # Initialize pipeline if not provided
        if self.pipeline_model is None:
            print("Initializing VCM preprocessing pipeline...")
            pipeline = VCMPreprocessingPipeline(
                input_shape=(None, 224, 224, 3),
                time_steps=time_steps
            )
            st_npp_model, qal_model, _ = pipeline.build()
        else:
            pipeline = self.pipeline_model
            
        # Preprocess video using our pipeline
        print("Preprocessing video segments...")
        preprocessed_segments = []
        for segment in video_segments:
            # Extract current frame and temporal sequence
            current_frame = segment[-1]
            # Process through the pipeline
            features = pipeline.preprocess_video(
                np.expand_dims(segment, 0),  # Add batch dimension
                qp=27  # Using a default QP for preprocessing
            )
            preprocessed_segments.append(features[0])  # Remove batch dimension
            
        # Evaluate BD-Rate across QP range
        print("Evaluating BD-Rate...")
        results['bdrate'] = self.evaluate_bdrate(processed_frames, preprocessed_segments, qp_range=qp_range)
        
        # Evaluate object detection at middle QP
        mid_qp = qp_range[len(qp_range) // 2]
        if self.detection_model is not None:
            print(f"Evaluating object detection at QP={mid_qp}...")
            results['detection'] = self.evaluate_detection(processed_frames, preprocessed_segments, qp=mid_qp)
        
        # Evaluate semantic segmentation at middle QP
        if self.segmentation_model is not None:
            print(f"Evaluating semantic segmentation at QP={mid_qp}...")
            results['segmentation'] = self.evaluate_segmentation(processed_frames, preprocessed_segments, qp=mid_qp)
        
        # Evaluate object tracking at middle QP
        if self.tracking_model is not None:
            print(f"Evaluating object tracking at QP={mid_qp}...")
            results['tracking'] = self.evaluate_tracking(processed_frames, preprocessed_segments, qp=mid_qp)
        
        return results

# Helper functions for evaluation

def calculate_psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio between original and reconstructed frames."""
    if isinstance(original, list):
        original = np.stack(original)
    if isinstance(reconstructed, list):
        reconstructed = np.stack(reconstructed)
        
    # Ensure same shape
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
        
    # Calculate MSE
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100  # Perfect reconstruction
        
    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def calculate_bdrate(original_results, preprocessed_results):
    """
    Calculate BD-Rate (Bjøntegaard Delta Rate).
    
    Args:
        original_results: List of (bitrate, PSNR) tuples for original videos
        preprocessed_results: List of (bitrate, PSNR) tuples for preprocessed videos
        
    Returns:
        BD-Rate value (percentage of bitrate savings)
    """
    # Extract bitrates and PSNRs
    original_bitrates = np.array([res[0] for res in original_results])
    original_psnrs = np.array([res[1] for res in original_results])
    preprocessed_bitrates = np.array([res[0] for res in preprocessed_results])
    preprocessed_psnrs = np.array([res[1] for res in preprocessed_results])
    
    # Convert bitrates to logarithmic scale
    log_original_bitrates = np.log(original_bitrates)
    log_preprocessed_bitrates = np.log(preprocessed_bitrates)
    
    # Create interpolation functions
    original_interp = interp1d(original_psnrs, log_original_bitrates, kind='cubic')
    preprocessed_interp = interp1d(preprocessed_psnrs, log_preprocessed_bitrates, kind='cubic')
    
    # Find the overlapping PSNR range
    min_psnr = max(min(original_psnrs), min(preprocessed_psnrs))
    max_psnr = min(max(original_psnrs), max(preprocessed_psnrs))
    
    # Sample points in the overlapping range
    psnr_samples = np.linspace(min_psnr, max_psnr, 100)
    
    # Calculate the average bitrate difference
    log_orig_bitrate_samples = original_interp(psnr_samples)
    log_prepr_bitrate_samples = preprocessed_interp(psnr_samples)
    avg_log_bitrate_diff = np.mean(log_prepr_bitrate_samples - log_orig_bitrate_samples)
    
    # Convert to percentage
    bdrate = (np.exp(avg_log_bitrate_diff) - 1) * 100
    
    return bdrate

def run_detection(model, frames):
    """
    Run object detection on frames.
    
    This is a placeholder function. In a real implementation, 
    you would use the provided detection model to detect objects in the frames.
    
    Args:
        model: Object detection model
        frames: List or array of frames
        
    Returns:
        List of detection results for each frame
    """
    # Placeholder - would be replaced with actual detection code
    detection_results = []
    for frame in frames:
        # Run detection on each frame
        # This would use the provided model to generate detections
        detections = {'boxes': [], 'scores': [], 'classes': []}
        detection_results.append(detections)
        
    return detection_results

def calculate_map(detections, ground_truth):
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    This is a placeholder function. In a real implementation,
    you would calculate mAP by comparing detections to ground truth.
    
    Args:
        detections: List of detection results
        ground_truth: Ground truth annotations
        
    Returns:
        mAP value
    """
    # Placeholder - would be replaced with actual mAP calculation
    # For demo purposes, return a random value between 0.7 and 0.9
    return np.random.uniform(0.7, 0.9)

def run_segmentation(model, frames):
    """
    Run semantic segmentation on frames.
    
    This is a placeholder function. In a real implementation,
    you would use the provided segmentation model.
    
    Args:
        model: Semantic segmentation model
        frames: List or array of frames
        
    Returns:
        List of segmentation results for each frame
    """
    # Placeholder - would be replaced with actual segmentation code
    segmentation_results = []
    for frame in frames:
        # Run segmentation on each frame
        # This would use the provided model to generate segmentations
        segmentation = np.zeros(frame.shape[:2], dtype=np.int32)
        segmentation_results.append(segmentation)
        
    return segmentation_results

def calculate_miou(segmentations, ground_truth):
    """
    Calculate mean Intersection over Union (mIoU) for semantic segmentation.
    
    This is a placeholder function. In a real implementation,
    you would calculate mIoU by comparing segmentations to ground truth.
    
    Args:
        segmentations: List of segmentation results
        ground_truth: Ground truth segmentations
        
    Returns:
        mIoU value
    """
    # Placeholder - would be replaced with actual mIoU calculation
    # For demo purposes, return a random value between 0.6 and 0.8
    return np.random.uniform(0.6, 0.8)

def run_tracking(model, frames):
    """
    Run object tracking on frames.
    
    This is a placeholder function. In a real implementation,
    you would use the provided tracking model.
    
    Args:
        model: Object tracking model
        frames: List or array of frames
        
    Returns:
        Tracking results across frames
    """
    # Placeholder - would be replaced with actual tracking code
    tracking_results = []
    for i, frame in enumerate(frames):
        # Run tracking on each frame
        # This would use the provided model to track objects
        tracks = {'ids': [], 'boxes': []}
        tracking_results.append(tracks)
        
    return tracking_results

def calculate_tracking_metrics(tracking_results, ground_truth):
    """
    Calculate tracking metrics: MOTA (Multiple Object Tracking Accuracy) and IDF1.
    
    This is a placeholder function. In a real implementation,
    you would calculate MOTA and IDF1 by comparing tracking results to ground truth.
    
    Args:
        tracking_results: Tracking results
        ground_truth: Ground truth tracking annotations
        
    Returns:
        Tuple of (MOTA, IDF1) values
    """
    # Placeholder - would be replaced with actual metric calculation
    # For demo purposes, return random values
    mota = np.random.uniform(0.6, 0.85)
    idf1 = np.random.uniform(0.55, 0.8)
    
    return mota, idf1 