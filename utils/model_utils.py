"""
Utility functions for model management including versioning, saving, and loading.
"""

import os
import json
import torch
import datetime
import hashlib
import numpy as np
from pathlib import Path
import subprocess
from typing import Dict, Any, Optional, Union, Tuple, List


def save_model_with_version(
    model: torch.nn.Module,
    model_dir: str,
    model_name: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    version: Optional[str] = None,
    save_format: str = "pt",
    git_info: bool = True
) -> str:
    """
    Save a model with version information and metadata.
    
    Args:
        model: The PyTorch model to save
        model_dir: Directory to save the model
        model_name: Base name for the model file
        optimizer: Optional optimizer to save
        epoch: Current training epoch
        metrics: Dictionary of metrics to save with the model
        version: Optional version string (if None, generates a timestamp)
        save_format: Format to save the model ('pt' for PyTorch or 'onnx' for ONNX)
        git_info: Whether to include git information in the metadata
        
    Returns:
        Path to the saved model file
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate version if not provided
    if version is None:
        version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model filename
    model_filename = f"{model_name}_v{version}.{save_format}"
    model_path = os.path.join(model_dir, model_filename)
    
    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "timestamp": datetime.datetime.now().isoformat(),
        "epoch": epoch,
    }
    
    # Add metrics to metadata
    if metrics is not None:
        metadata["metrics"] = metrics
    
    # Add git information if requested and available
    if git_info:
        try:
            # Get current commit hash
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                universal_newlines=True
            ).strip()
            
            # Get current branch
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                universal_newlines=True
            ).strip()
            
            # Add to metadata
            metadata["git"] = {
                "commit_hash": git_hash,
                "branch": git_branch
            }
        except (subprocess.SubprocessError, FileNotFoundError):
            # Git not available or not a git repository
            metadata["git"] = "Not available"
    
    # Save model in the specified format
    if save_format == "pt":
        # Save PyTorch checkpoint
        save_dict = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save the model
        torch.save(save_dict, model_path)
        
    elif save_format == "onnx":
        # Export to ONNX format (requires sample input)
        # This is a placeholder - you'll need to adapt this for your specific model
        raise NotImplementedError("ONNX export not implemented yet")
    
    # Save metadata separately for easier access
    metadata_path = os.path.join(model_dir, f"{model_name}_v{version}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")
    
    return model_path


def load_model_with_version(
    model: torch.nn.Module,
    model_path: str,
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a model with version information and metadata.
    
    Args:
        model: The PyTorch model to load weights into
        model_path: Path to the saved model file
        device: Device to load the model onto
        optimizer: Optional optimizer to load state into
        strict: Whether to strictly enforce that the keys in state_dict match model
        
    Returns:
        Tuple of (loaded_model, metadata)
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        # Direct model state dict
        model.load_state_dict(checkpoint, strict=strict)
    
    # Load optimizer state if provided and available
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Extract metadata if available
    metadata = checkpoint.get('metadata', {})
    
    # Move model to device
    model = model.to(device)
    
    print(f"Model loaded from {model_path}")
    if metadata:
        print(f"Model version: {metadata.get('version', 'unknown')}")
        print(f"Training epoch: {metadata.get('epoch', 'unknown')}")
    
    return model, metadata


def get_latest_model(model_dir: str, model_name: str) -> Optional[str]:
    """
    Find the latest version of a model in the specified directory.
    
    Args:
        model_dir: Directory containing model files
        model_name: Base name of the model
        
    Returns:
        Path to the latest model or None if not found
    """
    # Ensure directory exists
    if not os.path.exists(model_dir):
        print(f"Directory {model_dir} does not exist")
        return None
    
    # Find all model files matching the pattern
    model_files = list(Path(model_dir).glob(f"{model_name}_v*.pt"))
    
    # If no models found
    if not model_files:
        print(f"No models found for {model_name} in {model_dir}")
        return None
    
    # Sort by modification time (newest first)
    latest_model = str(sorted(model_files, key=os.path.getmtime, reverse=True)[0])
    
    print(f"Latest model: {latest_model}")
    return latest_model


def get_best_model(model_dir: str, model_name: str, metric: str = "loss", higher_better: bool = False) -> Optional[str]:
    """
    Find the best model according to a specific metric.
    
    Args:
        model_dir: Directory containing model files
        model_name: Base name of the model
        metric: Metric to use for comparison
        higher_better: Whether higher metric values are better
        
    Returns:
        Path to the best model or None if not found
    """
    # Ensure directory exists
    if not os.path.exists(model_dir):
        print(f"Directory {model_dir} does not exist")
        return None
    
    # Find all model metadata files
    metadata_files = list(Path(model_dir).glob(f"{model_name}_v*_metadata.json"))
    
    # If no metadata found
    if not metadata_files:
        print(f"No metadata found for {model_name} in {model_dir}")
        return None
    
    # Load metadata and find best model
    best_value = float('-inf') if higher_better else float('inf')
    best_model = None
    
    for metadata_file in metadata_files:
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check if metric exists in metadata
        if 'metrics' in metadata and metric in metadata['metrics']:
            value = metadata['metrics'][metric]
            
            # Compare with current best
            if (higher_better and value > best_value) or (not higher_better and value < best_value):
                best_value = value
                # Get corresponding model path
                model_path = str(metadata_file).replace('_metadata.json', '.pt')
                if os.path.exists(model_path):
                    best_model = model_path
    
    if best_model:
        print(f"Best model according to {metric}: {best_model}")
        print(f"Best {metric} value: {best_value}")
        return best_model
    else:
        print(f"No model found with metric {metric}")
        return None


def calculate_model_checksum(model: torch.nn.Module) -> str:
    """
    Calculate a checksum of model weights for verification.
    
    Args:
        model: PyTorch model
        
    Returns:
        Hexadecimal checksum string
    """
    # Get model parameters as a single tensor
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().flatten())
    
    # Concatenate all parameters
    all_params = np.concatenate(params)
    
    # Calculate SHA256 hash
    checksum = hashlib.sha256(all_params.tobytes()).hexdigest()
    
    return checksum 