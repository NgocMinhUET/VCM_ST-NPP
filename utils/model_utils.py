"""
Model utility functions for task-aware video compression.

This module provides utility functions for saving and loading models,
handling checkpoints, and managing model state.
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
import torch.nn as nn
import logging


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
    Calculate a checksum for model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Hexadecimal checksum string
    """
    # Get model parameters as a single flat array
    params = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    
    # Calculate MD5 checksum
    checksum = hashlib.md5(params.tobytes()).hexdigest()
    
    return checksum


def save_checkpoint(state: Dict[str, Any], filepath: str, is_best: bool = False, best_filepath: Optional[str] = None) -> None:
    """
    Save a training checkpoint with model state and training metadata.
    
    Args:
        state: Dictionary containing state to save, including:
            - model state_dict
            - optimizer state_dict
            - scheduler state_dict (optional)
            - epoch
            - best_val_loss (optional)
            - other metrics
        filepath: Path to save the checkpoint
        is_best: Whether this is the best model so far (for saving a copy)
        best_filepath: Path to save the best model (if is_best is True)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save checkpoint
    torch.save(state, filepath)
    
    # Save a copy if this is the best model
    if is_best:
        if best_filepath is None:
            # Default best filepath if not provided
            dirname = os.path.dirname(filepath)
            basename = os.path.basename(filepath)
            best_basename = f"best_{basename}"
            best_filepath = os.path.join(dirname, best_basename)
        
        # Copy to best model file
        torch.save(state, best_filepath)
        print(f"Best model saved to {best_filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
                    scheduler: Optional[Any] = None, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load a training checkpoint with model state and training metadata.
    
    Args:
        filepath: Path to the checkpoint
        model: The model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load the model onto
        
    Returns:
        Dictionary containing the checkpoint state
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError("No model state_dict found in checkpoint")
    
    # Load optimizer state if provided and available
    if optimizer is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        elif 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Warning: No optimizer state found in checkpoint")
    
    # Load scheduler state if provided and available
    if scheduler is not None:
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        elif 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Warning: No scheduler state found in checkpoint")
    
    # Move model to device
    model.to(device)
    
    print(f"Checkpoint loaded from {filepath}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    return checkpoint


def save_model(model: nn.Module, path: str, optimizer: Optional[torch.optim.Optimizer] = None, 
               epoch: Optional[int] = None, additional_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a PyTorch model to the specified path.
    
    Args:
        model: PyTorch model to save
        path: Path where the model will be saved
        optimizer: Optional optimizer to save state
        epoch: Optional epoch number to save
        additional_data: Optional dictionary with additional data to save
        
    Returns:
        None
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare the state dictionary
        checkpoint = {
            'model_state_dict': model.state_dict()
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add epoch if provided
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        # Add any additional data
        if additional_data is not None:
            checkpoint.update(additional_data)
        
        # Save the model
        torch.save(checkpoint, path)
        print(f"Model saved successfully to {path}")
        
    except Exception as e:
        print(f"Error saving model to {path}: {str(e)}")
        raise


def load_model(model: nn.Module, path: str, optimizer: Optional[torch.optim.Optimizer] = None, 
              map_location: Optional[Union[str, torch.device]] = None, strict: bool = True) -> Dict[str, Any]:
    """
    Load a PyTorch model from the specified path.
    
    Args:
        model: PyTorch model to load weights into
        path: Path to the saved model
        optimizer: Optional optimizer to load state into
        map_location: Optional device mapping for loading model on different devices
        strict: Whether to strictly enforce that the keys in state_dict match the keys in model
        
    Returns:
        Dictionary containing any additional saved data
    """
    try:
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the checkpoint
        checkpoint = torch.load(path, map_location=map_location)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        else:
            model.load_state_dict(checkpoint, strict=strict)
            return {}  # No additional data if only the state dict was saved
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Remove model and optimizer states from the returned dictionary
        additional_data = {k: v for k, v in checkpoint.items() 
                          if k not in ['model_state_dict', 'optimizer_state_dict']}
        
        print(f"Model loaded successfully from {path}")
        
        return additional_data
        
    except Exception as e:
        print(f"Error loading model from {path}: {str(e)}")
        raise


def create_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, 
                    loss: float, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Create a checkpoint dictionary with model state and training information.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer used for training
        epoch: Current epoch number
        loss: Current loss value
        metrics: Optional dictionary of evaluation metrics
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    return checkpoint


def save_checkpoint(checkpoint: Dict[str, Any], path: str, is_best: bool = False) -> None:
    """
    Save a checkpoint to the specified path, with option to save as best model.
    
    Args:
        checkpoint: Checkpoint dictionary to save
        path: Path where to save the checkpoint
        is_best: Whether this is the best model so far
        
    Returns:
        None
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the checkpoint
        torch.save(checkpoint, path)
        
        # If this is the best model, save it separately
        if is_best:
            best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
        
        print(f"Checkpoint saved to {path}")
        
    except Exception as e:
        print(f"Error saving checkpoint to {path}: {str(e)}")
        raise


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                   map_location: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
    """
    Load a checkpoint from the specified path.
    
    Args:
        path: Path to the checkpoint
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        map_location: Optional device mapping for loading model on different devices
        
    Returns:
        Checkpoint dictionary with additional data
    """
    return load_model(model, path, optimizer, map_location)


def get_model_size(model: nn.Module) -> float:
    """
    Calculate the size of a PyTorch model in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test code
if __name__ == "__main__":
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 3, kernel_size=3, padding=1)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test saving and loading
    save_path = "test_model.pth"
    save_model(model, save_path, optimizer, epoch=10, additional_data={"test_accuracy": 0.95})
    
    # Create a new model to load into
    new_model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 3, kernel_size=3, padding=1)
    )
    
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    
    # Load the model
    additional_data = load_model(new_model, save_path, new_optimizer)
    
    print(f"Additional data: {additional_data}")
    print(f"Model size: {get_model_size(new_model):.2f} MB")
    print(f"Trainable parameters: {count_parameters(new_model)}") 