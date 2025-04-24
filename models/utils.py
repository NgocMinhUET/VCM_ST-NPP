import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import time
import traceback

def set_random_seed(seed):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def get_optimizer(model, args):
    """
    Create optimizer based on args
    
    Args:
        model: PyTorch model
        args: Command line arguments
        
    Returns:
        PyTorch optimizer
    """
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                         weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

def get_scheduler(optimizer, args):
    """
    Create learning rate scheduler based on args
    
    Args:
        optimizer: PyTorch optimizer
        args: Command line arguments
        
    Returns:
        PyTorch scheduler or None if not specified
    """
    if not args.scheduler or args.scheduler.lower() == 'none':
        return None
    elif args.scheduler.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, 
                                         gamma=args.lr_gamma)
    elif args.scheduler.lower() == 'multistep':
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, 
                                             gamma=args.lr_gamma)
    elif args.scheduler.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_gamma, 
                                                   patience=args.lr_patience, verbose=True)
    elif args.scheduler.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise ValueError(f"Scheduler {args.scheduler} not supported")

def get_criterion(args):
    """
    Create loss function based on args
    
    Args:
        args: Command line arguments
        
    Returns:
        Loss function
    """
    from losses import TaskAwareLoss
    
    try:
        return TaskAwareLoss(
            lambda_distortion=args.lambda_distortion,
            lambda_rate=args.lambda_rate,
            lambda_task=args.lambda_task,
            task_type=args.task_type
        )
    except Exception as e:
        print(f"Error creating TaskAwareLoss: {e}")
        traceback.print_exc()
        raise

def get_data_loaders(args):
    """
    Create data loaders based on args
    
    Args:
        args: Command line arguments
        
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    try:
        # Get datasets
        from datasets import get_datasets
        train_dataset, val_dataset = get_datasets(args)
        print(f"Created datasets: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        traceback.print_exc()
        raise

def save_model(model, optimizer, scheduler, epoch, best_val_loss, metrics, path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: PyTorch scheduler (or None)
        epoch: Current epoch
        best_val_loss: Best validation loss
        metrics: Dictionary of metrics
        path: Path to save checkpoint
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'metrics': metrics
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        torch.save(checkpoint, path)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        traceback.print_exc()
        return False

def load_model(model, optimizer=None, scheduler=None, path=None, device='cuda'):
    """
    Load model from checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
        scheduler: PyTorch scheduler (optional)
        path: Path to checkpoint file
        device: Device to load the model on
        
    Returns:
        model, optimizer, scheduler, epoch, best_val_loss, metrics
    """
    if path is None or not os.path.isfile(path):
        print(f"Checkpoint not found at {path}")
        return model, optimizer, scheduler, 0, float('inf'), {}
    
    try:
        checkpoint = torch.load(path, map_location=device)
        
        # Load model state
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Error loading model state: {e}")
            print("Attempting partial load...")
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                              if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Partially loaded model state ({len(pretrained_dict)}/{len(model_dict)} keys)")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer state to device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except Exception as e:
                print(f"Error loading optimizer state: {e}")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"Error loading scheduler state: {e}")
        
        epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        metrics = checkpoint.get('metrics', {})
        
        print(f"Loaded checkpoint from epoch {epoch} with validation loss {best_val_loss:.4f}")
        return model, optimizer, scheduler, epoch, best_val_loss, metrics
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        traceback.print_exc()
        return model, optimizer, scheduler, 0, float('inf'), {} 