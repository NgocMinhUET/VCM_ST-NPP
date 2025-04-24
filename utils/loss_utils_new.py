"""
Loss utility functions for task-aware video preprocessing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss combining distortion, rate, and task losses."""
    
    def __init__(self, distortion_weight=1.0, rate_weight=0.01, task_weight=1.0):
        super().__init__()
        self.distortion_weight = distortion_weight
        self.rate_weight = rate_weight
        self.task_weight = task_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, original, reconstructed, rate_estimate, task_loss):
        distortion = self.mse_loss(original, reconstructed)
        total_loss = (
            self.distortion_weight * distortion +
            self.rate_weight * rate_estimate.mean() +
            self.task_weight * task_loss
        )
        
        return {
            'total_loss': total_loss,
            'distortion_loss': distortion,
            'rate_loss': rate_estimate.mean(),
            'task_loss': task_loss
        }


class DetectionLoss(nn.Module):
    """YOLO-style detection loss."""
    
    def __init__(self, bbox_weight=1.0, conf_weight=1.0, cls_weight=0.5):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.conf_weight = conf_weight
        self.cls_weight = cls_weight
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
    
    def forward(self, predictions, targets, anchors):
        bbox_loss = torch.tensor(0.0, device=predictions[0].device)
        conf_loss = torch.tensor(0.0, device=predictions[0].device)
        cls_loss = torch.tensor(0.0, device=predictions[0].device)
        
        # Calculation logic omitted for brevity
        # Implement YOLO detection loss calculation here
        
        batch_size = predictions[0].shape[0]
        bbox_loss = self.bbox_weight * bbox_loss / batch_size
        conf_loss = self.conf_weight * conf_loss / batch_size
        cls_loss = self.cls_weight * cls_loss / batch_size
        
        total_loss = bbox_loss + conf_loss + cls_loss
        
        return {
            'total_loss': total_loss,
            'bbox_loss': bbox_loss,
            'conf_loss': conf_loss,
            'cls_loss': cls_loss
        }


class SegmentationLoss(nn.Module):
    """Combined cross-entropy and Dice loss for segmentation."""
    
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def dice_loss(self, inputs, targets, smooth=1.0):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def forward(self, predictions, targets):
        ce_loss = self.ce_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss
        }


class TrackingLoss(nn.Module):
    """Loss for Siamese tracking networks."""
    
    def __init__(self, cls_weight=1.0, reg_weight=1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = nn.SmoothL1Loss()
    
    def forward(self, cls_pred, reg_pred, gt_cls, gt_reg):
        # Classification loss
        cls_loss = self.cls_loss_fn(cls_pred, gt_cls)
        
        # Regression loss (only for positive examples)
        pos_mask = (gt_cls > 0.5).float().unsqueeze(1)
        reg_loss = self.reg_loss_fn(reg_pred * pos_mask, gt_reg * pos_mask)
        
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss
        }


class RateLoss(nn.Module):
    """Entropy-based rate estimation loss."""
    
    def __init__(self, lambda_rate=0.01, eps=1e-10):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.eps = eps
    
    def forward(self, y_hat, importance_map=None):
        entropy = -torch.log2(y_hat + self.eps) * y_hat
        entropy = entropy.sum(dim=1)  # Sum over centers dimension
        
        if importance_map is not None:
            entropy = entropy * importance_map.squeeze(1)
        
        rate = entropy.mean(dim=(1, 2))  # Average over spatial dimensions
        
        return self.lambda_rate * rate.mean()


def get_loss_function(task_type, **kwargs):
    """Factory function for task-specific losses."""
    if task_type == 'detection':
        return DetectionLoss(**kwargs)
    elif task_type == 'segmentation':
        return SegmentationLoss(**kwargs)
    elif task_type == 'tracking':
        return TrackingLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def get_combined_loss(task_type, distortion_weight=1.0, rate_weight=0.01, task_weight=1.0, **kwargs):
    """Get combined rate-distortion and task-specific loss functions."""
    task_loss = get_loss_function(task_type, **kwargs)
    rd_loss = RateDistortionLoss(
        distortion_weight=distortion_weight,
        rate_weight=rate_weight,
        task_weight=task_weight
    )
    
    return rd_loss, task_loss
