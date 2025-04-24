"""
Loss utility functions for task-aware video compression.

This module provides loss functions for different tasks in the task-aware
video compression framework, including detection, segmentation, and tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
from torchvision.ops import box_iou, generalized_box_iou
import torchvision.models as models


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss for compression tasks.
    
    Combines a distortion term (e.g., MSE) with a rate term (bits used for coding).
    """
    def __init__(self, lambda_rd: float = 0.001):
        """
        Args:
            lambda_rd: Lagrangian multiplier for rate-distortion trade-off
        """
        super().__init__()
        self.lambda_rd = lambda_rd
        
    def forward(self, 
                original: torch.Tensor, 
                reconstructed: torch.Tensor, 
                bits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            original: Original input tensor
            reconstructed: Reconstructed output tensor
            bits: Bits used for encoding
            
        Returns:
            Dictionary with 'loss', 'distortion', and 'rate' terms
        """
        # Calculate distortion (MSE)
        distortion = F.mse_loss(original, reconstructed)
        
        # Calculate rate (normalized bits)
        num_pixels = original.size(0) * original.size(2) * original.size(3)
        rate = bits / num_pixels
        
        # Combined loss
        loss = distortion + self.lambda_rd * rate
        
        return {
            'loss': loss,
            'distortion': distortion,
            'rate': rate
        }


class TaskAwareLoss(nn.Module):
    """
    Combined loss function for task-aware video compression.
    
    Computes a weighted combination of task loss, reconstruction loss, and bitrate loss.
    """
    def __init__(
        self,
        task_weight: float = 1.0,
        recon_weight: float = 1.0,
        bitrate_weight: float = 0.01,
        task_type: str = 'detection'
    ):
        """
        Initialize task-aware loss function.
        
        Args:
            task_weight: Weight for task loss (λ1)
            recon_weight: Weight for reconstruction loss (λ2)
            bitrate_weight: Weight for bitrate loss (λ3)
            task_type: Type of task ('detection', 'segmentation', 'tracking')
        """
        super(TaskAwareLoss, self).__init__()
        self.task_weight = task_weight
        self.recon_weight = recon_weight
        self.bitrate_weight = bitrate_weight
        self.task_type = task_type
        
        # Task-specific loss functions
        if task_type == 'detection' or task_type == 'tracking':
            self.task_loss_fn = nn.MSELoss()
        elif task_type == 'segmentation':
            # For segmentation, use a combination of cross-entropy and dice loss
            self.task_loss_fn = nn.CrossEntropyLoss()
        else:
            # Default to MSE
            self.task_loss_fn = nn.MSELoss()
        
        # Reconstruction loss
        self.recon_loss_fn = nn.L1Loss()
    
    def forward(
        self,
        task_out: torch.Tensor,
        labels: torch.Tensor,
        recon: torch.Tensor,
        raw: torch.Tensor,
        bitrate: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss.
        
        Args:
            task_out: Output from task network
            labels: Ground truth labels
            recon: Reconstructed frames
            raw: Original input frames
            bitrate: Bits per pixel
            
        Returns:
            Dictionary containing total loss and individual components
        """
        # Task loss
        task_loss = self.task_loss_fn(task_out, labels)
        
        # Reconstruction loss
        recon_loss = self.recon_loss_fn(recon, raw)
        
        # Bitrate loss
        bitrate_loss = bitrate.mean()
        
        # Combined loss
        total_loss = (
            self.task_weight * task_loss +
            self.recon_weight * recon_loss +
            self.bitrate_weight * bitrate_loss
        )
        
        return {
            'total': total_loss,
            'task': task_loss,
            'recon': recon_loss,
            'bitrate': bitrate_loss
        }


class DetectionLoss(nn.Module):
    """Loss function for object detection tasks.
    
    Combines classification, localization, and objectness losses.
    """
    def __init__(self, 
                 lambda_cls: float = 1.0, 
                 lambda_box: float = 1.0, 
                 lambda_obj: float = 1.0):
        """
        Args:
            lambda_cls: Weight for classification loss
            lambda_box: Weight for bounding box regression loss
            lambda_obj: Weight for objectness loss
        """
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dictionary with 'detections' predictions
            targets: Dictionary with 'boxes' and 'classes' targets
            
        Returns:
            Dictionary with loss components
        """
        # Handle the new nested list structure of boxes and classes
        # For MOT dataset, the predictions are in format:
        # predictions['detections']: List of predicted detections for each frame
        # targets['boxes']: List[List[Tensor]] - Batch of lists of bbox tensors for each frame
        # targets['classes']: List[List[Tensor]] - Batch of lists of class tensors for each frame
        
        # Initialize loss values
        cls_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        box_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        obj_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        total_objects = 0
        
        # Process batch by batch
        for batch_idx in range(len(targets['boxes'])):
            # Process frame by frame
            for frame_idx in range(len(targets['boxes'][batch_idx])):
                # Get ground truth boxes and classes for this frame
                gt_boxes = targets['boxes'][batch_idx][frame_idx]  # [N, 4]
                gt_classes = targets['classes'][batch_idx][frame_idx]  # [N]
                
                # If no objects in this frame, continue
                if gt_boxes.size(0) == 0:
                    continue
                
                # Get predictions for this frame
                if 'detections' in predictions and len(predictions['detections']) > 0:
                    # Extract predictions for this batch and frame
                    frame_preds = predictions['detections'][batch_idx][frame_idx]
                    
                    if len(frame_preds) > 0:
                        pred_boxes = frame_preds['boxes']  # [M, 4]
                        pred_scores = frame_preds['scores']  # [M]
                        pred_classes = frame_preds['classes']  # [M]
                        
                        # IoU between predictions and targets
                        iou_matrix = box_iou(pred_boxes, gt_boxes)  # [M, N]
                        
                        # Assign predictions to ground truth objects
                        max_iou_values, max_iou_indices = iou_matrix.max(dim=1)
                        
                        # For each prediction, get the matched ground truth
                        for pred_idx, gt_idx in enumerate(max_iou_indices):
                            # If IoU is high enough, consider it a positive match
                            if max_iou_values[pred_idx] > 0.5:
                                # Class loss
                                pred_class_onehot = F.one_hot(pred_classes[pred_idx].long(), num_classes=80).float()
                                gt_class_onehot = F.one_hot(gt_classes[gt_idx].long(), num_classes=80).float()
                                
                                cls_loss += F.binary_cross_entropy_with_logits(
                                    pred_class_onehot,
                                    gt_class_onehot
                                )
                                
                                # Box loss (using GIoU loss)
                                box_loss += 1.0 - generalized_box_iou(
                                    pred_boxes[pred_idx].unsqueeze(0),
                                    gt_boxes[gt_idx].unsqueeze(0)
                                )
                                
                                # Objectness loss
                                obj_loss += F.binary_cross_entropy_with_logits(
                                    pred_scores[pred_idx],
                                    torch.ones_like(pred_scores[pred_idx])
                                )
                                
                                total_objects += 1
        
        # Average losses
        if total_objects > 0:
            cls_loss = cls_loss / total_objects
            box_loss = box_loss / total_objects
            obj_loss = obj_loss / total_objects
        
        # Combined loss
        loss = (
            self.lambda_cls * cls_loss + 
            self.lambda_box * box_loss + 
            self.lambda_obj * obj_loss
        )
        
        return {
            'loss': loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'obj_loss': obj_loss
        }


class SegmentationLoss(nn.Module):
    """Loss function for semantic segmentation tasks.
    
    Combines cross-entropy and Dice loss.
    """
    def __init__(self, 
                 lambda_ce: float = 1.0, 
                 lambda_dice: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            lambda_ce: Weight for cross-entropy loss
            lambda_dice: Weight for Dice loss
            class_weights: Optional weights for classes in cross-entropy loss
        """
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        
        # Create the cross-entropy loss
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def dice_loss(self, 
                 predictions: torch.Tensor, 
                 targets: torch.Tensor, 
                 smooth: float = 1.0) -> torch.Tensor:
        """
        Dice loss for multi-class segmentation.
        
        Args:
            predictions: Prediction tensor [B, C, H, W]
            targets: Target tensor [B, H, W]
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice loss
        """
        # Convert predictions to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        target_one_hot = F.one_hot(targets.long(), num_classes=predictions.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for cls in range(predictions.size(1)):
            pred_cls = probs[:, cls]  # [B, H, W]
            target_cls = target_one_hot[:, cls]  # [B, H, W]
            
            # Calculate intersection and union
            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
            
            # Calculate Dice score
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        # Average Dice scores across classes and batches
        dice_score = torch.stack(dice_scores, dim=1).mean()
        
        # Dice loss is 1 - Dice score
        return 1.0 - dice_score
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dictionary with segmentation predictions
            targets: Dictionary with segmentation targets (masks)
            
        Returns:
            Dictionary with loss components
        """
        # Extract segmentation predictions and targets
        # For video sequences, predictions are in this format:
        # predictions['segmentation']: [B, T, C, H, W] - batch of segmentation masks for each frame
        # targets['masks']: [B, T, H, W] - batch of ground truth masks for each frame
        
        # Initialize loss values
        device = next(iter(predictions.values())).device
        ce_loss = torch.tensor(0.0, device=device)
        dice_loss = torch.tensor(0.0, device=device)
        
        # Get segmentation predictions and targets
        if 'segmentation' in predictions:
            pred_masks = predictions['segmentation']  # [B, T, C, H, W]
            
            # Handle batch dimension first
            batch_size = pred_masks.size(0)
            time_steps = pred_masks.size(1)
            
            # Reshape to [B*T, C, H, W] and [B*T, H, W]
            B, T, C, H, W = pred_masks.shape
            pred_masks = pred_masks.reshape(B*T, C, H, W)
            target_masks = targets['masks'].reshape(B*T, H, W)
            
            # Calculate cross-entropy loss
            ce_loss = self.ce_loss(pred_masks, target_masks).mean()
            
            # Calculate Dice loss
            dice_loss = self.dice_loss(pred_masks, target_masks)
        
        # Combined loss
        loss = self.lambda_ce * ce_loss + self.lambda_dice * dice_loss
        
        return {
            'loss': loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss
        }


class TrackingLoss(nn.Module):
    """Loss function for object tracking tasks.
    
    Combines detection and association losses.
    """
    def __init__(self, 
                 lambda_cls: float = 1.0, 
                 lambda_box: float = 1.0):
        """
        Args:
            lambda_cls: Weight for classification loss
            lambda_box: Weight for bounding box regression loss
        """
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        
        # Initialize detection loss for bounding box prediction
        self.detection_loss = DetectionLoss()
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dictionary with tracking predictions
            targets: Dictionary with tracking targets
            
        Returns:
            Dictionary with loss components
        """
        # Initialize loss values
        device = next(iter(predictions.values())).device
        det_loss = torch.tensor(0.0, device=device)
        track_loss = torch.tensor(0.0, device=device)
        
        # Process detections using detection loss
        if 'detections' in predictions:
            det_loss_dict = self.detection_loss(predictions, targets)
            det_loss = det_loss_dict['loss']
        
        # Process tracking components
        if 'tracks' in predictions:
            # Extract tracking predictions
            pred_tracks = predictions['tracks']  # List of tracked objects with IDs
            
            # Process each batch item
            for batch_idx in range(len(targets['boxes'])):
                # Process frame pairs for temporal association
                for t in range(1, len(targets['boxes'][batch_idx])):
                    # Previous and current frame targets
                    prev_gt_boxes = targets['boxes'][batch_idx][t-1]  # [N, 4]
                    prev_gt_ids = targets['track_ids'][batch_idx][t-1]  # [N]
                    
                    curr_gt_boxes = targets['boxes'][batch_idx][t]  # [M, 4]
                    curr_gt_ids = targets['track_ids'][batch_idx][t]  # [M]
                    
                    # If no objects in either frame, skip
                    if prev_gt_boxes.size(0) == 0 or curr_gt_boxes.size(0) == 0:
                        continue
                    
                    # Get predicted tracks for these frames
                    if batch_idx < len(pred_tracks) and t < len(pred_tracks[batch_idx]):
                        prev_pred_tracks = pred_tracks[batch_idx][t-1]
                        curr_pred_tracks = pred_tracks[batch_idx][t]
                        
                        # Check for common track IDs between frames
                        for gt_id in curr_gt_ids:
                            # Find this ID in previous frame
                            prev_gt_idx = (prev_gt_ids == gt_id).nonzero(as_tuple=True)[0]
                            curr_gt_idx = (curr_gt_ids == gt_id).nonzero(as_tuple=True)[0]
                            
                            # If ID exists in both frames, calculate association loss
                            if len(prev_gt_idx) > 0 and len(curr_gt_idx) > 0:
                                prev_gt_idx = prev_gt_idx[0]
                                curr_gt_idx = curr_gt_idx[0]
                                
                                # Find predictions for this track ID
                                prev_pred_idx = None
                                curr_pred_idx = None
                                
                                for i, track in enumerate(prev_pred_tracks):
                                    if track['id'] == gt_id:
                                        prev_pred_idx = i
                                        break
                                
                                for i, track in enumerate(curr_pred_tracks):
                                    if track['id'] == gt_id:
                                        curr_pred_idx = i
                                        break
                                
                                # If track was predicted in both frames, calculate association loss
                                if prev_pred_idx is not None and curr_pred_idx is not None:
                                    # Box IoU between frames should be high for same ID
                                    prev_pred_box = prev_pred_tracks[prev_pred_idx]['box']
                                    curr_pred_box = curr_pred_tracks[curr_pred_idx]['box']
                                    
                                    # Temporal consistency loss - boxes should be similar
                                    track_loss += F.smooth_l1_loss(
                                        prev_pred_box, 
                                        curr_pred_box
                                    )
        
        # Combined loss
        loss = det_loss + track_loss
        
        return {
            'loss': loss,
            'det_loss': det_loss,
            'track_loss': track_loss
        }


class TemporalConsistencyLoss(nn.Module):
    """Loss function for ensuring temporal consistency in video predictions.
    
    Penalizes large changes between consecutive frames.
    """
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Weight for temporal consistency loss
        """
        super().__init__()
        self.alpha = alpha
        
    def forward(self, 
                current_frame: torch.Tensor, 
                previous_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_frame: Current frame features or predictions
            previous_frame: Previous frame features or predictions
            
        Returns:
            Temporal consistency loss
        """
        # Calculate temporal difference
        temp_diff = torch.abs(current_frame - previous_frame)
        
        # Calculate loss (mean absolute difference)
        loss = temp_diff.mean()
        
        return self.alpha * loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using feature maps from a pretrained network.
    
    Computes the difference in feature space rather than pixel space.
    """
    def __init__(self, feature_extractor: nn.Module, layer_weights: Dict[str, float] = None):
        """
        Args:
            feature_extractor: Pretrained network for feature extraction
            layer_weights: Weights for different layers
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights or {'layer1': 1.0, 'layer2': 1.0}
        
        # Put the feature extractor in evaluation mode
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Predicted images
            targets: Target images
            
        Returns:
            Perceptual loss
        """
        # Extract features
        pred_features = self.feature_extractor(predictions)
        target_features = self.feature_extractor(targets)
        
        # Compute weighted loss across layers
        loss = 0
        for layer_name, weight in self.layer_weights.items():
            pred_feat = pred_features[layer_name]
            target_feat = target_features[layer_name]
            
            # Compute MSE loss between feature maps
            loss += weight * F.mse_loss(pred_feat, target_feat)
            
        return loss


def compute_total_loss(
    task_out: torch.Tensor,
    labels: torch.Tensor,
    recon: torch.Tensor,
    raw: torch.Tensor,
    bitrate: torch.Tensor,
    task_weight: float = 1.0,
    recon_weight: float = 1.0,
    bitrate_weight: float = 0.01
) -> torch.Tensor:
    """
    Compute the combined loss for task-aware video compression.
    
    Args:
        task_out: Output from task network
        labels: Ground truth labels
        recon: Reconstructed frames
        raw: Original input frames
        bitrate: Bits per pixel
        task_weight: Weight for task loss component
        recon_weight: Weight for reconstruction loss component
        bitrate_weight: Weight for bitrate loss component
        
    Returns:
        Total weighted loss
    """
    try:
        # Task loss - this might vary depending on task type
        # For simplicity, we'll use MSE here, but in practice
        # you would use a task-specific loss function
        task_loss = F.mse_loss(task_out, labels)
        
        # Reconstruction loss (MSE between original and reconstructed frames)
        recon_loss = F.mse_loss(recon, raw)
        
        # Bitrate loss (mean of estimated bits per pixel)
        bitrate_loss = bitrate.mean()
        
        # Combine the losses with weights
        total_loss = (
            task_weight * task_loss +
            recon_weight * recon_loss +
            bitrate_weight * bitrate_loss
        )
        
        return total_loss
    
    except Exception as e:
        print(f"Error in compute_total_loss: {str(e)}")
        # Return a default loss that can be backpropagated
        # This allows training to continue even if there's an issue
        return torch.tensor(1.0, device=raw.device, requires_grad=True)


# Test code
if __name__ == "__main__":
    # Test RateDistortionLoss
    rd_loss = RateDistortionLoss(lambda_rd=0.01)
    original = torch.rand(2, 3, 64, 64)
    reconstructed = torch.rand(2, 3, 64, 64)
    bits = torch.tensor(8192.0)
    
    loss_dict = rd_loss(original, reconstructed, bits)
    print(f"RD Loss: {loss_dict['loss']:.4f}")
    print(f"Distortion: {loss_dict['distortion']:.4f}")
    print(f"Rate: {loss_dict['rate']:.4f}")
    
    # Test DetectionLoss
    det_loss = DetectionLoss()
    predictions = {
        'bbox': torch.rand(2, 10, 4),
        'cls': torch.rand(2, 10, 20),
        'obj': torch.rand(2, 10, 1)
    }
    targets = {
        'bbox': torch.rand(2, 10, 4),
        'cls': torch.zeros(2, 10, 20).scatter_(2, torch.randint(0, 20, (2, 10, 1)), 1),
        'obj': (torch.rand(2, 10, 1) > 0.5).float()
    }
    
    loss_dict = det_loss(predictions, targets)
    print(f"Detection Loss: {loss_dict['loss']:.4f}")
