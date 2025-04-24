import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from collections import defaultdict

class CompressionMetrics:
    """
    Class for computing video compression metrics
    """
    @staticmethod
    def psnr(original, reconstructed):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            original (torch.Tensor): Original video tensor [B, C, T, H, W]
            reconstructed (torch.Tensor): Reconstructed video tensor [B, C, T, H, W]
            
        Returns:
            float: PSNR value in dB
        """
        if isinstance(original, torch.Tensor):
            original = original.clamp(0, 1).detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.clamp(0, 1).detach().cpu().numpy()
            
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
        return psnr
    
    @staticmethod
    def msssim(original, reconstructed):
        """
        Calculate Multi-Scale Structural Similarity Index (MS-SSIM)
        
        Args:
            original (torch.Tensor): Original video tensor [B, C, T, H, W]
            reconstructed (torch.Tensor): Reconstructed video tensor [B, C, T, H, W]
            
        Returns:
            float: MS-SSIM value
        """
        # This is a placeholder for the actual implementation
        # In reality, you would use a proper MS-SSIM implementation
        # For example, from pytorch-msssim or other libraries
        # Here we use a simplified version based on structural similarity
        
        if isinstance(original, np.ndarray):
            original = torch.from_numpy(original)
        if isinstance(reconstructed, np.ndarray):
            reconstructed = torch.from_numpy(reconstructed)
            
        original = original.clamp(0, 1)
        reconstructed = reconstructed.clamp(0, 1)
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Calculate mean
        mu_x = F.avg_pool3d(original, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool3d(reconstructed, kernel_size=3, stride=1, padding=1)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Calculate variance and covariance
        sigma_x_sq = F.avg_pool3d(original ** 2, kernel_size=3, stride=1, padding=1) - mu_x_sq
        sigma_y_sq = F.avg_pool3d(reconstructed ** 2, kernel_size=3, stride=1, padding=1) - mu_y_sq
        sigma_xy = F.avg_pool3d(original * reconstructed, kernel_size=3, stride=1, padding=1) - mu_xy
        
        # SSIM formula
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        
        # Calculate mean SSIM
        ssim_value = torch.mean(ssim_map).item()
        
        return ssim_value
    
    @staticmethod
    def bpp(bits, height, width, frames):
        """
        Calculate bits per pixel (bpp)
        
        Args:
            bits (int): Number of bits used
            height (int): Height of the video
            width (int): Width of the video
            frames (int): Number of frames
            
        Returns:
            float: Bits per pixel
        """
        pixels = height * width * frames
        return bits / pixels


class DetectionMetrics:
    """
    Class for computing object detection metrics
    """
    @staticmethod
    def _box_iou(box1, box2):
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1 (torch.Tensor): Bounding box format [x1, y1, x2, y2]
            box2 (torch.Tensor): Bounding box format [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def mean_average_precision(pred_boxes, true_boxes, pred_scores, pred_labels, true_labels, iou_threshold=0.5):
        """
        Calculate mean Average Precision (mAP)
        
        Args:
            pred_boxes (list): Predicted bounding boxes, format [[x1, y1, x2, y2], ...]
            true_boxes (list): Ground truth bounding boxes, format [[x1, y1, x2, y2], ...]
            pred_scores (list): Confidence scores for predicted boxes
            pred_labels (list): Class labels for predicted boxes
            true_labels (list): Class labels for ground truth boxes
            iou_threshold (float): IoU threshold for a true positive
            
        Returns:
            float: mAP value
        """
        if not pred_boxes or not true_boxes:
            return 0.0
            
        # Convert to numpy arrays for easier manipulation
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.detach().cpu().numpy()
        if isinstance(true_boxes, torch.Tensor):
            true_boxes = true_boxes.detach().cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.detach().cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.detach().cpu().numpy()
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.detach().cpu().numpy()
            
        # Get unique class labels
        unique_classes = np.unique(np.concatenate((pred_labels, true_labels)))
        
        # Calculate AP for each class
        average_precisions = []
        
        for cls in unique_classes:
            # Get indices for the current class
            pred_cls_indices = np.where(pred_labels == cls)[0]
            true_cls_indices = np.where(true_labels == cls)[0]
            
            if len(pred_cls_indices) == 0 or len(true_cls_indices) == 0:
                continue
                
            # Get boxes and scores for current class
            cls_pred_boxes = pred_boxes[pred_cls_indices]
            cls_pred_scores = pred_scores[pred_cls_indices]
            cls_true_boxes = true_boxes[true_cls_indices]
            
            # Sort by confidence scores
            sorted_indices = np.argsort(-cls_pred_scores)
            cls_pred_boxes = cls_pred_boxes[sorted_indices]
            cls_pred_scores = cls_pred_scores[sorted_indices]
            
            # Calculate true positives and false positives
            tp = np.zeros(len(cls_pred_boxes))
            fp = np.zeros(len(cls_pred_boxes))
            
            # Create array to keep track of ground truth boxes that have been detected
            detected_gt = np.zeros(len(cls_true_boxes))
            
            for i, pred_box in enumerate(cls_pred_boxes):
                best_iou = 0.0
                best_gt_idx = -1
                
                # Find ground truth box with highest IoU
                for j, gt_box in enumerate(cls_true_boxes):
                    iou = DetectionMetrics._box_iou(pred_box, gt_box)
                    if iou > best_iou and not detected_gt[j]:
                        best_iou = iou
                        best_gt_idx = j
                
                # Check if the prediction is a true positive
                if best_iou >= iou_threshold and best_gt_idx != -1:
                    tp[i] = 1
                    detected_gt[best_gt_idx] = 1
                else:
                    fp[i] = 1
            
            # Calculate cumulative sum for precision and recall calculation
            cumsum_tp = np.cumsum(tp)
            cumsum_fp = np.cumsum(fp)
            
            # Calculate precision and recall
            precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
            recall = cumsum_tp / len(cls_true_boxes)
            
            # Calculate Average Precision using the 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11
            
            average_precisions.append(ap)
        
        # Calculate mAP
        return np.mean(average_precisions) if average_precisions else 0.0


class SegmentationMetrics:
    """
    Class for computing semantic segmentation metrics
    """
    @staticmethod
    def pixel_accuracy(pred_mask, true_mask):
        """
        Calculate pixel accuracy
        
        Args:
            pred_mask (torch.Tensor): Predicted segmentation mask [B, C, H, W]
            true_mask (torch.Tensor): Ground truth segmentation mask [B, C, H, W]
            
        Returns:
            float: Pixel accuracy
        """
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()
        if isinstance(true_mask, torch.Tensor):
            true_mask = true_mask.detach().cpu().numpy()
            
        # Convert to binary mask if necessary
        if pred_mask.shape[1] > 1:  # Multi-class segmentation
            pred_mask = np.argmax(pred_mask, axis=1)
        else:
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
        if true_mask.shape[1] > 1:
            true_mask = np.argmax(true_mask, axis=1)
        else:
            true_mask = (true_mask > 0.5).astype(np.uint8)
            
        # Calculate pixel accuracy
        correct = (pred_mask == true_mask).sum()
        total = true_mask.size
        
        return correct / total
    
    @staticmethod
    def mean_iou(pred_mask, true_mask, num_classes):
        """
        Calculate mean Intersection over Union (mIoU)
        
        Args:
            pred_mask (torch.Tensor): Predicted segmentation mask [B, C, H, W]
            true_mask (torch.Tensor): Ground truth segmentation mask [B, C, H, W]
            num_classes (int): Number of classes
            
        Returns:
            float: Mean IoU
        """
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()
        if isinstance(true_mask, torch.Tensor):
            true_mask = true_mask.detach().cpu().numpy()
            
        # Convert to class indices
        if pred_mask.shape[1] > 1:  # Multi-class segmentation
            pred_mask = np.argmax(pred_mask, axis=1)
        else:
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
        if true_mask.shape[1] > 1:
            true_mask = np.argmax(true_mask, axis=1)
        else:
            true_mask = (true_mask > 0.5).astype(np.uint8)
            
        # Calculate IoU for each class
        ious = []
        
        for cls in range(num_classes):
            pred_cls = (pred_mask == cls).astype(np.uint8)
            true_cls = (true_mask == cls).astype(np.uint8)
            
            intersection = (pred_cls & true_cls).sum()
            union = (pred_cls | true_cls).sum()
            
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)
            
        return np.mean(ious)
    
    @staticmethod
    def dice_coefficient(pred_mask, true_mask):
        """
        Calculate Dice coefficient (F1 score)
        
        Args:
            pred_mask (torch.Tensor): Predicted segmentation mask [B, C, H, W]
            true_mask (torch.Tensor): Ground truth segmentation mask [B, C, H, W]
            
        Returns:
            float: Dice coefficient
        """
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()
        if isinstance(true_mask, torch.Tensor):
            true_mask = true_mask.detach().cpu().numpy()
            
        # Convert to binary mask if necessary
        if pred_mask.shape[1] > 1:  # Multi-class segmentation
            pred_mask = np.argmax(pred_mask, axis=1)
        else:
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
        if true_mask.shape[1] > 1:
            true_mask = np.argmax(true_mask, axis=1)
        else:
            true_mask = (true_mask > 0.5).astype(np.uint8)
            
        # Calculate Dice coefficient
        intersection = (pred_mask & true_mask).sum()
        dice = (2. * intersection) / (pred_mask.sum() + true_mask.sum())
        
        return dice


class TrackingMetrics:
    """
    Class for computing multi-object tracking metrics
    """
    @staticmethod
    def calculate_matches(pred_tracks, gt_tracks, iou_threshold=0.5):
        """
        Calculate matches between predicted tracks and ground truth tracks
        
        Args:
            pred_tracks (dict): Dictionary mapping track IDs to bounding boxes
            gt_tracks (dict): Dictionary mapping track IDs to bounding boxes
            iou_threshold (float): IoU threshold for matching
            
        Returns:
            tuple: Matched, missed, and false positive counts
        """
        matched = 0
        missed = 0
        false_positives = 0
        
        for frame_id in gt_tracks:
            if frame_id not in pred_tracks:
                missed += len(gt_tracks[frame_id])
                continue
                
            gt_boxes = gt_tracks[frame_id]
            pred_boxes = pred_tracks[frame_id]
            
            # Create matrix of IoUs between all pairs of boxes
            ious = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    ious[i, j] = DetectionMetrics._box_iou(gt_box, pred_box)
            
            # Find matches using greedy matching
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
            pred_matched = np.zeros(len(pred_boxes), dtype=bool)
            
            # Sort IoUs in descending order
            indices = np.argsort(-ious.flatten())
            for idx in indices:
                i, j = np.unravel_index(idx, ious.shape)
                
                if ious[i, j] < iou_threshold:
                    break
                    
                if not gt_matched[i] and not pred_matched[j]:
                    gt_matched[i] = True
                    pred_matched[j] = True
                    matched += 1
            
            # Count unmatched ground truth boxes and predicted boxes
            missed += np.sum(~gt_matched)
            false_positives += np.sum(~pred_matched)
        
        # Add false positives for frames that don't exist in ground truth
        for frame_id in pred_tracks:
            if frame_id not in gt_tracks:
                false_positives += len(pred_tracks[frame_id])
                
        return matched, missed, false_positives
    
    @staticmethod
    def mota(pred_tracks, gt_tracks, iou_threshold=0.5):
        """
        Calculate Multiple Object Tracking Accuracy (MOTA)
        
        Args:
            pred_tracks (dict): Dictionary mapping frame IDs to lists of bounding boxes
            gt_tracks (dict): Dictionary mapping frame IDs to lists of bounding boxes
            iou_threshold (float): IoU threshold for matching
            
        Returns:
            float: MOTA value
        """
        matched, missed, false_positives = TrackingMetrics.calculate_matches(
            pred_tracks, gt_tracks, iou_threshold)
        
        # Calculate total number of ground truth objects
        gt_total = sum(len(boxes) for boxes in gt_tracks.values())
        
        if gt_total == 0:
            return 0.0
            
        # Calculate MOTA
        mota = 1.0 - (missed + false_positives) / gt_total
        
        return max(0.0, mota)  # Clip to [0, 1] range
    
    @staticmethod
    def motp(pred_tracks, gt_tracks, iou_threshold=0.5):
        """
        Calculate Multiple Object Tracking Precision (MOTP)
        
        Args:
            pred_tracks (dict): Dictionary mapping frame IDs to lists of bounding boxes
            gt_tracks (dict): Dictionary mapping frame IDs to lists of bounding boxes
            iou_threshold (float): IoU threshold for matching
            
        Returns:
            float: MOTP value
        """
        total_iou = 0.0
        matches = 0
        
        for frame_id in gt_tracks:
            if frame_id not in pred_tracks:
                continue
                
            gt_boxes = gt_tracks[frame_id]
            pred_boxes = pred_tracks[frame_id]
            
            # Create matrix of IoUs between all pairs of boxes
            ious = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    ious[i, j] = DetectionMetrics._box_iou(gt_box, pred_box)
            
            # Find matches using greedy matching
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
            pred_matched = np.zeros(len(pred_boxes), dtype=bool)
            
            # Sort IoUs in descending order
            indices = np.argsort(-ious.flatten())
            for idx in indices:
                i, j = np.unravel_index(idx, ious.shape)
                
                if ious[i, j] < iou_threshold:
                    break
                    
                if not gt_matched[i] and not pred_matched[j]:
                    gt_matched[i] = True
                    pred_matched[j] = True
                    matches += 1
                    total_iou += ious[i, j]
        
        if matches == 0:
            return 0.0
            
        # Calculate MOTP
        motp = total_iou / matches
        
        return motp


# Function to evaluate the combined task-aware compression model
def evaluate_model(model, test_dataloader, task_types=['detection', 'segmentation', 'tracking']):
    """
    Evaluate a task-aware compression model
    
    Args:
        model: The task-aware compression model
        test_dataloader: DataLoader for test data
        task_types: List of task types to evaluate
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    results = {
        'compression': {},
        'detection': {},
        'segmentation': {},
        'tracking': {}
    }
    
    # Example evaluation code - would need to be adapted to actual model outputs
    total_bits = 0
    total_pixels = 0
    psnr_values = []
    msssim_values = []
    
    # Detection metrics
    all_pred_boxes = []
    all_true_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_true_labels = []
    
    # Segmentation metrics
    seg_pixel_acc = []
    seg_mean_iou = []
    seg_dice = []
    
    # Tracking metrics
    pred_tracks = defaultdict(list)
    gt_tracks = defaultdict(list)
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # Extract data from batch
            videos = batch['video'].to(device)
            
            # Get model outputs
            outputs = model(videos)
            
            # Extract relevant outputs based on model architecture
            reconstructed_videos = outputs['reconstructed']
            bits_used = outputs.get('bits_used', 0)
            
            # Compression metrics
            b, c, t, h, w = videos.shape
            total_bits += bits_used
            total_pixels += b * t * h * w
            psnr_values.append(CompressionMetrics.psnr(videos, reconstructed_videos))
            msssim_values.append(CompressionMetrics.msssim(videos, reconstructed_videos))
            
            # Task-specific metrics
            if 'detection' in task_types and 'detection_output' in outputs:
                det_output = outputs['detection_output']
                
                # Extract predicted and ground truth boxes, scores, labels
                pred_boxes = det_output['boxes']
                true_boxes = batch['boxes']
                pred_scores = det_output['scores']
                pred_labels = det_output['labels']
                true_labels = batch['labels']
                
                # Append to lists for final evaluation
                all_pred_boxes.append(pred_boxes)
                all_true_boxes.append(true_boxes)
                all_pred_scores.append(pred_scores)
                all_pred_labels.append(pred_labels)
                all_true_labels.append(true_labels)
            
            if 'segmentation' in task_types and 'segmentation_output' in outputs:
                seg_output = outputs['segmentation_output']
                
                # Extract predicted and ground truth masks
                pred_masks = seg_output['masks']
                true_masks = batch['masks']
                
                # Calculate segmentation metrics
                num_classes = pred_masks.shape[1]
                seg_pixel_acc.append(SegmentationMetrics.pixel_accuracy(pred_masks, true_masks))
                seg_mean_iou.append(SegmentationMetrics.mean_iou(pred_masks, true_masks, num_classes))
                seg_dice.append(SegmentationMetrics.dice_coefficient(pred_masks, true_masks))
            
            if 'tracking' in task_types and 'tracking_output' in outputs:
                track_output = outputs['tracking_output']
                
                # Extract predicted and ground truth tracks
                for frame_idx in range(t):
                    frame_id = batch['frame_ids'][frame_idx].item()
                    
                    # Add predicted tracks
                    pred_tracks[frame_id].extend(track_output['tracks'][frame_idx])
                    
                    # Add ground truth tracks
                    gt_tracks[frame_id].extend(batch['tracks'][frame_idx])
    
    # Calculate final metrics
    
    # Compression metrics
    results['compression']['bpp'] = total_bits / total_pixels if total_pixels > 0 else 0
    results['compression']['psnr'] = np.mean(psnr_values) if psnr_values else 0
    results['compression']['msssim'] = np.mean(msssim_values) if msssim_values else 0
    
    # Detection metrics
    if all_pred_boxes and all_true_boxes:
        # Concatenate all batches
        all_pred_boxes = np.concatenate(all_pred_boxes)
        all_true_boxes = np.concatenate(all_true_boxes)
        all_pred_scores = np.concatenate(all_pred_scores)
        all_pred_labels = np.concatenate(all_pred_labels)
        all_true_labels = np.concatenate(all_true_labels)
        
        # Calculate mAP
        results['detection']['mAP'] = DetectionMetrics.mean_average_precision(
            all_pred_boxes, all_true_boxes, all_pred_scores, all_pred_labels, all_true_labels)
    
    # Segmentation metrics
    if seg_pixel_acc:
        results['segmentation']['pixel_accuracy'] = np.mean(seg_pixel_acc)
        results['segmentation']['mean_iou'] = np.mean(seg_mean_iou)
        results['segmentation']['dice'] = np.mean(seg_dice)
    
    # Tracking metrics
    if pred_tracks and gt_tracks:
        results['tracking']['MOTA'] = TrackingMetrics.mota(pred_tracks, gt_tracks)
        results['tracking']['MOTP'] = TrackingMetrics.motp(pred_tracks, gt_tracks)
    
    return results


# Test code
if __name__ == "__main__":
    # Test compression metrics
    original = torch.rand(2, 3, 5, 64, 64)
    reconstructed = original + 0.1 * torch.randn_like(original)
    
    psnr = CompressionMetrics.psnr(original, reconstructed)
    msssim = CompressionMetrics.msssim(original, reconstructed)
    bpp = CompressionMetrics.bpp(1000000, 64, 64, 5)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"MS-SSIM: {msssim:.4f}")
    print(f"BPP: {bpp:.4f}")
    
    # Test detection metrics
    pred_boxes = np.array([
        [10, 10, 50, 50],
        [20, 20, 60, 60],
        [70, 70, 120, 120]
    ])
    true_boxes = np.array([
        [12, 12, 52, 52],
        [70, 75, 125, 125]
    ])
    pred_scores = np.array([0.9, 0.8, 0.7])
    pred_labels = np.array([0, 0, 1])
    true_labels = np.array([0, 1])
    
    mAP = DetectionMetrics.mean_average_precision(
        pred_boxes, true_boxes, pred_scores, pred_labels, true_labels)
    print(f"mAP: {mAP:.4f}")
    
    # Test segmentation metrics
    batch_size = 2
    num_classes = 3
    height, width = 64, 64
    
    # Create random segmentation masks
    pred_mask = torch.zeros(batch_size, num_classes, height, width)
    true_mask = torch.zeros(batch_size, num_classes, height, width)
    
    # Set random classes for each pixel
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                pred_class = np.random.randint(0, num_classes)
                true_class = np.random.randint(0, num_classes)
                pred_mask[b, pred_class, h, w] = 1
                true_mask[b, true_class, h, w] = 1
    
    pixel_acc = SegmentationMetrics.pixel_accuracy(pred_mask, true_mask)
    mean_iou = SegmentationMetrics.mean_iou(pred_mask, true_mask, num_classes)
    dice = SegmentationMetrics.dice_coefficient(pred_mask, true_mask)
    
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}") 