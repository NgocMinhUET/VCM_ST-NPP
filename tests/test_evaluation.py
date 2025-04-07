import os
import sys
import unittest
import numpy as np
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import calculate_map, calculate_miou, calculate_iou


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for evaluation metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create some dummy detection data
        self.detections = [
            {
                'boxes': [
                    [10, 10, 20, 20],  # x1, y1, x2, y2
                    [30, 30, 40, 40]
                ],
                'scores': [0.9, 0.8],
                'classes': [1, 2]
            },
            {
                'boxes': [
                    [50, 50, 60, 60],
                    [70, 70, 80, 80]
                ],
                'scores': [0.7, 0.6],
                'classes': [1, 3]
            }
        ]
        
        # Create matching ground truth
        self.ground_truth = [
            {
                'boxes': [
                    [11, 11, 21, 21],  # Close to first detection
                    [31, 31, 41, 41]   # Close to second detection
                ],
                'classes': [1, 2]
            },
            {
                'boxes': [
                    [51, 51, 61, 61],  # Close to third detection
                    [90, 90, 100, 100] # No matching detection
                ],
                'classes': [1, 3]
            }
        ]
        
        # Create simple segmentation masks
        # 2x2 images with 2 classes (0 and 1)
        self.segmentation_pred = [
            np.array([[0, 0], [1, 1]]),
            np.array([[1, 0], [1, 0]])
        ]
        
        self.segmentation_gt = [
            np.array([[0, 0], [1, 1]]),  # Perfect match
            np.array([[1, 1], [1, 0]])   # Partial match
        ]
    
    def test_iou_calculation(self):
        """Test IoU calculation between bounding boxes"""
        # Perfect overlap should give IoU of 1.0
        box1 = [10, 10, 20, 20]
        box2 = [10, 10, 20, 20]
        self.assertEqual(calculate_iou(box1, box2), 1.0)
        
        # No overlap should give IoU of 0.0
        box1 = [10, 10, 20, 20]
        box2 = [30, 30, 40, 40]
        self.assertEqual(calculate_iou(box1, box2), 0.0)
        
        # Partial overlap
        box1 = [10, 10, 30, 30]
        box2 = [20, 20, 40, 40]
        expected_iou = 100.0 / 500.0  # Intersection: 10x10, Union: (20x20) + (20x20) - (10x10)
        self.assertAlmostEqual(calculate_iou(box1, box2), expected_iou, places=5)
    
    def test_map_calculation(self):
        """Test mAP calculation for detection evaluation"""
        # Test with the detection and ground truth data
        map_value, ap_per_class = calculate_map(self.detections, self.ground_truth)
        
        # Check that mAP is between 0 and 1
        self.assertGreaterEqual(map_value, 0.0)
        self.assertLessEqual(map_value, 1.0)
        
        # Check that per-class AP is returned
        self.assertIsInstance(ap_per_class, dict)
        
        # Since our test data has matching detections for most ground truth,
        # mAP should be relatively high
        self.assertGreater(map_value, 0.5)
        
        # Test with empty inputs
        map_value, ap_per_class = calculate_map([], [])
        self.assertEqual(map_value, 0.0)
        self.assertEqual(ap_per_class, {})
        
        # Test with mismatched inputs
        with self.assertWarns(Warning):
            map_value, _ = calculate_map(self.detections[:1], self.ground_truth)
    
    def test_miou_calculation(self):
        """Test mIoU calculation for segmentation evaluation"""
        # Test with the segmentation prediction and ground truth data
        miou_value, class_ious = calculate_miou(self.segmentation_pred, self.segmentation_gt, num_classes=2)
        
        # Check that mIoU is between 0 and 1
        self.assertGreaterEqual(miou_value, 0.0)
        self.assertLessEqual(miou_value, 1.0)
        
        # Check that per-class IoU is returned
        self.assertIsInstance(class_ious, dict)
        
        # Class 0 should have high IoU due to good overlap
        self.assertGreater(class_ious.get(0, 0), 0.5)
        
        # Check handling of empty inputs
        miou_value, class_ious = calculate_miou([], [])
        self.assertEqual(miou_value, 0.0)
        self.assertEqual(class_ious, {})
        
        # Check handling of mismatched shapes
        mismatched_pred = [np.array([[0, 0], [1, 1]]), np.array([[1, 0, 0], [1, 0, 0]])]
        with self.assertLogs(level='WARNING'):
            miou_value, _ = calculate_miou(mismatched_pred, self.segmentation_gt, num_classes=2)
    
    def test_miou_edge_cases(self):
        """Test mIoU calculation with edge cases"""
        # All predictions correct
        perfect_pred = [np.copy(gt) for gt in self.segmentation_gt]
        miou_value, class_ious = calculate_miou(perfect_pred, self.segmentation_gt, num_classes=2)
        self.assertAlmostEqual(miou_value, 1.0, places=5)
        
        # All predictions wrong (all zeros vs. all ones)
        all_zeros = [np.zeros_like(gt) for gt in self.segmentation_gt]
        all_ones = [np.ones_like(gt) for gt in self.segmentation_gt]
        miou_value, _ = calculate_miou(all_zeros, all_ones, num_classes=2)
        self.assertAlmostEqual(miou_value, 0.0, places=5)
        
        # Test with classes outside valid range
        invalid_pred = [np.array([[0, 5], [1, 1]]), np.array([[1, 0], [1, 0]])]
        with self.assertLogs(level='WARNING'):
            miou_value, _ = calculate_miou(invalid_pred, self.segmentation_gt, num_classes=2)


if __name__ == '__main__':
    unittest.main()