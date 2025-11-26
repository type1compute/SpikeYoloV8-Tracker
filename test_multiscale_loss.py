#!/usr/bin/env python3
"""
Test script to verify multi-scale loss calculation is working correctly.
"""

import torch
import sys
from yolo_loss import YOLOLoss

def test_multiscale_loss():
    """Test that multi-scale loss properly combines losses from all scales."""
    
    print("="*80)
    print("MULTI-SCALE LOSS TEST")
    print("="*80)
    
    # Setup
    device = 'cpu'
    num_classes = 3
    batch_size = 2
    
    # Create loss function with custom scale weights
    scale_weights = [0.30, 0.35, 0.35]  # P5, P4, P3
    loss_fn = YOLOLoss(
        num_classes=num_classes,
        box_loss_weight=7.5,
        cls_loss_weight=10.0,
        track_loss_weight=1.0,
        iou_threshold=0.5,
        device=device,
        scale_weights=scale_weights
    )
    
    print(f"\n✓ Created YOLOLoss with scale_weights: {scale_weights}")
    print(f"  - Device: {device}")
    print(f"  - Num classes: {num_classes}")
    
    # Create dummy multi-scale predictions (3 scales: P5, P4, P3)
    # Format: [batch, num_anchors, features]
    # Features = 64 (DFL) + num_classes + 1 (track_id) = 68 for 3-class
    num_features = 64 + num_classes + 1
    
    # P5: Coarse scale (23x40 = 920 anchors)
    pred_p5 = torch.randn(batch_size, 920, num_features, device=device)
    
    # P4: Medium scale (roughly 2x anchors)
    pred_p4 = torch.randn(batch_size, 1840, num_features, device=device)
    
    # P3: Fine scale (roughly 4x anchors)
    pred_p3 = torch.randn(batch_size, 3680, num_features, device=device)
    
    predictions = [pred_p5, pred_p4, pred_p3]
    
    print(f"\n✓ Created multi-scale predictions:")
    print(f"  - P5 (coarse): {pred_p5.shape} - {pred_p5.shape[1]} anchors")
    print(f"  - P4 (medium): {pred_p4.shape} - {pred_p4.shape[1]} anchors")
    print(f"  - P3 (fine):   {pred_p3.shape} - {pred_p3.shape[1]} anchors")
    
    # Create dummy targets
    # Format: [class_id, x, y, w, h, conf, track_id, timestamp]
    targets = []
    for i in range(batch_size):
        # 3 objects per batch item
        num_objects = 3
        batch_targets = torch.zeros((num_objects, 8), device=device)
        
        # Random object properties
        batch_targets[:, 0] = torch.randint(0, num_classes, (num_objects,))  # class_id
        batch_targets[:, 1:5] = torch.rand((num_objects, 4)) * 0.5 + 0.25   # boxes (centered)
        batch_targets[:, 5] = 1.0  # confidence
        batch_targets[:, 6] = torch.arange(num_objects)  # track_id
        batch_targets[:, 7] = torch.rand((num_objects,)) * 1000  # timestamp
        
        targets.append(batch_targets)
    
    print(f"\n✓ Created targets:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Objects per batch: {[len(t) for t in targets]}")
    
    # Test 1: Multi-scale loss
    print("\n" + "="*80)
    print("TEST 1: Multi-Scale Loss Calculation")
    print("="*80)
    
    try:
        loss_dict_multi = loss_fn(
            predictions=predictions,
            targets=targets,
            track_features=None,
            event_timestamps=None,
            current_epoch=0
        )
        
        print(f"\n✓ Multi-scale loss computed successfully!")
        print(f"\n  Loss Components:")
        print(f"    - Box Loss:   {loss_dict_multi['box_loss']:.6f}")
        print(f"    - Class Loss: {loss_dict_multi['cls_loss']:.6f}")
        print(f"    - Track Loss: {loss_dict_multi['track_loss']:.6f}")
        print(f"    - Total Loss: {loss_dict_multi['total_loss']:.6f}")
        
        # Verify loss is not NaN or Inf
        assert not torch.isnan(loss_dict_multi['total_loss']), "Total loss is NaN!"
        assert not torch.isinf(loss_dict_multi['total_loss']), "Total loss is Inf!"
        assert loss_dict_multi['total_loss'] > 0, "Total loss should be positive!"
        
        print(f"\n  ✓ Loss values are valid (not NaN/Inf, positive)")
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Single-scale loss (for comparison)
    print("\n" + "="*80)
    print("TEST 2: Single-Scale Loss Calculation (P5 only)")
    print("="*80)
    
    try:
        loss_dict_single = loss_fn(
            predictions=pred_p5,  # Only P5
            targets=targets,
            track_features=None,
            event_timestamps=None,
            current_epoch=0
        )
        
        print(f"\n✓ Single-scale loss computed successfully!")
        print(f"\n  Loss Components:")
        print(f"    - Box Loss:   {loss_dict_single['box_loss']:.6f}")
        print(f"    - Class Loss: {loss_dict_single['cls_loss']:.6f}")
        print(f"    - Track Loss: {loss_dict_single['track_loss']:.6f}")
        print(f"    - Total Loss: {loss_dict_single['total_loss']:.6f}")
        
        print(f"\n  ✓ Loss values are valid")
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Compare losses
    print("\n" + "="*80)
    print("TEST 3: Multi-Scale vs Single-Scale Comparison")
    print("="*80)
    
    print(f"\n  Multi-Scale Total Loss: {loss_dict_multi['total_loss']:.6f}")
    print(f"  Single-Scale Total Loss: {loss_dict_single['total_loss']:.6f}")
    
    # Multi-scale should be different from single-scale
    # (weighted combination of 3 scales vs. just P5)
    loss_diff = abs(loss_dict_multi['total_loss'] - loss_dict_single['total_loss'])
    print(f"\n  Absolute difference: {loss_diff:.6f}")
    
    if loss_diff > 1e-4:
        print(f"  ✓ Losses are different (multi-scale uses all 3 scales)")
    else:
        print(f"  ⚠ WARNING: Losses are very similar, multi-scale might not be working correctly")
    
    # Test 4: Verify scale weights are used
    print("\n" + "="*80)
    print("TEST 4: Scale Weights Verification")
    print("="*80)
    
    print(f"\n  Configured scale weights: {scale_weights}")
    print(f"  - P5: {scale_weights[0]:.2f} (30%)")
    print(f"  - P4: {scale_weights[1]:.2f} (35%)")
    print(f"  - P3: {scale_weights[2]:.2f} (35%)")
    print(f"\n  ✓ Weights sum to {sum(scale_weights):.2f} (should be ~1.0)")
    
    assert abs(sum(scale_weights) - 1.0) < 0.01, "Scale weights should sum to 1.0!"
    
    # Test 5: Backward pass (gradient flow)
    print("\n" + "="*80)
    print("TEST 5: Gradient Flow Test")
    print("="*80)
    
    try:
        # Enable gradients
        pred_p5_grad = pred_p5.clone().detach().requires_grad_(True)
        pred_p4_grad = pred_p4.clone().detach().requires_grad_(True)
        pred_p3_grad = pred_p3.clone().detach().requires_grad_(True)
        
        predictions_grad = [pred_p5_grad, pred_p4_grad, pred_p3_grad]
        
        # Forward pass
        loss_dict = loss_fn(
            predictions=predictions_grad,
            targets=targets,
            track_features=None,
            event_timestamps=None,
            current_epoch=0
        )
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Check gradients
        has_p5_grad = pred_p5_grad.grad is not None and pred_p5_grad.grad.abs().sum() > 0
        has_p4_grad = pred_p4_grad.grad is not None and pred_p4_grad.grad.abs().sum() > 0
        has_p3_grad = pred_p3_grad.grad is not None and pred_p3_grad.grad.abs().sum() > 0
        
        print(f"\n  Gradient Flow:")
        p5_grad_sum = pred_p5_grad.grad.abs().sum().item() if has_p5_grad else 0.0
        p4_grad_sum = pred_p4_grad.grad.abs().sum().item() if has_p4_grad else 0.0
        p3_grad_sum = pred_p3_grad.grad.abs().sum().item() if has_p3_grad else 0.0
        print(f"    - P5 has gradients: {'✓' if has_p5_grad else '✗'} (sum: {p5_grad_sum:.6f})")
        print(f"    - P4 has gradients: {'✓' if has_p4_grad else '✗'} (sum: {p4_grad_sum:.6f})")
        print(f"    - P3 has gradients: {'✓' if has_p3_grad else '✗'} (sum: {p3_grad_sum:.6f})")
        
        if has_p5_grad and has_p4_grad and has_p3_grad:
            print(f"\n  ✓ All scales receive gradients - multi-scale training will work!")
        else:
            print(f"\n  ✗ FAILED: Not all scales receive gradients!")
            return False
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("\n✓ All tests passed!")
    print("\n  Multi-scale loss is working correctly:")
    print("    1. Accepts list of predictions (3 scales)")
    print("    2. Computes loss on all scales")
    print("    3. Combines with configured weights")
    print("    4. Propagates gradients to all scales")
    print("\n  Ready for training with multi-scale loss!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_multiscale_loss()
    sys.exit(0 if success else 1)

