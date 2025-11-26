#!/usr/bin/env python3
"""
Utility to calculate class weights from training data for handling imbalanced classes.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import h5py


def calculate_class_frequencies_from_loader(dataloader, num_classes: int, max_samples: int = None):
    """
    Calculate class frequencies from a data loader.
    
    Works for both 3-class (0,1,2) and 8-class (0-7) annotations.
    The num_classes parameter should match the annotation type being used.
    """
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    total_samples = 0
    
    print(f"Calculating class frequencies from dataloader (max_samples={max_samples}, num_classes={num_classes})...")
    
    # Check if dataloader is empty
    # Note: We don't consume the first batch here since iterators reset
    # But we check if the dataloader has any batches
    try:
        # Try to get first batch to check if dataloader is empty
        # This will be consumed, but iterators reset, so it's okay
        first_batch = next(iter(dataloader))
        # Check if first batch has targets
        if first_batch.get('targets', None) is None:
            print("Warning: Dataloader has no targets. Returning zero counts.")
            return class_counts, total_samples
    except StopIteration:
        print("Warning: Dataloader is empty. Returning zero counts.")
        return class_counts, total_samples
    
    # Iterate through all batches (iterator resets automatically)
    for batch_idx, batch in enumerate(dataloader):
        if max_samples is not None and total_samples >= max_samples:
            break
            
        targets = batch.get('targets', None)
        if targets is None:
            continue
        
        # Process each target in the batch
        for target in targets:
            if target is not None and len(target) > 0:
                # Extract class IDs from target
                # Target format: [num_objects, 8] = [class_id, x, y, w, h, conf, track_id, timestamp]
                if len(target.shape) == 2 and target.shape[1] >= 1:
                    class_ids = target[:, 0].long()  # Get class IDs
                    # Clamp to valid range
                    class_ids = torch.clamp(class_ids, 0, num_classes - 1)
                    # Count occurrences (vectorized)
                    unique_ids, counts = torch.unique(class_ids, return_counts=True)
                    for cls_id, count in zip(unique_ids, counts):
                        cls_id_int = int(cls_id.item())
                        # Ensure class ID is within valid range
                        if 0 <= cls_id_int < num_classes:
                            class_counts[cls_id_int] += int(count.item())
                    total_samples += len(class_ids)
    
    if total_samples == 0:
        print("Warning: No object instances found in dataloader. Returning zero counts.")
    
    print(f"Processed {total_samples} object instances across {num_classes} classes")
    return class_counts, total_samples


def calculate_class_weights_from_frequencies(class_counts: torch.Tensor, method: str = 'balanced'):
    """
    Calculate class weights from class frequencies.
    
    Args:
        class_counts: Tensor of class counts [num_classes]
        method: 'balanced' (inverse frequency) or 'sklearn' (sklearn-style balanced)
    
    Returns:
        class_weights: Tensor of class weights [num_classes]
    """
    num_classes = len(class_counts)
    
    # Handle zero counts (avoid division by zero)
    class_counts = class_counts.float()
    non_zero_mask = class_counts > 0
    
    if method == 'balanced':
        # Inverse frequency weighting: weight = total_samples / (num_classes * class_count)
        total_samples = class_counts.sum()
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        
        if total_samples > 0:
            # For non-zero classes: weight inversely proportional to frequency
            # Avoid division by zero
            non_zero_counts = class_counts[non_zero_mask]
            if len(non_zero_counts) > 0:
                class_weights[non_zero_mask] = total_samples / (num_classes * non_zero_counts)
                # For zero classes: use maximum weight (but this shouldn't happen in practice)
                if not non_zero_mask.all():  # If there are zero classes
                    class_weights[~non_zero_mask] = class_weights[non_zero_mask].max() if non_zero_mask.any() else 1.0
            else:
                # All classes have zero counts - use uniform weights
                print("Warning: All class counts are zero. Using uniform weights.")
                class_weights = torch.ones(num_classes, dtype=torch.float32)
        else:
            # Total samples is zero - use uniform weights
            print("Warning: Total samples is zero. Using uniform weights.")
            class_weights = torch.ones(num_classes, dtype=torch.float32)
    elif method == 'sklearn':
        # Sklearn-style balanced weights: n_samples / (n_classes * np.bincount(y))
        total_samples = class_counts.sum()
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        
        if total_samples > 0:
            # Weight = n_samples / (n_classes * class_count)
            # Avoid division by zero
            non_zero_counts = class_counts[non_zero_mask]
            if len(non_zero_counts) > 0:
                class_weights[non_zero_mask] = total_samples / (num_classes * non_zero_counts)
                if not non_zero_mask.all():  # If there are zero classes
                    class_weights[~non_zero_mask] = class_weights[non_zero_mask].max() if non_zero_mask.any() else 1.0
            else:
                # All classes have zero counts - use uniform weights
                print("Warning: All class counts are zero. Using uniform weights.")
                class_weights = torch.ones(num_classes, dtype=torch.float32)
        else:
            # Total samples is zero - use uniform weights
            print("Warning: Total samples is zero. Using uniform weights.")
            class_weights = torch.ones(num_classes, dtype=torch.float32)
    else:
        # Uniform weights
        class_weights = torch.ones(num_classes, dtype=torch.float32)
    
    # Normalize weights (optional: make them sum to num_classes)
    # This ensures the average weight is 1.0
    # Avoid division by zero
    weights_mean = class_weights.mean()
    if weights_mean > 0:
        class_weights = class_weights / weights_mean * num_classes
    else:
        # If all weights are zero, use uniform weights
        print("Warning: All class weights are zero. Using uniform weights.")
        class_weights = torch.ones(num_classes, dtype=torch.float32)
    
    return class_weights


def calculate_class_weights_from_annotations(data_root: str, split: str = 'train', 
                                             num_classes: int = 3,
                                             use_3_class_annotations: bool = True,
                                             annotation_dir: Optional[str] = None,
                                             max_samples: int = 10000):
    """
    Calculate class weights directly from annotation files.
    
    Args:
        data_root: Root directory containing HDF5 files
        split: Dataset split ('train', 'val', 'test')
        num_classes: Number of classes
        use_3_class_annotations: Whether to use 3-class annotations
        annotation_dir: Directory for 8-class annotations (if not using 3-class)
        max_samples: Maximum number of annotations to process
    
    Returns:
        class_weights: Tensor of class weights [num_classes]
    """
    from pathlib import Path
    
    data_root = Path(data_root)
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    total_samples = 0
    
    # Find HDF5 files
    h5_files = sorted(data_root.glob(f"**/*{split}*.h5"))
    if len(h5_files) == 0:
        h5_files = sorted(data_root.glob(f"**/*{split}*.hdf5"))
    
    print(f"Found {len(h5_files)} HDF5 files for {split} split")
    print(f"Calculating class frequencies from annotations (max_samples={max_samples})...")
    
    for h5_file in h5_files:
        if total_samples >= max_samples:
            break
            
        # Load annotations
        annotations = None
        if use_3_class_annotations:
            # Look for _bbox.npy in same folder as h5 file
            h5_name = h5_file.stem
            annotation_name = h5_name.replace("_td", "_bbox") + ".npy"
            annotation_path = h5_file.parent / annotation_name
            
            if annotation_path.exists():
                try:
                    ann_data = np.load(annotation_path, allow_pickle=True)
                    if isinstance(ann_data, np.ndarray) and len(ann_data) > 0:
                        # Annotation format: [class_id, x, y, w, h, ...]
                        if ann_data.dtype.names is not None:
                            # Structured array
                            class_ids = ann_data['class_id']
                        else:
                            # Regular array
                            class_ids = ann_data[:, 0] if len(ann_data.shape) > 1 else ann_data
                        annotations = {'class_id': class_ids}
                except Exception as e:
                    print(f"Error loading {annotation_path}: {e}")
                    continue
        else:
            # 8-class annotations from annotation_dir
            if annotation_dir:
                h5_name = h5_file.stem
                annotation_name = h5_name.replace("_td", "_bbox") + ".npy"
                annotation_path = Path(annotation_dir) / f"eight_class_annotations_{split}" / annotation_name
                
                if not annotation_path.exists():
                    annotation_path = Path(annotation_dir) / annotation_name
                
                if annotation_path.exists():
                    try:
                        ann_data = np.load(annotation_path, allow_pickle=True)
                        if isinstance(ann_data, np.ndarray) and len(ann_data) > 0:
                            if ann_data.dtype.names is not None:
                                class_ids = ann_data['class_id']
                            else:
                                class_ids = ann_data[:, 0] if len(ann_data.shape) > 1 else ann_data
                            annotations = {'class_id': class_ids}
                    except Exception as e:
                        print(f"Error loading {annotation_path}: {e}")
                        continue
        
        if annotations is not None:
            class_ids = annotations['class_id']
            # Clamp to valid range
            class_ids = np.clip(class_ids, 0, num_classes - 1)
            # Count occurrences
            for cls_id in class_ids:
                class_counts[int(cls_id)] += 1
                total_samples += 1
                if total_samples >= max_samples:
                    break
    
    print(f"Processed {total_samples} object instances across {num_classes} classes")
    print(f"Class distribution: {dict(enumerate(class_counts.numpy()))}")
    
    # Calculate weights
    class_weights = calculate_class_weights_from_frequencies(class_counts, method='balanced')
    
    return class_weights, class_counts

