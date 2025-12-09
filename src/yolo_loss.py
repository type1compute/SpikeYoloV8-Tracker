#!/usr/bin/env python3
"""
YOLO Loss Implementation for eTraM SpikeYOLO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any
import math
from ultralytics.utils.tal import TaskAlignedAssigner

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

SUPCON_TEMPERATURE = 0.07

class YOLOLoss(nn.Module):
    """
    YOLO Detection Loss Implementation with Focal Loss
    
    Computes:
    - Box regression loss (IoU-based)
    - Classification loss with Focal Loss
    - Optional tracking loss
    """
    
    def __init__(self, 
                 num_classes: int = 8,
                 box_loss_weight: float = 5.0,      # Priority 1: Box regression (increased for better localization)
                 cls_loss_weight: float = 1.5,      # Priority 2: Classification  
                 track_loss_weight: float = 0.5,    # Priority 3: Tracking
                 iou_threshold: float = 0.5,        # Standard YOLO matching threshold
                 device: str = 'cpu',
                 use_focal_loss: bool = True,       # Enable focal loss
                 focal_alpha: float = 0.25,        # Focal loss alpha
                 focal_gamma: float = 2.0,         # Focal loss gamma
                 label_smoothing: float = 0.1,     # Label smoothing for regularization
                 use_adaptive_iou: bool = False,   # Enable adaptive IoU threshold
                 adaptive_iou_start: float = 0.05, # Starting IoU threshold
                 adaptive_iou_epochs: int = 50,    # Epochs to reach final threshold
                 image_width: float = 1280.0,      # Image width for coordinate normalization
                 image_height: float = 720.0,      # Image height for coordinate normalization
                 class_weights: torch.Tensor = None,  # Class weights for handling imbalanced data
                 scale_weights: List[float] = None,  # Weights for multi-scale loss (P5, P4, P3)
                 box_loss_type: str = 'ciou'):  # Box loss type: 'iou', 'giou', 'diou', 'ciou' (default: CIoU for best performance)
        super().__init__()
        
        self.num_classes = num_classes
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.track_loss_weight = track_loss_weight
        self.iou_threshold = iou_threshold
        self.device = device
        self.box_loss_type = box_loss_type.lower()  # Normalize to lowercase
        
        # Validate box loss type
        valid_box_loss_types = ['iou', 'giou', 'diou', 'ciou']
        if self.box_loss_type not in valid_box_loss_types:
            print(f"Warning: Invalid box_loss_type '{box_loss_type}', using 'ciou' instead")
            self.box_loss_type = 'ciou'
        
        print(f"Using {self.box_loss_type.upper()} loss for box regression (better gradients for non-overlapping boxes)")
        # Note: box_loss_type affects only the regression loss computation.
        # IoU used for assignment and logging remains plain IoU for stability/comparability.
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        # Adaptive IoU threshold settings
        self.use_adaptive_iou = use_adaptive_iou
        self.adaptive_iou_start = adaptive_iou_start
        self.adaptive_iou_epochs = adaptive_iou_epochs
        self.base_iou_threshold = iou_threshold  # Store final threshold

        # Image dimensions for coordinate normalization
        self.image_width = image_width
        self.image_height = image_height

        # Temperature parameter for contrastive loss calculation
        self.supcon_temperature = SUPCON_TEMPERATURE

        # DFL integration (when training head outputs per-side logits)
        self.dfl_loss_weight = getattr(self, "dfl_loss_weight", 0.5)  # can be overridden by config
        self.reg_max = getattr(self, "reg_max", 16)  # default number of distance bins per side

        # Multi-scale weights (P5, P4, P3)
        # Default: Favor finer scales slightly (P3 > P4 > P5) for better small object detection
        if scale_weights is None:
            scale_weights = [0.4, 0.35, 0.25]  # P5, P4, P3
        self.scale_weights = scale_weights
        print(f"Multi-scale loss weights: P5={scale_weights[0]:.2f}, P4={scale_weights[1]:.2f}, P3={scale_weights[2]:.2f}")

        # Class weights for handling imbalanced data
        if class_weights is not None:
            if isinstance(class_weights, (list, np.ndarray)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            self.class_weights = class_weights.to(device)
            # Ensure weights are normalized (sum to num_classes)
            # Avoid division by zero
            weights_sum = self.class_weights.sum()
            if weights_sum > 0:
                self.class_weights = self.class_weights / weights_sum * num_classes
            else:
                # If all weights are zero, use uniform weights
                print("Warning: All class weights are zero. Using uniform weights.")
                self.class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
            print(f"Class weights initialized: {self.class_weights.cpu().numpy()}")
        else:
            self.class_weights = None

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

        # --- Task-Aligned Assigner (Ultralytics TAL) ---
        self.use_task_aligned_assigner = True
        self.ta_topk = 10  # top-k anchors per GT
        self.ta_alpha = 0.5  # classification prior exponent
        self.ta_beta = 6.0  # IoU exponent
        self.tal_assigner = TaskAlignedAssigner(
            topk=self.ta_topk,
            num_classes=self.num_classes,
            alpha=self.ta_alpha,
            beta=self.ta_beta
        )

    def forward(self,
                predictions: torch.Tensor,
                targets: List[torch.Tensor],
                track_features: torch.Tensor = None,
                event_timestamps: torch.Tensor = None,
                current_epoch: int = 0) -> Dict[str, torch.Tensor]:
        # Handle 5D predictions from train head: [T, B, (4*reg_max + C), H, W]
        if isinstance(predictions, torch.Tensor) and predictions.dim() == 5:
            T5, B5, Cfeat5, H5, W5 = predictions.shape
            # -> [T, B, H*W, (4*reg_max + C)]
            predictions = predictions.permute(0, 1, 3, 4, 2).contiguous().view(T5, B5, H5 * W5, Cfeat5)
        """
        Compute YOLO loss with multi-scale support

        Args:
            predictions: Model outputs - either single tensor or list of tensors for multi-scale
                        Single: [batch_size, num_anchors, features]
                        Multi-scale: List of [batch_size, num_anchors_i, features] for each scale
                        Format: [x, y, w, h, class_0, ..., class_N-1] (no objectness, no track_id)
            targets: List of target tensors for each batch item
                     Each target: [num_objects, 8] = [class_id, x, y, w, h, conf, track_id, timestamp]
                     Note: timestamp is metadata, not part of model predictions
            track_features: Optional tracking features [batch_size, track_feature_dim]
            event_timestamps: Event timestamps [batch_size, num_events] for temporal-aware matching

        Returns:
            Dictionary with individual loss components
        """

        # Handle list of predictions (multi-scale output)
        if isinstance(predictions, list) and len(predictions) > 1:
            # MULTI-SCALE LOSS: Compute loss on all scales and combine with weights
            return self._compute_multiscale_loss(
                predictions, targets, track_features, event_timestamps, current_epoch
            )

        # Single scale or list with single element
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Handle list of tracking features (multi-scale output)
        if isinstance(track_features, list):
            # Use the first scale (P5) for tracking loss computation
            track_features = track_features[0]

        # Check for temporal dimension
        if predictions.dim() == 4:
            # Temporal-aware predictions: [T, B, H*W, features]
            T, B, num_anchors, num_features = predictions.shape
            # Reshape to [T*B, H*W, features] to process each temporal step separately
            predictions = predictions.view(T * B, num_anchors, num_features)
            batch_size = B  # Original batch size
            temporal_batch_size = T * B  # Expanded batch size
        elif predictions.dim() == 3:
            batch_size = predictions.shape[0]
            temporal_batch_size = batch_size
            T = 1  # No temporal dimension
        else:
            raise ValueError(f"Expected predictions shape [B, N, 4+C] or [T, B, N, 4+C], got {predictions.shape}")

        if targets and isinstance(targets[0], torch.Tensor):
            # Expand targets for temporal dimension if needed
            if T > 1:
                # Filter targets by temporal bin for each temporal step
                # This ensures predictions from temporal step t only match annotations in temporal bin t
                expanded_targets = []

                # Calculate temporal bin boundaries from event timestamps if available
                # Otherwise, use a simple approach (divide time window equally)
                if event_timestamps is not None and len(event_timestamps) > 0:
                    # Get event timestamps for each batch item
                    # event_timestamps can be a list of tensors or a tensor
                    for batch_idx in range(B):
                        batch_event_timestamps = None
                        if isinstance(event_timestamps, (list, tuple)):
                            # List of tensors, one per batch item
                            if batch_idx < len(event_timestamps):
                                batch_event_timestamps = event_timestamps[batch_idx]
                        elif isinstance(event_timestamps, torch.Tensor):
                            # Single tensor with batch dimension
                            if event_timestamps.dim() == 1:
                                # 1D tensor: [num_events] - same for all batch items
                                batch_event_timestamps = event_timestamps
                            elif event_timestamps.dim() == 2:
                                # 2D tensor: [batch_size, num_events]
                                if batch_idx < event_timestamps.shape[0]:
                                    batch_event_timestamps = event_timestamps[batch_idx]

                        if batch_event_timestamps is not None and len(batch_event_timestamps) > 0:
                            # Ensure it's a tensor
                            if not isinstance(batch_event_timestamps, torch.Tensor):
                                batch_event_timestamps = torch.tensor(batch_event_timestamps, device=self.device)

                            # Remove padding (negative timestamps)
                            valid_timestamps = batch_event_timestamps[batch_event_timestamps >= 0]

                            if len(valid_timestamps) > 0:
                                # Calculate temporal bin boundaries (same as events_to_spike_frames)
                                t_min = float(valid_timestamps.min())
                                t_max = float(valid_timestamps.max())
                                time_bins = torch.linspace(t_min, t_max, T + 1, device=valid_timestamps.device)

                                # Get annotations for this batch item
                                batch_targets = targets[batch_idx]  # [num_annotations, 8]

                                # Ensure batch_targets are on the same device as time_bins
                                if batch_targets.device != valid_timestamps.device:
                                    batch_targets = batch_targets.to(valid_timestamps.device)

                                # Filter annotations per temporal step
                                for t in range(T):
                                    bin_start = time_bins[t]
                                    bin_end = time_bins[t + 1]

                                    # Extract annotation timestamps (index 7)
                                    if len(batch_targets) > 0:
                                        ann_timestamps = batch_targets[:, 7]  # [num_annotations]

                                        # Filter annotations in this temporal bin
                                        # Use inclusive boundaries to avoid missing annotations at bin edges
                                        # This ensures no annotation is lost due to floating point precision
                                        bin_mask = (ann_timestamps >= bin_start) & (ann_timestamps <= bin_end)

                                        # Get filtered annotations for this temporal bin
                                        if bin_mask.any():
                                            filtered_targets = batch_targets[bin_mask]
                                        else:
                                            # No annotations in this temporal bin
                                            filtered_targets = torch.zeros((0, 8), device=batch_targets.device, dtype=batch_targets.dtype)

                                        expanded_targets.append(filtered_targets)
                                    else:
                                        # No annotations for this batch item
                                        expanded_targets.append(torch.zeros((0, 8), device=valid_timestamps.device, dtype=torch.float32))
                            else:
                                # No valid timestamps, fallback to repeating all annotations
                                for t in range(T):
                                    expanded_targets.append(targets[batch_idx])
                        else:
                            # No event timestamps for this batch item, fallback to repeating all annotations
                            for t in range(T):
                                expanded_targets.append(targets[batch_idx])
                else:
                    # No event timestamps available, fallback to repeating all annotations
                    # (This maintains backward compatibility)
                    for batch_idx in range(B):
                        for t in range(T):
                            expanded_targets.append(targets[batch_idx])

                targets = expanded_targets
        loss_dict = {
            'box_loss': 0.0,
            'dfl_loss': 0.0,
            'cls_loss': 0.0,
            'track_loss': 0.0,
            'total_loss': 0.0
        }

        # Collect matched anchor information for tracking loss (using track_id from annotations)
        matched_anchor_info = []  # List of (batch_idx, anchor_idx, track_id) tuples

        # Track how many batches had valid targets for proper averaging
        num_valid_batches = 0

        # Process each batch item (including temporal steps)
        for batch_idx in range(temporal_batch_size):
            pred = predictions[batch_idx]  # [num_anchors, 4 + num_classes]

            # Map temporal batch index to original batch index
            original_batch_idx = batch_idx % B if T > 1 else batch_idx

            target = targets[batch_idx] if batch_idx < len(targets) else torch.zeros((0, 8), device=self.device)

            # Extract event timestamps for this batch item if available
            batch_event_timestamps = None
            if event_timestamps is not None:
                # Use original batch index for event timestamps
                if original_batch_idx < len(event_timestamps):
                    batch_event_timestamps = event_timestamps[original_batch_idx]  # [num_events]

            if len(target) == 0:
                # No targets - skip this batch (background samples are handled via classification loss)
                continue

            # Count this as a valid batch with targets
            num_valid_batches += 1

            # Compute losses for this batch item with temporal information
            batch_losses = self._compute_batch_loss(pred, target, batch_event_timestamps, current_epoch)

            # Accumulate losses
            for key, value in batch_losses.items():
                loss_dict[key] += value

            # Collect matched anchor information for tracking loss
            if track_features is not None and len(target) > 0:
                matched_info = self._extract_matched_anchor_info(pred, target, batch_event_timestamps)
                for anchor_idx, track_id in matched_info:
                    matched_anchor_info.append((batch_idx, anchor_idx, track_id))

        # Average over only the batches that had valid targets (not all temporal_batch_size)
        # This prevents diluting the loss signal from sparse annotations
        if num_valid_batches > 0:
            for key in ('box_loss', 'dfl_loss', 'cls_loss', 'track_loss'):
                loss_dict[key] /= num_valid_batches
        else:
            # No valid batches - all losses remain zero
            pass

        # Compute feature-based tracking loss if provided
        if track_features is not None:
            # Contrastive loss using track_id from annotations
            # Same track_id = positive pairs (should be similar)
            # Different track_id = negative pairs (should be dissimilar)
            track_loss_from_features = self._compute_tracking_loss(
                track_features, matched_anchor_info, predictions
            )
            loss_dict['track_loss'] = track_loss_from_features
            # Details are already stored in _last_track_loss_details by _compute_tracking_loss
        else:
            # Use tracking loss from batch items (if any)
            pass  # Already accumulated in loss_dict['track_loss']

        # Compute total loss - weighted sum of all components
        loss_dict['total_loss'] = (
            self.box_loss_weight * loss_dict['box_loss'] +
            self.dfl_loss_weight  * loss_dict['dfl_loss'] +
            self.cls_loss_weight  * loss_dict['cls_loss'] +
            self.track_loss_weight * loss_dict['track_loss']
        )

        return loss_dict

    def _compute_multiscale_loss(self,
                                 predictions_list: List[torch.Tensor],
                                 targets: List[torch.Tensor],
                                 track_features: torch.Tensor = None,
                                 event_timestamps: torch.Tensor = None,
                                 current_epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute loss across multiple detection scales and combine with weights.

        Args:
            predictions_list: List of predictions from different scales [P5, P4, P3]
                             Each element: [batch_size, num_anchors_i, features]
            targets: List of target tensors for each batch item
            track_features: Optional tracking features (list or single tensor)
            event_timestamps: Event timestamps for temporal matching
            current_epoch: Current training epoch for adaptive IoU

        Returns:
            Dictionary with combined loss components across all scales
        """
        num_scales = len(predictions_list)

        # Start with provided weights (copy slice), extend with ones if missing, then normalize
        weights = list(self.scale_weights[:num_scales])
        if len(weights) < num_scales:
            weights.extend([1.0] * (num_scales - len(weights)))
        total_w = sum(weights)
        if total_w <= 0:
            # Fallback to uniform if something went wrong
            normalized_weights = [1.0 / num_scales] * num_scales
        else:
            normalized_weights = [w / total_w for w in weights]

        # Initialize combined loss dictionary
        combined_loss_dict = {
            'box_loss': 0.0,
            'dfl_loss': 0.0,
            'cls_loss': 0.0,
            'track_loss': 0.0,
            'total_loss': 0.0
        }

        # Compute loss for each scale and accumulate with weights
        scale_names = ['P5 (coarse)', 'P4 (medium)', 'P3 (fine)']
        for scale_idx, scale_predictions in enumerate(predictions_list):
            scale_name = scale_names[scale_idx] if scale_idx < len(scale_names) else f'Scale {scale_idx}'
            scale_weight = normalized_weights[scale_idx]

            # Extract tracking features for this scale if available
            scale_track_features = None
            if track_features is not None:
                if isinstance(track_features, list):
                    scale_track_features = track_features[scale_idx] if scale_idx < len(track_features) else None
                else:
                    # Single tracking feature tensor - use for first scale only
                    scale_track_features = track_features if scale_idx == 0 else None

            # Recursively call forward with single scale (will use the single-scale path)
            scale_loss_dict = self.forward(
                predictions=scale_predictions,
                targets=targets,
                track_features=scale_track_features,
                event_timestamps=event_timestamps,
                current_epoch=current_epoch
            )

            # Accumulate weighted component losses (do not include total_loss from subcalls)
            for key in ('box_loss', 'dfl_loss', 'cls_loss', 'track_loss'):
                if key in scale_loss_dict:
                    combined_loss_dict[key] += scale_weight * scale_loss_dict[key]

        # Compute overall total loss from combined components
        combined_loss_dict['total_loss'] = (
            self.box_loss_weight * combined_loss_dict['box_loss'] +
            self.dfl_loss_weight  * combined_loss_dict['dfl_loss'] +
            self.cls_loss_weight  * combined_loss_dict['cls_loss'] +
            self.track_loss_weight * combined_loss_dict['track_loss']
        )

        return combined_loss_dict

    def _project_dfl(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Integral projection for DFL: logits [N, M] -> expectation over bins in [0, M-1].
        Returns distances in 'bin units'.
        """
        # logits: [N, M]; softmax -> probs
        probs = torch.softmax(logits, dim=-1)
        M = probs.shape[-1]
        bins = torch.arange(M, device=probs.device, dtype=probs.dtype).view(1, -1)
        return (probs * bins).sum(dim=-1)  # [N]

    def _build_dfl_targets(self, gt_ltrb: torch.Tensor, reg_max: int) -> torch.Tensor:
        """
        Build soft DFL targets for each of l,t,r,b distances.
        gt_ltrb: [P, 4] distances in 'bin units' (i.e., scaled to 0..reg_max)
        Returns: soft labels [P, 4, reg_max]
        """
        P = gt_ltrb.shape[0]
        M = reg_max
        tgt = torch.zeros((P, 4, M), device=gt_ltrb.device, dtype=gt_ltrb.dtype)

        # For each side, split fractional target into two neighboring bins
        for q in range(4):
            v = gt_ltrb[:, q].clamp(0, M - 1 - 1e-6)  # keep inside range
            l = v.floor().long()
            r = l + 1
            w_r = v - l.float()
            w_l = 1.0 - w_r
            tgt[torch.arange(P, device=tgt.device), q, l] += w_l
            # Only add right bin if it’s within range
            in_range = r < M
            if in_range.any():
                idx = torch.arange(P, device=tgt.device)[in_range]
                tgt[idx, q, r[in_range]] += w_r[in_range]
        return tgt  # [P,4,M]

    def _dfl_ce_loss(self, pred_logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy for DFL with soft targets.
        pred_logits: [P, 4, M], soft_targets: [P, 4, M]
        """
        logp = torch.log_softmax(pred_logits, dim=-1)  # [P,4,M]
        loss = -(soft_targets * logp).sum(dim=-1).mean()  # mean over sides and samples
        return loss

    def _compute_batch_loss(self, pred: torch.Tensor, target: torch.Tensor, event_timestamps: torch.Tensor = None, current_epoch: int = 0) -> Dict[str, torch.Tensor]:
        """Compute loss for a single batch item with optional temporal matching"""

        # Update IoU threshold if adaptive IoU is enabled
        if self.use_adaptive_iou:
            # Linearly interpolate from adaptive_iou_start to base_iou_threshold over adaptive_iou_epochs
            if current_epoch < self.adaptive_iou_epochs:
                progress = current_epoch / self.adaptive_iou_epochs
                self.iou_threshold = self.adaptive_iou_start + (self.base_iou_threshold - self.adaptive_iou_start) * progress
            else:
                self.iou_threshold = self.base_iou_threshold

        # Ensure both pred and target are on the same device
        # Determine device - prefer CUDA if available, otherwise use self.device
        if pred.is_cuda:
            device = pred.device
        elif isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device

        # Move target to the same device as pred
        target = target.to(device)
        if event_timestamps is not None and isinstance(event_timestamps, torch.Tensor):
            event_timestamps = event_timestamps.to(device)

        num_anchors = pred.shape[0]
        num_targets = len(target) if isinstance(target, torch.Tensor) else target.shape[0]
        num_features = pred.shape[1]

        # Support two train-time formats:
        #  A) per-side DFL logits + cls: [4*reg_max + C]
        #  B) decoded xywh + cls:        [4 + C]  (no objectness)
        reg_max = getattr(self, "reg_max", 16)
        A_dfl = 4 * reg_max + self.num_classes
        A_dec = 4 + self.num_classes

        pred_boxes_normalized = None
        pred_cls = None
        pred_dfl_logits = None  # [A, 4, reg_max] if present

        if num_features == A_dfl:

            # RAW per-side DFL logits and class logits
            pred_dist = pred[:, :4 * reg_max]
            pred_cls  = pred[:, 4 * reg_max : 4 * reg_max + self.num_classes]
            pred_dfl_logits = pred_dist.view(-1, 4, reg_max)  # [A,4,M]

            # Decode distances via integral projection to get xywh (normalized)
            A = pred.shape[0]
            H, W = self._infer_hw_from_num_cells(A)
            anc_points = self._make_grid_points(H, W, pred.device, pred.dtype)  # [A,2] in 0..1

            per_side_expect = self._project_dfl(pred_dfl_logits.view(-1, reg_max)).view(-1, 4)  # [A,4] bins
            l, t, r, b = (per_side_expect / float(reg_max)).unbind(dim=1)  # 0..1 distances
            cx, cy = anc_points[:, 0], anc_points[:, 1]
            x1 = (cx - l).clamp(0, 1); y1 = (cy - t).clamp(0, 1)
            x2 = (cx + r).clamp(0, 1); y2 = (cy + b).clamp(0, 1)
            w  = (x2 - x1).clamp_min(1e-6); h = (y2 - y1).clamp_min(1e-6)
            x  = 0.5 * (x1 + x2); y = 0.5 * (y1 + y2)
            pred_boxes_normalized = torch.stack([x, y, w, h], dim=1)  # [A,4]
        elif num_features == A_dec:
            pred_boxes_normalized = pred[:, :4]
            pred_cls = pred[:, 4 : 4 + self.num_classes]
        else:
            raise ValueError(
                f"Prediction tensor has unexpected features: {num_features}, expected {A_dfl} (DFL) or {A_dec} (decoded)."
            )

        # Validate target shape
        expected_target_features = 8  # [class_id, x, y, w, h, conf, track_id, timestamp]
        if target.shape[1] < expected_target_features:
            raise ValueError(f"Target tensor has insufficient features: {target.shape[1]}, expected at least {expected_target_features}")

        # Extract targets
        target_cls = target[:, 0].long()  # Class IDs

        # CRITICAL FIX: Clamp class IDs to valid range to prevent CUDA device-side asserts
        target_cls = torch.clamp(target_cls, 0, self.num_classes - 1)

        target_boxes = target[:, 1:5]      # [x, y, w, h] - in centered pixel coordinates
        target_conf = target[:, 5].clamp(0.0, 1.0)  # Annotation-provided class confidence per GT
        target_track_id = target[:, 6]     # Track ID
        # Note: target[:, 7] = timestamp is metadata, not used in loss computation

        # CRITICAL FIX: Convert target boxes from pixel coordinates to normalized [0,1]
        # Use configurable image dimensions
        # Targets are already in CENTER format (xywh) from the dataloader.
        # Normalize to [0,1] **only if needed** (i.e., if values look like pixels > 1).
        target_boxes_normalized = target_boxes.clone()
        if torch.any(target_boxes_normalized > 1.0) or torch.any(target_boxes_normalized < 0.0):
            # Pixels -> normalized using configured image dimensions
            target_boxes_normalized[:, 0] = target_boxes[:, 0] / self.image_width  # x_center
            target_boxes_normalized[:, 1] = target_boxes[:, 1] / self.image_height  # y_center
            target_boxes_normalized[:, 2] = target_boxes[:, 2] / self.image_width  # width
            target_boxes_normalized[:, 3] = target_boxes[:, 3] / self.image_height  # height
        # Clamp to valid range
        target_boxes_normalized = torch.clamp(target_boxes_normalized, 0.0, 1.0)
        # Initialize losses with proper device and dtype
        box_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        cls_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        dfl_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        if num_targets > 0:
            # DEBUG: Check for NaN/Inf in inputs before IoU
            if torch.isnan(pred_boxes_normalized).any() or torch.isinf(pred_boxes_normalized).any():
                print(f"ERROR: NaN/Inf in pred_boxes_normalized!")
                print(f"  pred_boxes stats: min={pred_boxes_normalized.min():.4f}, max={pred_boxes_normalized.max():.4f}, mean={pred_boxes_normalized.mean():.4f}")
                print(f"  NaN count: {torch.isnan(pred_boxes_normalized).sum()}, Inf count: {torch.isinf(pred_boxes_normalized).sum()}")
                zero = torch.tensor(0.0, device=device)
                return {
                    'box_loss': box_loss,
                    'dfl_loss': zero,
                    'cls_loss': cls_loss,
                    'track_loss': zero,
                    'total_loss': self.box_loss_weight * box_loss + self.dfl_loss_weight * zero + self.cls_loss_weight * cls_loss
                }

            if torch.isnan(target_boxes_normalized).any() or torch.isinf(target_boxes_normalized).any():
                print(f"ERROR: NaN/Inf in target_boxes_normalized!")
                print(f"  target_boxes stats: min={target_boxes_normalized.min():.4f}, max={target_boxes_normalized.max():.4f}, mean={target_boxes_normalized.mean():.4f}")
                zero = torch.tensor(0.0, device=device)
                return {
                    'box_loss': box_loss,
                    'dfl_loss': zero,
                    'cls_loss': cls_loss,
                    'track_loss': zero,
                    'total_loss': self.box_loss_weight * box_loss + self.dfl_loss_weight * zero + self.cls_loss_weight * cls_loss
                }

            # Compute IoU between normalized predictions and normalized targets
            ious = self._compute_iou(pred_boxes_normalized, target_boxes_normalized)

            # Check for NaN in IoU
            if torch.isnan(ious).any():
                print(f"WARNING: NaN detected in IoU computation (after valid inputs!)")
                print(f"  IoU stats: min={ious[~torch.isnan(ious)].min():.4f}, max={ious[~torch.isnan(ious)].max():.4f}")
                zero = torch.tensor(0.0, device=device)
                return {
                    'box_loss': box_loss,
                    'dfl_loss': zero,
                    'cls_loss': cls_loss,
                    'track_loss': zero,
                    'total_loss': self.box_loss_weight * box_loss + self.dfl_loss_weight * zero + self.cls_loss_weight * cls_loss
                }

            # Temporal-aware matching if event_timestamps available
            if event_timestamps is not None and len(event_timestamps) > 0:
                best_ious, best_indices = self._temporal_aware_matching(
                    pred_boxes_normalized,
                    target,
                    target_boxes_normalized,
                    event_timestamps,
                    pred_cls=pred_cls,  # <<< pass logits so TAL can run
                    time_window_frac=0.2  # tweak if bins are tight/loose
                )
            else:
                # TAL without temporal pre-filtering (if you still want plain TAL here)
                A = pred_boxes_normalized.shape[0]
                G = target_boxes_normalized.shape[0]
                pd_scores = pred_cls.unsqueeze(0)
                pred_xyxy = self._xywh_to_xyxy(pred_boxes_normalized)
                pd_bboxes = pred_xyxy.unsqueeze(0)
                H, W = self._infer_hw_from_num_cells(A)
                anc_points = self._make_grid_points(H, W, pred_boxes_normalized.device,
                                                    pred_boxes_normalized.dtype)
                gt_labels = target_cls.view(1, G, 1).to(pred_boxes_normalized.device)
                gt_bboxes_xyxy = self._xywh_to_xyxy(target_boxes_normalized).view(1, G, 4)
                mask_gt = torch.ones((1, G, 1), device=pred_boxes_normalized.device, dtype=torch.bool)
                _, _, _, fg_mask, target_gt_idx = self.tal_assigner(pd_scores, pd_bboxes, anc_points, gt_labels,
                                                                    gt_bboxes_xyxy, mask_gt)
                ious_full = self._compute_iou(pred_boxes_normalized, target_boxes_normalized)
                best_indices = target_gt_idx.view(-1)
                best_ious = torch.zeros(A, device=pred_boxes_normalized.device, dtype=pred_boxes_normalized.dtype)
                if fg_mask.any():
                    am = fg_mask.view(-1)
                    best_ious[am] = ious_full[am, best_indices[am]]

            # Compute IoU statistics for monitoring
            max_iou = best_ious.max().item()
            min_iou = best_ious.min().item()
            mean_iou = best_ious.mean().item()
            matches = (best_ious > self.iou_threshold).sum().item()

            # Store IoU statistics for periodic logging
            if not hasattr(self, '_iou_stats'):
                self._iou_stats = {'max': [], 'mean': [], 'matches': [], 'total': []}
            self._iou_stats['max'].append(max_iou)
            self._iou_stats['mean'].append(mean_iou)
            self._iou_stats['matches'].append(matches)
            self._iou_stats['total'].append(num_anchors)

            # Log IoU statistics every 100 batches to monitor learning progress
            if len(self._iou_stats['max']) % 100 == 0:
                avg_max_iou = sum(self._iou_stats['max'][-100:]) / 100
                avg_mean_iou = sum(self._iou_stats['mean'][-100:]) / 100
                total_matches = sum(self._iou_stats['matches'][-100:])
                total_anchors = sum(self._iou_stats['total'][-100:])
                match_rate = (total_matches / total_anchors * 100) if total_anchors > 0 else 0
                print(
                    f"IoU Stats (last 100 batches, independent of box_loss_type={self.box_loss_type}): "
                    f"avg_max={avg_max_iou:.3f}, avg_mean={avg_mean_iou:.3f}, "
                    f"match_rate={match_rate:.1f}%, threshold={self.iou_threshold:.3f}"
                )
                # Limit stored stats to prevent memory growth
                if len(self._iou_stats['max']) > 1000:
                    for key in self._iou_stats:
                        self._iou_stats[key] = self._iou_stats[key][-500:]

            # Determine matched vs unmatched anchors
            matched_mask = best_ious > self.iou_threshold

            # CRITICAL: Apply classification loss to ALL anchors (matched + unmatched)
            # This is how the model learns background (negative samples)

            # Initialize classification targets for ALL anchors as background (all zeros)
            # Shape: [num_anchors, num_classes] filled with zeros (background)
            # Use the device from pred/target (which should be CUDA if available)
            device = pred_boxes_normalized.device if pred_boxes_normalized.is_cuda else (torch.device(self.device) if isinstance(self.device, str) else self.device)
            cls_targets_all = torch.zeros(num_anchors, self.num_classes, device=device)

            # For MATCHED anchors, compute box and class losses
            if matched_mask.any():
                matched_indices = matched_mask.nonzero(as_tuple=True)[0]
                matched_target_indices = best_indices[matched_mask]
                # Validate indices are within bounds
                matched_target_indices = torch.clamp(matched_target_indices, 0, num_targets - 1)

                # Box regression loss (only for matched anchors)
                matched_pred_boxes = pred_boxes_normalized[matched_mask]
                matched_target_boxes = target_boxes_normalized[matched_target_indices]
                box_loss = self._compute_box_loss(matched_pred_boxes, matched_target_boxes)
                matched_target_cls = target_cls[matched_target_indices]
                matched_target_conf = target_conf[matched_target_indices].clamp(0.0, 1.0)
                # Weight the box loss by the average GT confidence of the matched pairs
                if matched_target_indices.numel() > 0:
                    box_conf_w = matched_target_conf.mean().item()
                    box_loss = box_loss * box_conf_w
                # Debug (lightweight):
                # print(f"Box loss weighted by mean GT conf={box_conf_w:.3f}; matched={matched_indices.shape[0]}")
                one_hot = F.one_hot(matched_target_cls.long(), num_classes=self.num_classes).float()
                # Use annotation confidence as a soft label target (class logit is supervised toward the provided confidence)
                cls_targets_all[matched_indices] = one_hot * matched_target_conf.unsqueeze(1)

                # If per-side logits are available, compute DFL CE on matched anchors
                if pred_dfl_logits is not None and matched_indices.numel() > 0:
                    # Build soft DFL targets in bin space from GT vs anchor points
                    gt_xyxy = self._xywh_to_xyxy(matched_target_boxes)           # normalized
                    anc_m   = anc_points[matched_indices] if 'anc_points' in locals() else \
                              self._make_grid_points(*self._infer_hw_from_num_cells(num_anchors),
                                                     pred_boxes_normalized.device, pred_boxes_normalized.dtype)[matched_indices]
                    l_t = (anc_m[:, 0] - gt_xyxy[:, 0]).clamp(0, 1)
                    t_t = (anc_m[:, 1] - gt_xyxy[:, 1]).clamp(0, 1)
                    r_t = (gt_xyxy[:, 2] - anc_m[:, 0]).clamp(0, 1)
                    b_t = (gt_xyxy[:, 3] - anc_m[:, 1]).clamp(0, 1)
                    gt_ltrb_bins = torch.stack([l_t, t_t, r_t, b_t], dim=1) * float(reg_max)  # [P,4] in bins
                    soft_targets = self._build_dfl_targets(gt_ltrb_bins, reg_max)             # [P,4,M]
                    pred_logits_m = pred_dfl_logits[matched_indices]                          # [P,4,M]
                    dfl_loss = self._dfl_ce_loss(pred_logits_m, soft_targets)
            else:
                # No matches found - all anchors are background
                # Debug logging disabled for performance
                # max_iou = best_ious.max().item() if num_targets > 0 else 0.0
                # min_iou = best_ious.min().item() if num_targets > 0 else 0.0
                # mean_iou = best_ious.mean().item() if num_targets > 0 else 0.0
                # print(f"DEBUG: No matches found - all {num_anchors} anchors are background (threshold={self.iou_threshold}, max_iou={max_iou:.3f}, min_iou={min_iou:.3f}, mean_iou={mean_iou:.3f})")
                # cls_targets_all is already all zeros (background)
                pass

            # CRITICAL: Compute classification loss for ALL anchors (matched + unmatched)
            # Matched anchors: learn to predict the correct class (one-hot)
            # Unmatched anchors: learn to predict background (all zeros)
            # Focal loss will automatically down-weight easy examples (backgrounds)
            cls_loss = self._compute_classification_loss_from_one_hot(pred_cls, cls_targets_all)

        return {
            'box_loss': box_loss,
            'dfl_loss': dfl_loss,
            'cls_loss': cls_loss
        }

    # Plain IoU for matching/statistics; intentionally not tied to box_loss_type.
    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes"""

        # Ensure both tensors are on the same device
        device = boxes1.device if boxes1.is_cuda else (torch.device(self.device) if isinstance(self.device, str) else self.device)
        boxes1 = boxes1.to(device)
        boxes2 = boxes2.to(device)

        # Convert to [x1, y1, x2, y2] format
        boxes1_xyxy = self._xywh_to_xyxy(boxes1)
        boxes2_xyxy = self._xywh_to_xyxy(boxes2)

        # Compute intersection
        x1 = torch.max(boxes1_xyxy[:, 0:1], boxes2_xyxy[:, 0:1].T)
        y1 = torch.max(boxes1_xyxy[:, 1:2], boxes2_xyxy[:, 1:2].T)
        x2 = torch.min(boxes1_xyxy[:, 2:3], boxes2_xyxy[:, 2:3].T)
        y2 = torch.min(boxes1_xyxy[:, 3:4], boxes2_xyxy[:, 3:4].T)

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Compute areas
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])

        # Compute union
        union = area1[:, None] + area2[None, :] - intersection

        # Compute IoU
        iou = intersection / (union + 1e-6)

        return iou

    def _infer_hw_from_num_cells(self, A: int) -> Tuple[int, int]:
        """Infer (H, W) from number of prediction **cells** A (anchor-free) by factoring A, preferring H<=W and closest factors.
            Works for non-square maps like 23×40, 45×80, 90×160, etc.
        """
        best = (1, A)
        dmin = A
        # Iterate up to sqrt(A) to find factor pairs
        for h in range(1, int(math.sqrt(A)) + 1):
            if A % h == 0:
                w = A // h
                # Prefer w >= h and minimal difference between w and h
                if w >= h and (w - h) < dmin:
                    dmin = w - h
                    best = (h, w)
        return best

    def _make_grid_points(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """Create normalized grid-center points of shape [A,2] with A=H*W (anchor-free).
        Points are ((j+0.5)/W, (i+0.5)/H) in (x,y) order.
        """
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
        anc = torch.stack([xx, yy], dim=-1).view(-1, 2)  # [A,2]
        return anc

    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        # Clamp to [0, 1] to ensure valid normalized coordinates
        # xyxy = torch.clamp(xyxy, 0.0, 1.0) # Clamping here could hide OOB errors and give seemingly good results even though it isn't
        return xyxy

    def _compute_box_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute box regression loss using IoU/GIoU/DIoU/CIoU"""

        if self.box_loss_type == 'iou':
            return self._compute_iou_loss(pred_boxes, target_boxes)
        elif self.box_loss_type == 'giou':
            return self._compute_giou_loss(pred_boxes, target_boxes)
        elif self.box_loss_type == 'diou':
            return self._compute_diou_loss(pred_boxes, target_boxes)
        elif self.box_loss_type == 'ciou':
            return self._compute_ciou_loss(pred_boxes, target_boxes)
        else:
            # Fallback to IoU
            return self._compute_iou_loss(pred_boxes, target_boxes)

    def _compute_iou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute basic IoU loss (1 - IoU)"""
        # Ensure both tensors are on the same device
        device = pred_boxes.device if pred_boxes.is_cuda else (torch.device(self.device) if isinstance(self.device, str) else self.device)
        pred_boxes = pred_boxes.to(device)
        target_boxes = target_boxes.to(device)

        pred_xyxy = self._xywh_to_xyxy(pred_boxes)
        target_xyxy = self._xywh_to_xyxy(target_boxes)

        intersection = torch.clamp(
            torch.min(pred_xyxy[:, 2:], target_xyxy[:, 2:]) -
            torch.max(pred_xyxy[:, :2], target_xyxy[:, :2]),
            min=0
        ).prod(dim=1)

        area_pred = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
        area_target = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
        union = area_pred + area_target - intersection

        iou = intersection / (union + 1e-6)
        return (1 - iou).mean()

    def _compute_giou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute Generalized IoU (GIoU) loss - provides gradients even when boxes don't overlap"""
        # Ensure both tensors are on the same device
        device = pred_boxes.device if pred_boxes.is_cuda else (torch.device(self.device) if isinstance(self.device, str) else self.device)
        pred_boxes = pred_boxes.to(device)
        target_boxes = target_boxes.to(device)

        pred_xyxy = self._xywh_to_xyxy(pred_boxes)
        target_xyxy = self._xywh_to_xyxy(target_boxes)

        # Compute IoU
        intersection = torch.clamp(
            torch.min(pred_xyxy[:, 2:], target_xyxy[:, 2:]) -
            torch.max(pred_xyxy[:, :2], target_xyxy[:, :2]),
            min=0
        ).prod(dim=1)

        area_pred = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
        area_target = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
        union = area_pred + area_target - intersection

        iou = intersection / (union + 1e-6)

        # Compute enclosing box (smallest box containing both pred and target)
        enclose_x1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
        enclose_y1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
        enclose_x2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
        enclose_y2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        # GIoU = IoU - (enclose_area - union) / enclose_area
        giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
        return (1 - giou).mean()

    def _compute_diou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute Distance IoU (DIoU) loss - considers center distance and box size"""
        # Ensure both tensors are on the same device
        device = pred_boxes.device if pred_boxes.is_cuda else (torch.device(self.device) if isinstance(self.device, str) else self.device)
        pred_boxes = pred_boxes.to(device)
        target_boxes = target_boxes.to(device)

        pred_xyxy = self._xywh_to_xyxy(pred_boxes)
        target_xyxy = self._xywh_to_xyxy(target_boxes)

        # Compute IoU
        intersection = torch.clamp(
            torch.min(pred_xyxy[:, 2:], target_xyxy[:, 2:]) -
            torch.max(pred_xyxy[:, :2], target_xyxy[:, :2]),
            min=0
        ).prod(dim=1)

        area_pred = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
        area_target = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
        union = area_pred + area_target - intersection

        iou = intersection / (union + 1e-6)

        # Compute center points
        pred_center_x = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
        pred_center_y = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
        target_center_x = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
        target_center_y = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2

        # Compute center distance squared
        center_distance_sq = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

        # Compute diagonal distance of enclosing box
        enclose_x1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
        enclose_y1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
        enclose_x2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
        enclose_y2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
        enclose_diagonal_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

        # DIoU = IoU - (center_distance^2 / enclose_diagonal^2)
        diou = iou - center_distance_sq / (enclose_diagonal_sq + 1e-6)
        return (1 - diou).mean()

    def _compute_ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute Complete IoU (CIoU) loss - considers IoU, center distance, and aspect ratio with stable numerics."""
        # Device/dtype
        device = pred_boxes.device if pred_boxes.is_cuda else (torch.device(self.device) if isinstance(self.device, str) else self.device)
        dtype = pred_boxes.dtype
        eps = torch.tensor(1e-7, device=device, dtype=dtype)

        pred_boxes = pred_boxes.to(device, dtype=dtype)
        target_boxes = target_boxes.to(device, dtype=dtype)

        # Convert to xyxy in normalized coords [0,1] (helper already clamps)
        pred_xyxy = self._xywh_to_xyxy(pred_boxes)
        target_xyxy = self._xywh_to_xyxy(target_boxes)

        # IoU components
        x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
        y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
        x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
        y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

        inter_w = (x2 - x1).clamp_min(0)
        inter_h = (y2 - y1).clamp_min(0)
        intersection = inter_w * inter_h

        pred_w = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp_min(eps)
        pred_h = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp_min(eps)
        target_w = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp_min(eps)
        target_h = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp_min(eps)

        area_pred = pred_w * pred_h
        area_tgt = target_w * target_h
        union = (area_pred + area_tgt - intersection).clamp_min(eps)
        iou = (intersection / union).clamp(0, 1)

        # Centers
        pred_cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) * 0.5
        pred_cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) * 0.5
        tgt_cx = (target_xyxy[:, 0] + target_xyxy[:, 2]) * 0.5
        tgt_cy = (target_xyxy[:, 1] + target_xyxy[:, 3]) * 0.5
        center_dist_sq = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

        # Enclosing diagonal squared
        enc_x1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
        enc_y1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
        enc_x2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
        enc_y2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
        enc_w = (enc_x2 - enc_x1).clamp_min(eps)
        enc_h = (enc_y2 - enc_y1).clamp_min(eps)
        enc_diag_sq = enc_w ** 2 + enc_h ** 2

        # Aspect ratio term v using atan2 for stability
        v = (4.0 / (torch.pi ** 2)) * (torch.atan2(target_w, target_h) - torch.atan2(pred_w, pred_h)) ** 2
        alpha = v / ((1.0 - iou).clamp_min(0) + v + eps)

        ciou = iou - (center_dist_sq / (enc_diag_sq + eps)) - alpha * v
        return (1.0 - ciou).mean()

    def _compute_classification_loss_from_one_hot(self, pred_cls: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
        """Compute classification loss from one-hot targets (used for ALL anchors including background)"""

        if self.use_focal_loss:
            return self._compute_focal_loss_from_one_hot(pred_cls, target_one_hot)
        else:
            # Apply label smoothing to one-hot targets
            if self.label_smoothing > 0 and pred_cls.shape[1] > 1:
                smooth_value = self.label_smoothing / (pred_cls.shape[1] - 1)
                target_one_hot = target_one_hot * (1.0 - self.label_smoothing) + smooth_value

            cls_loss = self.bce_loss(pred_cls, target_one_hot)
            return cls_loss

    def _compute_classification_loss(self, pred_cls: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
        """Compute classification loss with optional Focal Loss and Label Smoothing (from class indices)"""

        if self.use_focal_loss:
            return self._compute_focal_loss(pred_cls, target_cls)
        else:
            # Create one-hot targets with label smoothing
            # CRITICAL: Ensure target_cls is clamped before indexing (extra safety)
            num_classes = pred_cls.shape[1]
            target_cls = torch.clamp(target_cls, 0, num_classes - 1)

            target_one_hot = torch.zeros(pred_cls.shape, device=self.device)
            # Use torch.arange for proper device handling
            indices = torch.arange(len(target_cls), device=self.device)
            target_one_hot[indices, target_cls.long()] = 1.0

            # Apply label smoothing
            if self.label_smoothing > 0 and pred_cls.shape[1] > 1:
                smooth_value = self.label_smoothing / (pred_cls.shape[1] - 1)
                target_one_hot = target_one_hot * (1.0 - self.label_smoothing) + smooth_value

            cls_loss = self.bce_loss(pred_cls, target_one_hot)
            return cls_loss

    def _compute_focal_loss_from_one_hot(self, pred_cls: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss from one-hot targets (used for ALL anchors including background)"""

        # Apply label smoothing
        if self.label_smoothing > 0 and pred_cls.shape[1] > 1:
            smooth_value = self.label_smoothing / (pred_cls.shape[1] - 1)
            target_one_hot = target_one_hot * (1.0 - self.label_smoothing) + smooth_value

        # Get probabilities using sigmoid
        pred_probs = torch.sigmoid(pred_cls)

        # Compute cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred_cls, target_one_hot, reduction='none')

        # Compute focal term: (1 - pt)^gamma
        pt = torch.where(target_one_hot == 1, pred_probs, 1 - pred_probs)
        focal_weight = (1 - pt) ** self.focal_gamma

        # Apply alpha weight
        alpha_t = self.focal_alpha * target_one_hot + (1 - self.focal_alpha) * (1 - target_one_hot)

        # Apply class weights if provided
        if self.class_weights is not None:
            # Expand class weights to match target shape
            class_weights_expanded = self.class_weights.unsqueeze(0).expand_as(target_one_hot)
            focal_loss = alpha_t * focal_weight * ce_loss * class_weights_expanded
        else:
            focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()

    def _compute_focal_loss(self, pred_cls: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss for hard example emphasis with label smoothing and class weights (from class indices)"""

        # Create one-hot targets with label smoothing
        num_classes = pred_cls.shape[1]

        # CRITICAL: Ensure target_cls is clamped before indexing (extra safety)
        target_cls = torch.clamp(target_cls, 0, num_classes - 1)

        target_one_hot = torch.zeros(pred_cls.shape, device=self.device)
        # Use torch.arange for proper device handling
        indices = torch.arange(len(target_cls), device=self.device)
        target_one_hot[indices, target_cls.long()] = 1.0

        # Apply label smoothing
        if self.label_smoothing > 0 and num_classes > 1:
            smooth_value = self.label_smoothing / (num_classes - 1)
            target_one_hot = target_one_hot * (1.0 - self.label_smoothing) + smooth_value

        # Get probabilities using sigmoid
        pred_probs = torch.sigmoid(pred_cls)

        # Compute cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred_cls, target_one_hot, reduction='none')

        # Compute focal term: (1 - pt)^gamma
        pt = torch.where(target_one_hot == 1, pred_probs, 1 - pred_probs)
        focal_weight = (1 - pt) ** self.focal_gamma

        # Apply alpha weight
        alpha_t = self.focal_alpha * target_one_hot + (1 - self.focal_alpha) * (1 - target_one_hot)

        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss

        # Apply class weights if provided (for handling imbalanced data)
        if self.class_weights is not None:
            # Get class weights for each target class (vectorized)
            # target_cls: [num_samples]
            # class_weights: [num_classes]
            # Ensure target_cls is on the same device as class_weights
            target_cls_long = target_cls.long().to(self.class_weights.device)
            # Ensure indices are within valid range (already clamped, but double-check)
            target_cls_long = torch.clamp(target_cls_long, 0, len(self.class_weights) - 1)
            class_weight_per_sample = self.class_weights[target_cls_long]  # [num_samples]

            # Apply class weights: weight the entire loss tensor by the target class weight
            # The focal_loss is [num_samples, num_classes]
            # We want to weight each sample's loss by its target class weight
            # Vectorized approach: multiply each row by its corresponding class weight
            class_weight_expanded = class_weight_per_sample.unsqueeze(1)  # [num_samples, 1]
            weighted_focal_loss = focal_loss * class_weight_expanded

            focal_loss = weighted_focal_loss

        # Return mean over all classes and all samples
        return focal_loss.mean()

    def _extract_matched_anchor_info(self, pred: torch.Tensor, target: torch.Tensor,
                                     event_timestamps: torch.Tensor = None) -> List[Tuple[int, int]]:
        """
        Extract matched anchor indices and their corresponding track_ids from annotations.

        Args:
            pred: Predictions [num_anchors, features]
            target: Targets [num_targets, 8] = [class_id, x, y, w, h, conf, track_id, timestamp]
            event_timestamps: Optional event timestamps for temporal matching

        Returns:
            List of (anchor_idx, track_id) tuples for matched anchors
        """
        if len(target) == 0:
            return []

        num_anchors = pred.shape[0]
        num_targets = len(target)

        # Ensure target and pred are on the same device
        # Determine device - prefer CUDA if available, otherwise use self.device
        if pred.is_cuda:
            device = pred.device
        elif isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device

        # Move target to the same device as pred
        target = target.to(device)
        if event_timestamps is not None and isinstance(event_timestamps, torch.Tensor):
            event_timestamps = event_timestamps.to(device)

        # Extract target boxes and track_ids
        target_boxes = target[:, 1:5]  # [x, y, w, h] - in pixel coordinates
        target_track_id = target[:, 6].long()  # Track ID

        # Convert target boxes to normalized [0,1]
        target_boxes_normalized = target_boxes.clone()
        target_boxes_normalized[:, 0] = target_boxes[:, 0] / self.image_width
        target_boxes_normalized[:, 1] = target_boxes[:, 1] / self.image_height
        target_boxes_normalized[:, 2] = target_boxes[:, 2] / self.image_width
        target_boxes_normalized[:, 3] = target_boxes[:, 3] / self.image_height
        target_boxes_normalized = torch.clamp(target_boxes_normalized, 0.0, 1.0)

        # Predictions are already decoded & normalized by the head: [A, 4 + C]
        pred_boxes_normalized = pred[:, :4]

        # Match anchors to targets
        # decode preds to pred_boxes_normalized earlier (as you already do)
        C = self.num_classes
        pred_cls_logits = pred[:, -C:]  # [A,C]

        if event_timestamps is not None and len(event_timestamps) > 0:
            best_ious, best_indices = self._temporal_aware_matching(
                pred_boxes_normalized,
                target,
                target_boxes_normalized,
                event_timestamps,
                pred_cls=pred_cls_logits,
                time_window_frac=0.2
            )
        else:
            # Plain TAL (no temporal pre-filter)
            A = pred_boxes_normalized.shape[0]
            G = target_boxes_normalized.shape[0]
            pd_scores = pred_cls_logits.unsqueeze(0)
            pred_xyxy = self._xywh_to_xyxy(pred_boxes_normalized)
            pd_bboxes = pred_xyxy.unsqueeze(0)
            H, W = self._infer_hw_from_num_cells(A)
            anc_points = self._make_grid_points(H, W, pred_boxes_normalized.device,
                                                pred_boxes_normalized.dtype)
            gt_labels = target[:, 0].long().view(1, G, 1).to(pred_boxes_normalized.device)
            gt_bboxes_xyxy = self._xywh_to_xyxy(target_boxes_normalized).view(1, G, 4)
            mask_gt = torch.ones((1, G, 1), device=pred_boxes_normalized.device, dtype=torch.bool)
            _, _, _, fg_mask, target_gt_idx = self.tal_assigner(pd_scores, pd_bboxes, anc_points, gt_labels,
                                                                gt_bboxes_xyxy, mask_gt)
            ious_full = self._compute_iou(pred_boxes_normalized, target_boxes_normalized)
            best_indices = target_gt_idx.view(-1)
            best_ious = torch.zeros(A, device=pred_boxes_normalized.device, dtype=pred_boxes_normalized.dtype)
            if fg_mask.any():
                am = fg_mask.view(-1)
                best_ious[am] = ious_full[am, best_indices[am]]
        
        # Get matched anchors (IoU > threshold)
        matched_mask = best_ious > self.iou_threshold
        matched_indices = matched_mask.nonzero(as_tuple=True)[0]
        matched_target_indices = best_indices[matched_mask]
        matched_target_indices = torch.clamp(matched_target_indices, 0, num_targets - 1)
        
        # Extract track_ids for matched anchors
        matched_track_ids = target_track_id[matched_target_indices]
        
        # Return list of (anchor_idx, track_id) tuples
        matched_info = [
            (matched_indices[i].item(), matched_track_ids[i].item())
            for i in range(len(matched_indices))
        ]
        
        return matched_info
    
    def _compute_tracking_loss(self, track_features: torch.Tensor, 
                                matched_anchor_info: List[Tuple[int, int, int]] = None,
                                predictions: torch.Tensor = None) -> torch.Tensor:
        """
        Compute contrastive tracking loss using track_id from annotations.
        
        This loss uses ground truth track_id to create:
        1. Positive pairs: Same track_id → similar features (pull together)
        2. Negative pairs: Different track_id → dissimilar features (push apart)
        
        Args:
            track_features: Tracking features [T, B, H*W, feat_dim] or [B, H*W, feat_dim]
            matched_anchor_info: List of (batch_idx, anchor_idx, track_id) tuples for matched anchors
            predictions: Optional predictions for shape inference
        
        Returns:
            Tracking loss scalar
        """
        # Handle temporal-aware features: [T, B, H*W, features]
        has_temporal = track_features.dim() == 4
        if has_temporal:
            T, B, num_anchors, feat_dim = track_features.shape
            # Reshape to [T*B, H*W, features] for processing
            track_features_flat = track_features.view(T * B, num_anchors, feat_dim)
        else:
            # [B, H*W, features]
            B, num_anchors, feat_dim = track_features.shape
            T = 1
            track_features_flat = track_features
        
        # If no matched anchor info (no annotations or no matches), use simple regularization
        if matched_anchor_info is None or len(matched_anchor_info) == 0:
            # Simple L2 regularization to prevent features from exploding
            track_loss = track_features_flat.norm(dim=-1).mean() * 0.01
            # Set empty details for logging
            self._last_track_loss_details = {
                'total': float(track_loss.item() if isinstance(track_loss, torch.Tensor) else track_loss),
                'supcon_used': False,
                'num_samples': 0,
                'num_unique_labels': 0,
                'num_labels_with_ge2': 0,
                'temperature': float(self.supcon_temperature),
                'mean_feature_norm': float(track_features_flat.norm(dim=-1).mean().item()),
            }
            return track_loss
        
        # Normalize features to unit length for cosine similarity
        track_features_norm = F.normalize(track_features_flat, p=2, dim=-1)  # [T*B, H*W, feat_dim]
        
        # Extract features for matched anchors and group by track_id
        # matched_anchor_info: List of (batch_idx, anchor_idx, track_id)
        features_by_track_id = {}  # track_id -> List of feature vectors
        
        for batch_idx, anchor_idx, track_id in matched_anchor_info:
            # Get feature for this matched anchor
            if batch_idx < track_features_norm.shape[0] and anchor_idx < track_features_norm.shape[1]:
                feature = track_features_norm[batch_idx, anchor_idx]  # [feat_dim]
                
                if track_id not in features_by_track_id:
                    features_by_track_id[track_id] = []
                features_by_track_id[track_id].append(feature)
        
        # If no valid features, use simple regularization
        if len(features_by_track_id) == 0:
            track_loss = track_features_flat.norm(dim=-1).mean() * 0.01
            # Set empty details for logging
            self._last_track_loss_details = {
                'total': track_loss.item() if isinstance(track_loss, torch.Tensor) else track_loss,
                'positive': 0.0,
                'negative': 0.0,
                'num_positive_pairs': 0,
                'num_negative_pairs': 0
            }
            return track_loss

        # Compute supervised contrastive loss using all matched anchors
        # Gather (feature, label) pairs from matched anchors across (T*B, A)
        feat_list = []
        label_list = []
        for batch_idx, anchor_idx, track_id in matched_anchor_info:
            if batch_idx < track_features_norm.shape[0] and anchor_idx < track_features_norm.shape[1]:
                feat_list.append(track_features_norm[batch_idx, anchor_idx])  # [feat_dim]
                label_list.append(int(track_id))

        # Fallback regularization if we lack enough pairs (need at least 2 samples and at least one valid positive)
        if len(feat_list) < 2:
            track_loss = track_features_flat.norm(dim=-1).mean() * 0.01
            self._last_track_loss_details = {
                'total': float(track_loss.item() if isinstance(track_loss, torch.Tensor) else track_loss),
                'supcon_used': False,
                'num_samples': len(feat_list),
                'num_unique_labels': int(len(set(label_list))) if len(label_list) else 0,
                'num_labels_with_ge2': 0,
                'temperature': float(self.supcon_temperature),
                'mean_feature_norm': float(torch.stack(feat_list).norm(dim=-1).mean().item()) if len(
                    feat_list) else 0.0,
            }
            return track_loss

        # Also require that at least one label has >=2 occurrences (a real positive set)
        counts = {}
        for l in label_list:
            counts[l] = counts.get(l, 0) + 1
        has_positive_class = any(c >= 2 for c in counts.values())
        if not has_positive_class:
            track_loss = track_features_flat.norm(dim=-1).mean() * 0.01
            self._last_track_loss_details = {
                'total': float(track_loss.item() if isinstance(track_loss, torch.Tensor) else track_loss),
                'supcon_used': False,
                'num_samples': len(feat_list),
                'num_unique_labels': int(len(counts)),
                'num_labels_with_ge2': 0,
                'temperature': float(self.supcon_temperature),
                'mean_feature_norm': float(torch.stack(feat_list).norm(dim=-1).mean().item()) if len(
                    feat_list) else 0.0,
            }
            return track_loss

        feats = torch.stack(feat_list, dim=0)  # [N, D], already normalized
        labels = torch.tensor(label_list, device=feats.device, dtype=torch.long)  # [N]

        # Label stats for logging
        unique_labels_t, counts_t = labels.unique(return_counts=True)
        num_labels_ge2 = int((counts_t >= 2).sum().item())
        num_unique_labels = int(unique_labels_t.numel())

        # Use centralized temperature everywhere
        track_loss = self._supcon_loss(feats, labels, temperature=self.supcon_temperature)

        # Consistent, available logging fields
        self._last_track_loss_details = {
            'total': float(track_loss.detach().item()),
            'supcon_used': True,
            'num_samples': int(feats.size(0)),
            'num_unique_labels': num_unique_labels,
            'num_labels_with_ge2': num_labels_ge2,
            'temperature': float(self.supcon_temperature),
            'mean_feature_norm': float(feats.norm(dim=-1).mean().item()),
        }

        return track_loss

    @staticmethod
    def _supcon_loss(feats: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """
        Supervised Contrastive loss (SupCon) on L2-normalized feats.
        feats: [N, D] normalized
        labels: [N] long track_id (arbitrary ints)
        """
        device = feats.device
        sim = feats @ feats.t()  # cosine, since feats are normalized
        sim = sim / temperature

        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.t()).to(device)  # [N,N]
        mask_neg = ~mask_pos

        # Exclude self-contrast, improve stability
        logits = sim - torch.max(sim, dim=1, keepdim=True).values
        logits = logits - torch.diag(torch.diag(logits))  # zero self logits

        # Partition function over all other samples
        exp_logits = torch.exp(logits) * (mask_neg | mask_pos)  # self stays 0 on diag
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positives for each anchor (exclude self)
        eye = torch.eye(len(feats), device=device, dtype=torch.bool)
        pos_mask_no_self = mask_pos & ~eye
        pos_count = pos_mask_no_self.sum(dim=1)  # number of positives for each anchor

        # Anchors with no positives are ignored
        valid = pos_count > 0
        loss = torch.zeros(feats.size(0), device=device, dtype=feats.dtype)
        if valid.any():
            loss_pos = -(log_prob * pos_mask_no_self).sum(dim=1)
            loss[valid] = loss_pos[valid] / pos_count[valid].clamp_min(1)

        denom = valid.float().sum().clamp_min(1.0)
        return loss.sum() / denom

    def _temporal_aware_matching(
            self,
            pred_boxes_normalized: torch.Tensor,  # [A,4] xywh in [0,1]
            target: torch.Tensor,  # [G,8] [cls,x,y,w,h,conf,track_id,timestamp]
            target_boxes_normalized: torch.Tensor,  # [G,4] xywh in [0,1]
            event_timestamps: torch.Tensor,  # [Ne] timestamps for this sample/bin
            pred_cls: torch.Tensor,  # [A,C] class logits (required)
            time_window_frac: float = 0.2  # +/- fraction of total span to keep GTs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TAL-only temporal matching:
          1) Pick temporal bin time from event_timestamps.
          2) Keep GTs within +/- time_window_frac of that bin.
          3) Run TaskAlignedAssigner on the filtered GT set.
          4) Return (best_ious, best_indices) per anchor.
        """
        device = pred_boxes_normalized.device
        dtype = pred_boxes_normalized.dtype

        A = pred_boxes_normalized.shape[0]
        G = target_boxes_normalized.shape[0]
        assert G > 0, "TAL temporal matching requires at least one GT box"
        assert pred_cls is not None, "Provide class logits to TAL (pred_cls)"
        assert target.shape[1] >= 8, "Targets must include timestamps at [:,7]"

        # --- Temporal bin selection ---
        evt_ts = event_timestamps.to(device, dtype=dtype)
        valid = evt_ts >= 0
        assert valid.any(), "No valid event timestamps for temporal matching"
        evt_ts_valid = evt_ts[valid]
        t_min, t_max = evt_ts_valid.min(), evt_ts_valid.max()
        assert (t_max - t_min) > 0, "Degenerate time span for this sample"
        t_bin = evt_ts_valid.median()
        t_bin_n = (t_bin - t_min) / (t_max - t_min)

        gt_ts = target[:, 7].to(device, dtype=dtype)
        gt_ts_n = (gt_ts - t_min) / (t_max - t_min)

        # --- Pre-filter GTs by time window ---
        dt = (gt_ts_n - t_bin_n).abs()
        keep_mask = dt <= time_window_frac
        assert keep_mask.any(), "No GTs within temporal window; consider widening time_window_frac"

        tgt_keep_idx = keep_mask.nonzero(as_tuple=True)[0]  # [Gk]
        tgt_keep = target[tgt_keep_idx]
        tgt_boxes_keep = target_boxes_normalized[tgt_keep_idx]
        Gk = tgt_boxes_keep.shape[0]

        # --- TAL inputs ---
        # predictions
        pd_scores = pred_cls.unsqueeze(0)  # [1,A,C]
        pred_xyxy = self._xywh_to_xyxy(pred_boxes_normalized)
        pd_bboxes = pred_xyxy.unsqueeze(0)  # [1,A,4]

        # anchors (normalized grid centers)
        H, W = self._infer_hw_from_num_cells(A)
        anc_points = self._make_grid_points(H, W, device, dtype)  # [A,2]

        # ground truth (filtered)
        gt_labels_k = tgt_keep[:, 0].long().view(1, Gk, 1).to(device)
        gt_bboxes_xyxy_k = self._xywh_to_xyxy(tgt_boxes_keep).view(1, Gk, 4)
        mask_gt_k = torch.ones((1, Gk, 1), device=device, dtype=torch.bool)

        # --- Run TAL ---
        _, _, _, fg_mask, target_gt_idx_k = self.tal_assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels_k, gt_bboxes_xyxy_k, mask_gt_k
        )

        # --- Convert to (best_ious, best_indices) ---
        # Compute IoU against filtered GTs once
        ious_keep = self._compute_iou(pred_boxes_normalized, tgt_boxes_keep)  # [A,Gk]

        best_indices_k = target_gt_idx_k.view(-1)  # [A] in 0..Gk-1
        best_ious = torch.zeros(A, device=device, dtype=dtype)
        if fg_mask.any():
            am = fg_mask.view(-1)  # [A] bool
            best_ious[am] = ious_keep[am, best_indices_k[am]]

        # Map filtered GT indices back to original GT indices
        best_indices = torch.zeros(A, device=device, dtype=torch.long)
        best_indices[:] = tgt_keep_idx[best_indices_k.clamp_min(0).clamp_max(Gk - 1)]

        return best_ious, best_indices
