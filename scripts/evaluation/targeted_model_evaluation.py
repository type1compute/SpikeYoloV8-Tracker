#!/usr/bin/env python3
"""
Targeted Model Evaluation Script for SpikeYOLO Traffic Monitoring
Evaluates model on specific time windows that contain annotations.
"""
import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from torchvision.ops import nms
from PIL import Image, ImageDraw
import h5py
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import create_ultra_low_memory_dataloader
from src.etram_spikeyolo_tracking import eTraMSpikeYOLOWithTracking

# Import binary (eval) SpikeYOLO modules for inference
from ultralytics.nn.modules import yolo_spikformer_bin as Mbin
from src.config_loader import ConfigLoader
from src.logging_utils import setup_logging

# Set up logging - will be initialized in main() or when class is instantiated
logger = logging.getLogger(__name__)

class TargetedModelEvaluator:
    """Targeted model evaluation on annotated time windows."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda'):
        """Initialize the evaluator."""
        self.config = ConfigLoader(config_path)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None

        # Set up file logging for evaluation
        log_dir = self.config.get_logs_dir()
        log_file_name = f"evaluation_{Path(checkpoint_path).stem}.log"
        eval_logger, log_file = setup_logging(
            log_dir=log_dir,
            log_level="INFO",
            log_file_name=log_file_name,
            script_name="evaluation"
        )
        # Update module logger to use the new logger
        global logger
        logger = eval_logger
        logger.info(f"Evaluation logging initialized. Log file: {log_file}")

        # Class names for visualization (corrected to match eTraM dataset)
        # Get class names from config (handles both 3-class and 8-class)
        self.class_names = self.config.get_class_names()

    def load_model(self):
        """Load the trained model from checkpoint.

        The model uses FPN-style detection heads with upsampling and feature fusion:
        - P5: Starts from backbone output (23×40) → [B, 920, 64+nc]
        - P4: Upsamples P5 (2×) and fuses with backbone P4 features (45×80) → [B, 3600, 64+nc]
        - P3: Upsamples P4 (2×) and fuses with backbone P3 features (90×160) → [B, 14400, 64+nc]

        In eval mode, the model returns multi-scale predictions as a list of [B, H*W, 64+nc] tensors
        (temporal dimension already averaged). These are concatenated along the anchor dimension
        to get [B, total_anchors, 64+nc] for batch inference.
        """
        logger.info(f"Loading model from {self.checkpoint_path}")
        logger.info("Model architecture: FPN-style detection heads with upsampling and feature fusion")

        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            logger.info(f"Checkpoint loaded: epoch {checkpoint.get('epoch', 'unknown')}")

            # Initialize model with FPN-style architecture, using binary (eval) modules
            self.model = eTraMSpikeYOLOWithTracking(
                num_classes=self.config.get_num_classes(),
                input_size=self.config.get_input_size(),
                time_steps=self.config.get_time_steps(),
                track_feature_dim=self.config.get_track_feature_dim(),
                class_names=self.config.get_class_names(),
                mode="eval",  # ensure eval/binary path so inference uses binary MultiSpike4
                window_duration_us=self.config.get_window_us()

            )

            # Load model state with strict=False to handle missing BN stats and architecture changes
            # Note: Checkpoints saved with old architecture (direct P4/P3 heads) may have incompatible
            # detection_head_p4 and detection_head_p3 weights. These will be skipped and the new
            # FPN-style heads (p4_lateral, p4_fusion, p3_lateral, p3_fusion) will be initialized.
            if 'model_state_dict' in checkpoint:
                checkpoint_state = checkpoint['model_state_dict']
                model_state = self.model.state_dict()

                # Filter checkpoint state to only include compatible keys (handle architecture changes)
                filtered_state = {}
                incompatible_keys = []
                for key, value in checkpoint_state.items():
                    # Check if key exists in current model
                    if key in model_state:
                        # Check if shapes match
                        if model_state[key].shape == value.shape:
                            filtered_state[key] = value
                        else:
                            incompatible_keys.append(key)
                            logger.warning(f"Skipping incompatible key (shape mismatch): {key} - "
                                         f"checkpoint: {value.shape}, model: {model_state[key].shape}")
                    else:
                        # Key not in current model (architecture changed)
                        logger.warning(f"Skipping key not in current model: {key}")

                # Load filtered state
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    filtered_state, strict=False
                )
                logger.info(f"Model state loaded successfully (loaded {len(filtered_state)}/{len(checkpoint_state)} keys)")
                if incompatible_keys:
                    logger.warning(f"Incompatible keys (architecture changed): {len(incompatible_keys)}")
                    logger.warning(f"  Examples: {incompatible_keys[:5]}")
                if missing_keys:
                    logger.warning(f"Missing keys: {len(missing_keys)} (new parameters in current model)")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
            else:
                logger.error("No model_state_dict found in checkpoint")
                return False

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Ensure detect head keeps temporal axis at inference (per-timestep decoding)
            if hasattr(self.model, 'detect'):
                try:
                    self.model.detect.infer_temporal = True
                except Exception:
                    pass

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def find_annotated_time_windows(self, annotation_file: str, num_windows: int = 9999):
        """Find time windows that contain annotations - groups annotations into windows."""
        logger.info(f"Finding annotated time windows in {annotation_file}")

        try:
            # Load annotations
            annotations = np.load(annotation_file, allow_pickle=True)
            logger.info(f"Loaded {len(annotations)} annotations")

            if len(annotations) == 0:
                return []

            # Get timestamps
            times = annotations['t']
            unique_times = np.unique(times)

            # Group annotations into time windows (1 second each)
            window_size = self.config.get_window_us()
            annotated_windows = []

            for t in unique_times:
                # Find all annotations within 500ms of this timestamp
                mask = (times >= t - window_size // 2) & (times <= t + window_size // 2)
                annotation_indices = np.where(mask)[0]

                if len(annotation_indices) > 0:
                    start_time = times[annotation_indices].min() - window_size // 4
                    end_time = times[annotation_indices].max() + window_size // 4

                    annotated_windows.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'annotation_count': len(annotation_indices),
                        'annotations': annotations[mask]
                    })

            logger.info(f"Found {len(annotated_windows)} annotated time windows from {len(unique_times)} unique timestamps")
            return annotated_windows

        except Exception as e:
            logger.error(f"Error finding annotated time windows: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def load_events_from_time_window(self, h5_file: str, start_time: int, end_time: int):
        """Load events from a specific time window."""
        try:
            with h5py.File(h5_file, 'r') as f:
                # Check if events group exists
                if 'events' not in f:
                    logger.error(f"No 'events' group found in HDF5 file. Available keys: {list(f.keys())}")
                    return None

                events_group = f['events']
                logger.info(f"Events group keys: {list(events_group.keys())}")

                # Load individual event arrays
                x = events_group['x'][:]
                y = events_group['y'][:]
                t = events_group['t'][:]
                p = events_group['p'][:]

                logger.info(f"Loaded event arrays - x: {x.shape}, y: {y.shape}, t: {t.shape}, p: {p.shape}")

                # Filter events by time
                time_mask = (t >= start_time) & (t <= end_time)
                filtered_indices = np.where(time_mask)[0]

                if len(filtered_indices) == 0:
                    logger.warning(f"No events found in time window {start_time}-{end_time}")
                    return None

                # Create filtered event arrays
                filtered_x = x[filtered_indices]
                filtered_y = y[filtered_indices]
                filtered_t = t[filtered_indices]
                filtered_p = p[filtered_indices]

                # Combine into event array format [x, y, t, p]
                filtered_events = np.column_stack([filtered_x, filtered_y, filtered_t, filtered_p])

                # Limit events to prevent CUDA OOM (max 50K events)
                max_events = 50000
                if len(filtered_events) > max_events:
                    logger.info(f"Limiting events from {len(filtered_events)} to {max_events}")
                    filtered_events = filtered_events[:max_events]

                logger.info(f"Loaded {len(filtered_events)} events from time window {start_time}-{end_time}")
                return filtered_events

        except Exception as e:
            logger.error(f"Error loading events: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes with correct coordinate handling."""
        try:
            # FIX 3: Get image dimensions from config instead of hard-coding
            image_width = float(self.config.get('data_processing.image_width', 1280.0))
            image_height = float(self.config.get('data_processing.image_height', 720.0))

            # Debug: Log input boxes
            logger.debug(f"Box1: {box1}, Box2: {box2}")

            # Convert both boxes to [x1, y1, x2, y2] format
            # Both box1 (predictions) and box2 (ground truth) are in [x_center, y_center, width, height] format (PIXELS)

            # Box1: Convert from center format to corner format
            # Ensure we're working with float values
            if isinstance(box1, torch.Tensor):
                box1_x_center = float(box1[0].item())
                box1_y_center = float(box1[1].item())
                box1_width = float(box1[2].item())
                box1_height = float(box1[3].item())
            else:
                box1_x_center = float(box1[0])
                box1_y_center = float(box1[1])
                box1_width = float(box1[2])
                box1_height = float(box1[3])

            box1_x1 = box1_x_center - box1_width / 2.0
            box1_y1 = box1_y_center - box1_height / 2.0
            box1_x2 = box1_x_center + box1_width / 2.0
            box1_y2 = box1_y_center + box1_height / 2.0

            # Box2: Convert from center format to corner format
            # Ensure we're working with float values
            if isinstance(box2, torch.Tensor):
                box2_x_center = float(box2[0].item())
                box2_y_center = float(box2[1].item())
                box2_width = float(box2[2].item())
                box2_height = float(box2[3].item())
            else:
                box2_x_center = float(box2[0])
                box2_y_center = float(box2[1])
                box2_width = float(box2[2])
                box2_height = float(box2[3])

            box2_x1 = box2_x_center - box2_width / 2.0
            box2_y1 = box2_y_center - box2_height / 2.0
            box2_x2 = box2_x_center + box2_width / 2.0
            box2_y2 = box2_y_center + box2_height / 2.0

            # Clamp to valid pixel ranges from config
            box1_x1 = max(0.0, min(image_width, box1_x1))
            box1_y1 = max(0.0, min(image_height, box1_y1))
            box1_x2 = max(0.0, min(image_width, box1_x2))
            box1_y2 = max(0.0, min(image_height, box1_y2))

            box2_x1 = max(0.0, min(image_width, box2_x1))
            box2_y1 = max(0.0, min(image_height, box2_y1))
            box2_x2 = max(0.0, min(image_width, box2_x2))
            box2_y2 = max(0.0, min(image_height, box2_y2))

            # Debug: Log converted boxes
            logger.debug(f"Box1 (pred) converted: ({box1_x1:.3f}, {box1_y1:.3f}, {box1_x2:.3f}, {box1_y2:.3f})")
            logger.debug(f"Box2 (gt) converted: ({box2_x1:.3f}, {box2_y1:.3f}, {box2_x2:.3f}, {box2_y2:.3f})")

            # Calculate intersection
            x1 = max(box1_x1, box2_x1)
            y1 = max(box1_y1, box2_y1)
            x2 = min(box1_x2, box2_x2)
            y2 = min(box1_y2, box2_y2)

            if x2 <= x1 or y2 <= y1:
                logger.debug("No intersection")
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)

            # Calculate union
            area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
            area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
            union = area1 + area2 - intersection

            iou = intersection / union if union > 0 else 0.0

            # Debug logging for low IoU cases to help diagnose issues
            if iou < 0.1 and (abs(box1_x_center - box2_x_center) > 200 or abs(box1_y_center - box2_y_center) > 100):
                logger.debug(f"Low IoU ({iou:.4f}): Box1 center=({box1_x_center:.1f}, {box1_y_center:.1f}), size=({box1_width:.1f}, {box1_height:.1f}), "
                           f"Box2 center=({box2_x_center:.1f}, {box2_y_center:.1f}), size=({box2_width:.1f}, {box2_height:.1f}), "
                           f"Intersection={intersection:.2f}, Union={union:.2f}")

            logger.debug(f"IoU: {iou:.4f} (intersection: {intersection:.4f}, union: {union:.4f})")
            return iou

        except Exception as e:
            logger.error(f"Error calculating IoU: {e}")
            return 0.0

    def evaluate_annotated_windows(self):
        """Evaluate model using the same targeted training approach with dataloader.

        Uses evaluation.max_annotations_per_class from config to control evaluation size.
        """
        logger.info("Starting targeted evaluation with dataloader...")

        if not self.load_model():
            logger.error("Failed to load model")
            return None

        # Build test loader

        use_class_balanced_sampling = self.config.get('data_processing.use_class_balanced_sampling', False)
        min_samples_per_class = self.config.get('data_processing.min_samples_per_class', 1)

        training_max_annotations_per_class = self.config.get('data_processing.max_annotations_per_class', None)
        evaluation_ratio = self.config.get('evaluation.evaluation_annotation_ratio', 0.2)

        if training_max_annotations_per_class is not None:
            eval_max_annotations_per_class = int(training_max_annotations_per_class * evaluation_ratio)
            logger.info(
                f"Evaluation using max_annotations_per_class={eval_max_annotations_per_class} "
                f"({evaluation_ratio * 100:.0f}% of training limit={training_max_annotations_per_class})"
            )
        else:
            eval_max_annotations_per_class = None
            logger.info("Evaluation balancing to rarest class's count (training max_annotations_per_class not set)")

        samples_per_file = self.config.get('training.max_samples_per_file', 100)

        test_loader = create_ultra_low_memory_dataloader(
            data_root=self.config.get_data_root(),
            split='test',
            batch_size=self.config.get('training.batch_size', 20),
            max_events_per_sample=self.config.get('data_processing.max_events_per_sample', 10000),
            num_workers=self.config.get('data_processing.num_workers', 10),
            shuffle=False,
            annotation_dir=self.config.get_annotation_dir(),
            max_samples_per_file=samples_per_file if eval_max_annotations_per_class is None else None,
            targeted_training=True,
            num_classes=self.config.get_num_classes(),
            use_class_balanced_sampling=use_class_balanced_sampling,
            min_samples_per_class=min_samples_per_class,
            max_annotations_per_class=eval_max_annotations_per_class,
            time_steps=self.config.get_time_steps(),
            image_height=self.config.get('data_processing.image_height', 720),
            image_width=self.config.get('data_processing.image_width', 1280),
            config=self.config,  # Pass config for DataLoader parameters
            time_window_us=self.config.get_window_us()
        )

        logger.info(f"Created test dataloader with {len(test_loader)} batches")

        # Paths / constants
        results_dir = self.config.get_results_dir()
        vis_dir = Path(results_dir) / 'eval_vis'
        vis_dir.mkdir(parents=True, exist_ok=True)

        image_height = int(self.config.get('data_processing.image_height', 720))
        image_width = int(self.config.get('data_processing.image_width', 1280))
        conf_thres = float(self.config.get('evaluation.confidence_threshold', 0.2))
        iou_thres = float(self.config.get('evaluation.iou_threshold', 0.4))
        nc = self.config.get_num_classes()

        logger.info("Starting evaluation loop...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                try:
                    # Raw events [B, N, 4] are kept on CPU for timestamps / debugging only
                    events = batch['events']  # [B, N, 4]
                    frames = batch.get('frames', None)  # [B, T, H, W] signed spike frames
                    targets = batch['targets'].to(self.device)  # [B, N, 8]
                    event_timestamps = batch.get('event_timestamps', None)

                    if frames is None:
                        logger.error("Batch is missing 'frames' tensor required for model input; skipping batch.")
                        continue

                    frames = frames.to(self.device)

                    logger.info(f"Batch {batch_idx}: forward on frames {tuple(frames.shape)}")
                    outputs = self.model(frames)

                    # Accept either:
                    #   A) (decoded_predictions, raw_features): decoded per-scale temporal outputs
                    #   B) tracked_seq OR (tracked_seq, track_features): list length T with per-timestep tracked dets
                    raw_features = None
                    decoded_predictions = None

                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        decoded_predictions, raw_features = outputs
                    else:
                        decoded_predictions = outputs

                    if not isinstance(decoded_predictions, list) or len(decoded_predictions) == 0:
                        logger.error("Model output not in expected list/tuple format; skipping batch.")
                        continue

                    first_elem = decoded_predictions[0]

                    # Case A: per-scale temporal decodes → list per scale of [T,B,A_i,4+nc]
                    if isinstance(first_elem, torch.Tensor) and first_elem.dim() == 4:
                        T, B, _, _ = first_elem.shape
                        if B != events.shape[0]:
                            logger.warning(
                                f"Predictions batch size {B} differs from events batch size {events.shape[0]}")

                        # Concatenate scales along anchors per timestep -> [T,B,A_total,4+nc]
                        temporal_predictions = []
                        for t in range(T):
                            per_scale_t = [p[t] for p in decoded_predictions]  # each [B, A_i, 4+nc]
                            temporal_predictions.append(torch.cat(per_scale_t, dim=1))
                        predictions = torch.stack(temporal_predictions, dim=0)  # [T,B,A_total,4+nc]

                        is_tracked_sequence = False

                    # Case B: per-timestep tracked sequence → list length T of [N_t, 4+nc+2]
                    elif isinstance(first_elem, torch.Tensor) and first_elem.dim() == 2:
                        tracked_seq = decoded_predictions
                        T = len(tracked_seq)
                        B = events.shape[0]
                        if B != 1:
                            logger.warning("Tracked-sequence path currently assumes B==1; got B=%d", B)
                        predictions = tracked_seq  # list of tensors [N_t, 4+nc+2] (xywh_px + class_probs + conf + track_id)
                        is_tracked_sequence = True

                    else:
                        logger.error(
                            f"Unsupported model output format: first element type={type(first_elem)}, shape={getattr(first_elem, 'shape', None)}")
                        continue

                    for i in range(events.shape[0]):
                        # Valid annotated targets for sample i (remove zero-padded rows)
                        target_tensor = targets[i]
                        valid_mask = target_tensor.sum(dim=1) != 0
                        valid_targets = target_tensor[valid_mask] if valid_mask.any() else None

                        # Per-sample timestamps
                        sample_ts = None
                        if event_timestamps is not None:
                            if isinstance(event_timestamps, torch.Tensor) and event_timestamps.dim() >= 2 and i < \
                                    event_timestamps.shape[0]:
                                sample_ts = event_timestamps[i]
                            elif isinstance(event_timestamps, list) and i < len(event_timestamps):
                                sample_ts = event_timestamps[i]

                        # Temporal bin edges
                        time_bins = None
                        if sample_ts is not None and len(sample_ts) > 0:
                            if not isinstance(sample_ts, torch.Tensor):
                                sample_ts = torch.tensor(sample_ts, device=events.device)
                            valid_ts = sample_ts[sample_ts >= 0]
                            if len(valid_ts) > 0:
                                t_min = float(valid_ts.min());
                                t_max = float(valid_ts.max())
                                time_bins = torch.linspace(t_min, t_max, T + 1, device=events.device)

                        # Evaluate each timestep
                        for t_idx in range(T):
                            if not is_tracked_sequence:
                                # ---- Case A: predictions is [T,B,A,4+nc] ----
                                preds_t = predictions[t_idx, i:i + 1]  # [1,A,4+nc]
                                boxes_n = preds_t[0, :, :4]           # [A,4] normalized xywh
                                cls_logits = preds_t[0, :, 4:4 + nc]  # [A,nc]

                                cls_probs = torch.sigmoid(cls_logits)
                                conf_per_anchor, cls_ids = cls_probs.max(dim=1)
                                keep = conf_per_anchor >= conf_thres
                                if not keep.any():
                                    continue

                                boxes_n = boxes_n[keep]
                                cls_ids = cls_ids[keep]
                                scores = conf_per_anchor[keep]

                                # Normalize -> pixel xyxy
                                scale_vec = torch.tensor([image_width, image_height, image_width, image_height],
                                                         device=boxes_n.device, dtype=boxes_n.dtype)
                                xywh_px = boxes_n * scale_vec
                                x1 = xywh_px[:, 0] - xywh_px[:, 2] / 2
                                y1 = xywh_px[:, 1] - xywh_px[:, 3] / 2
                                x2 = xywh_px[:, 0] + xywh_px[:, 2] / 2
                                y2 = xywh_px[:, 1] + xywh_px[:, 3] / 2
                                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

                                keep_idx = nms(boxes_xyxy, scores, iou_thres)
                                if keep_idx.numel() == 0:
                                    continue

                                boxes_xyxy = boxes_xyxy[keep_idx]
                                scores = scores[keep_idx]
                                cls_ids = cls_ids[keep_idx]

                                # Final detections [N, 4+nc+1] in pixels (xywh + class probs + score)
                                w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp_min(0)
                                h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp_min(0)
                                cx = boxes_xyxy[:, 0] + 0.5 * w
                                cy = boxes_xyxy[:, 1] + 0.5 * h
                                xywh_px_kept = torch.stack([cx, cy, w, h], dim=1)

                                class_probs_full = torch.zeros((xywh_px_kept.shape[0], nc),
                                                               device=xywh_px_kept.device, dtype=xywh_px_kept.dtype)
                                rows = torch.arange(xywh_px_kept.shape[0], device=xywh_px_kept.device)
                                class_probs_full[rows, cls_ids] = scores
                                dets_out = torch.cat([xywh_px_kept, class_probs_full, scores.unsqueeze(1)], dim=1)

                                # Optional feature alignment for tracker (if raw_features returned)
                                tracked = dets_out
                                if isinstance(raw_features, list) and len(raw_features) == len(decoded_predictions):
                                    feats_t = [feats[t_idx, i] for feats in raw_features]  # each [A_i,D]
                                    feats_cat = torch.cat(feats_t, dim=0)                  # [A_total, D]
                                    feats_kept = feats_cat[keep][keep_idx]
                                    # Run tracking via model helper (preferred) or directly on tracker
                                    try:
                                        if hasattr(self.model, "apply_tracking"):
                                            tracked = self.model.apply_tracking(
                                                dets_out, feats_kept, (int(image_height), int(image_width))
                                            )
                                        elif hasattr(self.model, "tracker"):
                                            tracked = self.model.tracker.update(
                                                dets_out, feats_kept, (int(image_height), int(image_width))
                                            )
                                        else:
                                            tracked = dets_out
                                    except Exception as e:
                                        logger.warning(f"Tracking failed at t={t_idx}: {e}")
                            else:
                                # ---- Case B: tracked sequence: predictions[t_idx] is [N_t, 4+nc+2] ----
                                tracked = predictions[t_idx]
                                if tracked.numel() == 0:
                                    continue

                                # Parse fields for metrics/visualization
                                xywh_px_kept = tracked[:, :4]  # pixel xywh
                                class_probs_full = tracked[:, 4:4 + nc]  # [N_t, nc]
                                scores = tracked[:, 4 + nc]  # [N_t]
                                _, cls_ids = class_probs_full.max(dim=1)  # for coloring/metrics, if needed

                            # Visualization dump (optional)
                                try:
                                    frame_t = frames[i, t_idx]  # [H,W]
                                    frame_np = frame_t.detach().float().cpu().numpy()
                                    fmin, fmax = float(frame_np.min()), float(frame_np.max())
                                    if fmax > fmin:
                                        frame_np = (255.0 * (frame_np - fmin) / (fmax - fmin)).astype('uint8')
                                    else:
                                        frame_np = np.zeros_like(frame_np, dtype='uint8')

                                    img = Image.fromarray(frame_np)
                                    if img.size != (image_width, image_height):
                                        img = img.resize((image_width, image_height), Image.NEAREST)
                                    draw = ImageDraw.Draw(img)

                                    # Draw GT (green) for this timestep if available
                                    if valid_targets is not None and time_bins is not None:
                                        bin_start = time_bins[t_idx]
                                        bin_end = time_bins[t_idx + 1]
                                        ann_ts = valid_targets[:, 7]
                                        # Half-open [start, end) for all but the last bin; last bin is [start, end]
                                        if t_idx < T - 1:
                                            bm = (ann_ts >= bin_start) & (ann_ts < bin_end)
                                        else:
                                            bm = (ann_ts >= bin_start) & (ann_ts <= bin_end)
                                        if bm.any():
                                            gts = valid_targets[bm]
                                            for gt in gts:
                                                gx, gy, gw, gh = float(gt[1]), float(gt[2]), float(gt[3]), float(gt[4])
                                                gx1, gy1 = gx - gw / 2.0, gy - gh / 2.0
                                                gx2, gy2 = gx + gw / 2.0, gy + gh / 2.0
                                                draw.rectangle([gx1, gy1, gx2, gy2], outline=(0, 255, 0), width=2)

                                    # Draw predictions (red)
                                    bx = tracked[:, 0:4]
                                    dx1 = bx[:, 0] - bx[:, 2] / 2.0
                                    dy1 = bx[:, 1] - bx[:, 3] / 2.0
                                    dx2 = bx[:, 0] + bx[:, 2] / 2.0
                                    dy2 = bx[:, 1] + bx[:, 3] / 2.0
                                    for k in range(bx.shape[0]):
                                        draw.rectangle([float(dx1[k]), float(dy1[k]), float(dx2[k]), float(dy2[k])],
                                                       outline=(255, 0, 0), width=2)

                                    # File id (best effort)
                                    file_id = 'unk'
                                    try:
                                        for kkey in ('file_id', 'file_idx', 'filename', 'file_path', 'paths'):
                                            if kkey in batch:
                                                val = batch[kkey]
                                                if isinstance(val, (list, tuple)):
                                                    val = val[i]
                                                if isinstance(val, torch.Tensor):
                                                    val = val.item() if val.numel() == 1 else str(val)
                                                file_id = str(val)
                                                if '/' in file_id or '\\' in file_id:
                                                    file_id = file_id.replace('\\', '/').split('/')[-1]
                                                break
                                    except Exception:
                                        pass

                                    if time_bins is not None:
                                        tspan = f"{int(time_bins[t_idx].item())}-{int(time_bins[t_idx + 1].item())}us"
                                    else:
                                        tspan = "na"

                                    out_name = f"vis_b{batch_idx:03d}_f{file_id}_t{t_idx:02d}_{tspan}.png"
                                    out_path = vis_dir / out_name
                                    img.save(out_path)
                                    logger.info(f"Saved visualization: {out_path}")
                                except Exception as viz_e:
                                    logger.warning(
                                        f"Visualization save failed for batch {batch_idx}, sample {i}, t {t_idx}: {viz_e}")

                except Exception as e:
                    logger.error(f"Unhandled error while processing batch {batch_idx}: {e}")
                    logger.error(traceback.format_exc())
                    continue