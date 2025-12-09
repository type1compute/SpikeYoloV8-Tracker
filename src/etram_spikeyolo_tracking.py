"""
eTraM SpikeYOLO with Tracking – model, heads, and temporal decode

Quick shape map:
  Input (training):               [B, T, H, W] or [B, T, 2, H, W] (ON/OFF)
  Backbone/heads internal:        [T, B, C, H, W]
  Train head outputs (per scale): [T, B, (4*reg_max + C), H, W]
  Track feature taps (per scale): [B, D, H, W] after temporal mean
  Eval temporal decode output:    per scale list of [T, B, A, 4+C]

Notes:
  • Multi‑scale is represented as a Python list: [P5, P4, P3], not a tensor, because H×W differ.
  • Training returns raw logits for loss (no decoding). Evaluation can decode per‑timestep for tracking.
"""
#!/usr/bin/env python3
"""
SpikeYOLO Tracking Extension Module

This module adds object tracking capability to the BICLab SpikeYOLO architecture
to accommodate the track_id field in the eTraM annotation dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union, List
import logging
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils.tal import dist2bbox, make_anchors


# Helper to select module implementation (train/integer or eval/binary)
def _select_spike_module_impl(arch: str):
    """
    Return the SpikeYOLO module namespace according to arch:
    - 'train' -> yolo_spikformer (integer multi-spike training graph)
    - 'eval'  -> yolo_spikformer_bin (binary inference graph)
    """
    arch = (arch or "train").lower()
    if arch == "eval":
        from ultralytics.nn.modules import yolo_spikformer_bin as M
    else:
        from ultralytics.nn.modules import yolo_spikformer as M
    return M


logger = logging.getLogger(__name__)

class SpikeYOLOTracker:
    """
    Post-processing tracker for SpikeYOLO detections.

    This class wraps the BYTETracker to add track_id to SpikeYOLO detections,
    making them compatible with the eTraM annotation format.
    """

    def __init__(self,
                 nc: int = 3,
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 frame_rate: float = None):
        """
        Initialize SpikeYOLO tracker.

        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for track matching
            frame_rate: Frame rate for tracking (10K FPS for event cameras)
            Since we are aggregating events within each timestep, the frame rate would change depending on config.
        """
        from argparse import Namespace
        tracker_args = Namespace(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh
        )
        self.tracker = BYTETracker(
            args=tracker_args,
            frame_rate=frame_rate
        )


        # Store match threshold persistently for later use
        self.match_thresh = match_thresh
        self.nc = nc
        self.frame_count = 0

        # Storage for tracking features per track
        self.track_features = {}  # track_id -> feature_vector (128-dim)
        self.next_track_id = 1  # Counter for new track IDs
        self.use_feature_matching = True  # Enable feature-based matching
        self.feature_weight = 0.3  # Weight for feature similarity (0.7 for IoU, 0.3 for features)
        self.feature_similarity_threshold = 0.5  # Minimum cosine similarity for feature matching

        logger.info(f"SpikeYOLOTracker initialized:")
        logger.info(f"  Track threshold: {track_thresh}")
        logger.info(f"  Track buffer: {track_buffer}")
        logger.info(f"  Match threshold: {match_thresh}")
        logger.info(f"  Frame rate: {frame_rate} FPS")
        logger.info(f"  Feature-based matching: {self.use_feature_matching}")
        logger.info(f"  Feature weight: {self.feature_weight}, IoU weight: {1.0 - self.feature_weight}")
        logger.info(f"  Num classes (dynamic): {self.nc}")

    def update(self, detections: torch.Tensor, track_features: torch.Tensor = None,
               image_shape: Tuple[int, int] = (720, 1280)) -> torch.Tensor:
        """
        Update tracker with new detections and return tracked objects.

        Args:
            detections: Detection tensor [N, 4 + C + 1] = [x, y, w, h, class_0, ..., class_{C-1}, conf]
            track_features: Optional tracking features [N, 128] for feature-based matching
            image_shape: Image dimensions (height, width)

        Returns:
            Tracked detections tensor [N, 4 + C + 2] = [x, y, w, h, class_0, ..., class_{C-1}, conf, track_id]
        """
        if detections.numel() == 0:
            return torch.zeros((0, 4 + self.nc + 2), device=detections.device)

        # If tracking features are provided and feature matching is enabled, use enhanced matching
        if track_features is not None and self.use_feature_matching:
            if len(self.track_features) > 0:
                # Use feature-based matching with existing tracks
                tracked_detections = self._update_with_features(detections, track_features, image_shape)
            else:
                # First frame: use standard tracker, then store features
                dets = self._convert_to_bytetracker_format(detections, image_shape)  # numpy Nx6
                tracks = self._bytetracker_update(dets, image_shape)
                tracked_detections = self._convert_bytetracker_tracks_to_tensor(tracks, detections.device)
                # Store tracking features for next frame
                self._store_track_features(tracked_detections, track_features)
            self.frame_count += 1
            return tracked_detections
        else:
            # Use standard tracker (IoU-based only)
            dets = self._convert_to_bytetracker_format(detections, image_shape)  # numpy Nx6
            tracks = self._bytetracker_update(dets, image_shape)
            tracked_detections = self._convert_bytetracker_tracks_to_tensor(tracks, detections.device)
            # Store tracking features for matched tracks (if provided)
            if track_features is not None:
                self._store_track_features(tracked_detections, track_features)
            self.frame_count += 1
            return tracked_detections


    def _bytetracker_update(self, dets: np.ndarray, img=None):
        """
        Call BYTETracker.update with detections in tlbr+score+cls numpy format.
        dets: numpy [N,6] = [x1, y1, x2, y2, score, cls]
        Returns the tracker-native tracks (typically a list of STrack) or an empty list.
        """
        if dets is None or len(dets) == 0:
            return []
        if not hasattr(self, 'tracker') or self.tracker is None:
            return []
        return self.tracker.update(dets, img)

    def _convert_bytetracker_tracks_to_tensor(self, tracks, device: torch.device) -> torch.Tensor:
        """Convert ByteTrack outputs (list of STrack or numpy Nx8) to our tensor format:
        [xywh, class_probs, conf, track_id] with C = self.nc.
        """
        C = self.nc
        if tracks is None:
            return torch.zeros((0, 4 + C + 2), device=device)

        # Case 1: numpy array Nx8 [x1,y1,x2,y2,track_id,score,cls,idx]
        if isinstance(tracks, np.ndarray):
            if tracks.size == 0:
                return torch.zeros((0, 4 + C + 2), device=device)
            tracked = torch.from_numpy(tracks).to(device=device, dtype=torch.float32)
            tlbr = tracked[:, :4]
            track_id = tracked[:, 4:5]
            score = tracked[:, 5:6]
            cls_from_bt = tracked[:, 6].long()
        else:
            # Case 2: list of STrack-like objects
            rows = []
            for t in tracks:
                if hasattr(t, 'tlbr'):
                    x1, y1, x2, y2 = map(float, t.tlbr)
                elif hasattr(t, 'tlwh'):
                    x, y, w, h = map(float, t.tlwh)
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:
                    continue
                conf = float(getattr(t, 'score', 1.0))
                cls_id = int(getattr(t, 'cls', 0))
                track_id_val = int(getattr(t, 'track_id', -1))
                rows.append([x1, y1, x2, y2, track_id_val, conf, cls_id])
            if len(rows) == 0:
                return torch.zeros((0, 4 + C + 2), device=device)
            tracked = torch.tensor(rows, device=device, dtype=torch.float32)
            tlbr = tracked[:, :4]
            track_id = tracked[:, 4:5]
            score = tracked[:, 5:6]
            cls_from_bt = tracked[:, 6].long()

        # Convert TLBR -> XYWH
        xywh_out = self._xyxy_to_xywh(tlbr)

        # Build class probabilities tensor
        class_probs_out = torch.zeros((xywh_out.shape[0], C), device=device, dtype=torch.float32)
        ok = (cls_from_bt >= 0) & (cls_from_bt < C)
        if ok.any():
            rows = torch.arange(xywh_out.shape[0], device=device)[ok]
            class_probs_out[rows, cls_from_bt[ok]] = score[ok, 0]

        # Output: [xywh, class_probs, conf, track_id]
        out = torch.cat([xywh_out, class_probs_out, score, track_id], dim=1)
        return out


    def _update_with_features(self, detections: torch.Tensor, track_features: torch.Tensor,
                               image_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Update tracker using combined IoU and feature similarity matching.

        Args:
            detections: Detection tensor [N, 4 + C + 1] = [x, y, w, h, class_0, ..., class_{C-1}, conf]
            track_features: Tracking features [N, 128] for each detection
            image_shape: Image dimensions (height, width)

        Returns:
            Tracked detections tensor [N, 4 + C + 2] = [x, y, w, h, class_0, ..., class_{C-1}, conf, track_id]
        """
        device = detections.device
        num_detections = len(detections)

        # Normalize tracking features for cosine similarity
        track_features_norm = F.normalize(track_features, p=2, dim=1)  # [N, 128]

        # Extract boxes and convert to xyxy format for IoU computation
        boxes = detections[:, :4]  # [x, y, w, h]
        boxes_xyxy = self._xywh_to_xyxy(boxes)  # [x1, y1, x2, y2]

        # Initialize track IDs
        track_ids = torch.zeros(num_detections, dtype=torch.long, device=device)
        matched_track_ids = set()

        # Match each detection with existing tracks
        for det_idx in range(num_detections):
            best_match_id = None
            best_combined_score = 0.0

            det_box = boxes_xyxy[det_idx]  # [x1, y1, x2, y2]
            det_feature = track_features_norm[det_idx]  # [128]

            # Try to match with existing tracks
            for track_id, track_data in self.track_features.items():
                if track_id in matched_track_ids:
                    continue  # Track already matched

                track_box = track_data['box']  # [x1, y1, x2, y2]
                track_feature = track_data['feature']  # [128]

                # Compute IoU
                iou = self._compute_iou_xyxy(det_box, track_box)

                # Compute feature similarity (cosine similarity)
                feature_sim = torch.dot(det_feature, track_feature).item()  # Cosine similarity

                # Combined score: weighted combination of IoU and feature similarity
                iou_weight = 1.0 - self.feature_weight
                combined_score = iou_weight * iou + self.feature_weight * feature_sim

                # Only consider matches above threshold
                if combined_score > self.match_thresh and combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_match_id = track_id

            # Assign track ID
            if best_match_id is not None:
                track_ids[det_idx] = best_match_id
                matched_track_ids.add(best_match_id)
                # Update track feature (exponential moving average)
                old_feature = self.track_features[best_match_id]['feature']
                alpha = 0.9  # Smoothing factor
                new_feature = alpha * old_feature + (1 - alpha) * track_features_norm[det_idx]
                self.track_features[best_match_id]['feature'] = F.normalize(new_feature, p=2, dim=0)
                self.track_features[best_match_id]['box'] = det_box
            else:
                # Create new track
                new_track_id = self.next_track_id
                self.next_track_id += 1
                track_ids[det_idx] = new_track_id
                self.track_features[new_track_id] = {
                    'feature': track_features_norm[det_idx].clone(),
                    'box': det_box.clone()
                }

        # Combine detections with track IDs
        tracked_detections = torch.cat([detections, track_ids.unsqueeze(1).float()], dim=1)

        return tracked_detections

    def _store_track_features(self, tracked_detections: torch.Tensor, track_features: torch.Tensor):
        """
        Store tracking features for each track after standard matching.

        Args:
            tracked_detections: Tracked detections [N, 13] with track_id at last column
            track_features: Tracking features [N, 128]
        """
        if len(tracked_detections) == 0:
            return

        # Normalize features
        track_features_norm = F.normalize(track_features, p=2, dim=1)

        # Extract boxes and track IDs
        boxes = tracked_detections[:, :4]  # [x, y, w, h]
        boxes_xyxy = self._xywh_to_xyxy(boxes)
        track_ids = tracked_detections[:, -1].long()  # track_id

        # Store or update features for each track
        for idx, track_id in enumerate(track_ids):
            track_id_item = track_id.item()
            if track_id_item in self.track_features:
                # Update existing track (exponential moving average)
                old_feature = self.track_features[track_id_item]['feature']
                alpha = 0.9
                new_feature = alpha * old_feature + (1 - alpha) * track_features_norm[idx]
                self.track_features[track_id_item]['feature'] = F.normalize(new_feature, p=2, dim=0)
                self.track_features[track_id_item]['box'] = boxes_xyxy[idx]
            else:
                # Create new track
                self.track_features[track_id_item] = {
                    'feature': track_features_norm[idx].clone(),
                    'box': boxes_xyxy[idx].clone()
                }
                # Update next_track_id to avoid conflicts
                if track_id_item >= self.next_track_id:
                    self.next_track_id = track_id_item + 1

    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format."""
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    def _compute_iou_xyxy(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two boxes in xyxy format."""
        # Ensure boxes are 1D tensors
        if box1.dim() > 1:
            box1 = box1.squeeze()
        if box2.dim() > 1:
            box2 = box2.squeeze()

        # Calculate intersection
        x1_inter = torch.max(box1[0], box2[0])
        y1_inter = torch.max(box1[1], box2[1])
        x2_inter = torch.min(box1[2], box2[2])
        y2_inter = torch.min(box1[3], box2[3])

        # Check if there's an intersection
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return (inter_area / union_area).item()

    def _convert_to_bytetracker_format(self, detections: torch.Tensor, image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert SpikeYOLO detections to BYTETracker format."""
        if detections.numel() == 0:
            return np.empty((0, 6), dtype=np.float32)

        # Extract bounding boxes and confidence
        boxes = detections[:, :4].clone()  # [x, y, w, h]
        confs = detections[:, -1].clone()  # confidence
        class_probs = detections[:, 4:-1]
        class_ids = torch.argmax(class_probs, dim=1)

        # If boxes look normalized (<= 1.5), scale to pixels
        H, W = image_shape
        if torch.max(boxes) <= 1.5:
            scale = torch.tensor([W, H, W, H], device=boxes.device, dtype=boxes.dtype)
            boxes = boxes * scale

        # Convert to [x1, y1, x2, y2]
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        dets = torch.stack([x1, y1, x2, y2, confs, class_ids.float()], dim=1)
        return dets.detach().cpu().numpy().astype(np.float32)


    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()
        self.frame_count = 0
        self.track_features.clear()  # Clear stored tracking features
        self.next_track_id = 1  # Reset track ID counter

    def _xyxy_to_xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert [x1, y1, x2, y2] -> [x, y, w, h]."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = (x2 - x1).clamp_min(0)
        h = (y2 - y1).clamp_min(0)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return torch.stack([cx, cy, w, h], dim=1)


class SpikeDetectTrainWithTracking(nn.Module):
    """
      Training‑time SpikeDetect head with temporal dimension and tracking taps.

      Inputs:
        x: list of per‑scale features, each [T, B, C_i, H_i, W_i]
      Returns (training mode):
        (z, track_feats)
          • z: list per scale of raw logits [T, B, (4*reg_max + C), H, W]
          • track_feats: list per scale of [B, D, H, W] feature taps (temporal mean over T, then 1×1 proj)

      Example:
        >>> z, track = head([p5, p4, p3])
        >>> z[0].shape  # [T,B,4*reg_max + C, H5, W5]
        >>> track[0].shape  # [B, D, H5, W5]
    """
    dynamic = False
    export = False
    shape = None

    def __init__(self, nc: int, ch: Tuple[int, int, int], track_feature_dim: int,
                 module_impl, reg_max: int = 16, infer_temporal: bool = False):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.infer_temporal = infer_temporal

        # modules (train graph)
        M = module_impl
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                M.MS_StandardConv(x, c2, 3),
                M.MS_StandardConv(c2, c2, 3),
                M.MS_StandardConv(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                M.MS_StandardConv(x, c3, 3),
                M.MS_StandardConv(c3, c3, 3),
                M.MS_StandardConv(c3, self.nc, 1)
            ) for x in ch
        )

        self.dfl = M.SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.concat1 = Concat(2)
        self.concat2 = Concat(2)
        self.concat3 = Concat(1)

        # tracking feature taps (keep simple: 1×1 projection on each scale)
        self.track_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(ch[i] if isinstance(ch[i], int) else int(ch[i]), track_feature_dim, 1, bias=False),
                nn.BatchNorm2d(track_feature_dim)
            ) for i in range(self.nl)
        )

    def forward(self, x: List[torch.Tensor]):
        """
        x: list of [T,B,C,H,W] per scale (P5,P4,P3)
        returns:
          - training: raw per-scale logits (kept as list)
          - eval path in this module is used only for temporal decode (training still returns lists)
        """
        # track features are taken from *inputs* (before splits) for stability
        track_feats = [t.mean(0) for t in x]  # [B,C,H,W] per scale
        track_feats = [self.track_proj[i](t) for i, t in enumerate(track_feats)]  # [B,128,H,W] per scale

        # standard detect head compute
        z = [None] * self.nl
        y_shapes = [xi.mean(0) for xi in x]  # BCHW
        for i in range(self.nl):
            z[i] = self.concat1((self.cv2[i](x[i]), self.cv3[i](x[i])))  # [T,B,(4*reg+C),H,W]
            z[i] = z[i]  # keep T in training

        # Training path: return raw per‑scale logits (no decode) so the loss can:
        #   1) reshape [T,B,C,H,W] → [T,B,H*W,C]
        #   2) run TAL/DFL/IoU on anchors per scale
        # Track features are per‑scale [B,D,H,W] taps to support optional feature losses and runtime tracking.
        if self.training:
            return z, track_feats

        # ------- (eval path on train head, for completeness) ----------
        # anchors/strides build
        if (self.dynamic or self.shape != y_shapes[0].shape) or (not hasattr(self, "anchors") or self.anchors.numel() == 0):
            anchors, strides = (t.transpose(0, 1) for t in make_anchors(y_shapes, self.stride, 0.5))
            self.anchors, self.strides, self.shape = anchors, strides, y_shapes[0].shape

        T = x[0].shape[0]
        if self.infer_temporal:
            # decode per-timestep
            A_per_scale = [y.shape[2] * y.shape[3] for y in y_shapes]
            a_slices, s = [], 0
            for A in A_per_scale:
                a_slices.append(slice(s, s + A))
                s += A

            det_list = []
            for i, zi in enumerate(z):
                B, _, H, W = y_shapes[i].shape
                A = H * W
                box_logits, cls_logits = zi.split((self.reg_max * 4, self.nc), 2)
                box_logits_flat = box_logits.flatten(0, 1).view(T * B, self.reg_max * 4, A)
                distances = self.dfl(box_logits_flat)

                a_slice = a_slices[i]
                anchors_i = self.anchors[a_slice]
                stride_i = self.strides[a_slice][:, :1]

                dbox = dist2bbox(distances, anchors_i.unsqueeze(0), xywh=True, dim=1) * stride_i.T  # [T*B,4,A]

                img_h = H * float(self.stride[i]); img_w = W * float(self.stride[i])
                norm = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device, dtype=dbox.dtype).view(1, 4, 1)
                dbox_n = (dbox / norm).clamp(0, 1).view(T, B, 4, A).permute(0, 1, 3, 2)  # [T,B,A,4]
                cls_flat = cls_logits.flatten(0, 1).view(T, B, self.nc, A).permute(0, 1, 3, 2)  # [T,B,A,C]
                det_list.append(torch.cat([dbox_n, cls_flat], dim=-1))  # [T,B,A,4+C]
            return det_list, track_feats
        else:
            # fast: collapse T before decode
            x_mean = [zi.mean(0) for zi in z]
            shape = y_shapes[0].shape
            x_cat = self.concat2([xi.view(shape[0], self.no, -1) for xi in x_mean])
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = self.concat3((dbox, cls.sigmoid()))
            return (y, x_mean)


# Binary Spike version for use during inference
class SpikeDetectBinWithTracking(nn.Module):
    """
      Inference‑time SpikeDetect head (binary graph) with optional temporal decoding and tracking taps.

      Inputs:
        x: list of per‑scale features, each [T, B, C_i, H_i, W_i]
      Returns (eval path):
        (det_list, track_feats)
          • det_list: list per scale of [T, B, A, 4+C] decoded predictions in normalized coords
          • track_feats: list per scale of [B, D, H, W]

      Notes:
        – When `infer_temporal` is True, decoding is done per timestep; otherwise T is collapsed first.
    """
    dynamic = False
    export = False
    shape = None

    def __init__(self, nc: int, ch: Tuple[int, int, int], track_feature_dim: int,
                 module_impl, reg_max: int = 16, infer_temporal: bool = False):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.infer_temporal = infer_temporal

        M = module_impl
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                M.MS_StandardConv(x, c2, 3),
                M.MS_StandardConv(c2, c2, 3),
                M.MS_StandardConvWithoutBN(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                M.MS_StandardConv(x, c3, 3),
                M.MS_StandardConv(c3, c3, 3),
                M.MS_StandardConvWithoutBN(c3, self.nc, 1)
            ) for x in ch
        )

        self.dfl = M.SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.concat1 = Concat(2)
        self.concat2 = Concat(2)
        self.concat3 = Concat(1)

        self.track_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(ch[i] if isinstance(ch[i], int) else int(ch[i]), track_feature_dim, 1, bias=False),
                nn.BatchNorm2d(track_feature_dim)
            ) for i in range(self.nl)
        )

    def forward(self, x: List[torch.Tensor]):
        # taps for tracking (binary path uses the same projection)
        track_feats = [t.mean(0) for t in x]
        track_feats = [self.track_proj[i](t) for i, t in enumerate(track_feats)]

        z = [None] * self.nl
        y_shapes = [xi.mean(0) for xi in x]  # BCHW
        for i in range(self.nl):
            z[i] = self.concat1((self.cv2[i](x[i]), self.cv3[i](x[i])))
            z[i] = z[i]  # keep T if present in input

        # anchors/strides build
        if (self.dynamic or self.shape != y_shapes[0].shape) or (not hasattr(self, "anchors") or self.anchors.numel() == 0):
            anchors, strides = (t.transpose(0, 1) for t in make_anchors(y_shapes, self.stride, 0.5))
            self.anchors, self.strides, self.shape = anchors, strides, y_shapes[0].shape

        T = x[0].shape[0]
        if self.infer_temporal:
            # per-timestep decode
            A_per_scale = [y.shape[2] * y.shape[3] for y in y_shapes]
            a_slices, s = [], 0
            for A in A_per_scale:
                a_slices.append(slice(s, s + A))
                s += A

            det_list = []
            for i, zi in enumerate(z):
                B, _, H, W = y_shapes[i].shape
                A = H * W
                box_logits, cls_logits = zi.split((self.reg_max * 4, self.nc), 2)
                box_logits_flat = box_logits.flatten(0, 1).view(T * B, self.reg_max * 4, A)
                distances = self.dfl(box_logits_flat)

                a_slice = a_slices[i]
                anchors_i = self.anchors[a_slice]
                stride_i = self.strides[a_slice][:, :1]

                dbox = dist2bbox(distances, anchors_i.unsqueeze(0), xywh=True, dim=1) * stride_i.T
                img_h = H * float(self.stride[i]); img_w = W * float(self.stride[i])
                norm = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device, dtype=dbox.dtype).view(1, 4, 1)
                dbox_n = (dbox / norm).clamp(0, 1).view(T, B, 4, A).permute(0, 1, 3, 2)
                cls_flat = cls_logits.flatten(0, 1).view(T, B, self.nc, A).permute(0, 1, 3, 2)
                det_list.append(torch.cat([dbox_n, cls_flat], dim=-1))
            return det_list, track_feats

        # fast collapsed-T decode
        x_mean = [zi.mean(0) for zi in z]
        shape = y_shapes[0].shape
        x_cat = self.concat2([xi.view(shape[0], self.no, -1) for xi in x_mean])
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = self.concat3((dbox, cls.sigmoid()))
        return (y, x_mean)

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)

class eTraMSpikeYOLOWithTracking(nn.Module):
    """
      End‑to‑end eTraM SpikeYOLO model with tracking support.

      Modes:
        • train: uses SpikeDetectTrainWithTracking → returns raw logits per scale for loss
        • eval/inference: uses SpikeDetectBinWithTracking → returns decoded per‑timestep detections

      Forward() accepts [B,T,H,W] (auto‑splits ON/OFF) or [B,T,2,H,W]. It rearranges to [T,B,2,H,W] for the backbone.

      Output (train):
        detections: list per scale of [T,B,4*reg_max + C,H,W]
        track_features: list per scale of [B,D,H,W]

      Output (eval):
        det_list: list per scale of [T,B,A,4+C]
        track_features: list per scale of [B,D,H,W]

      Args:
          num_classes: Number of traffic participant classes
          input_size: Input image size (height, width)
          time_steps: Number of temporal steps
          track_feature_dim: Dimension of tracking features
          device_manager: Device manager for multi-GPU
          verbose: Enable verbose logging
          class_names: Optional class names for display
          mode: "train" or "eval"
          frame_rate: Optional override; if None, computed as time_steps * 1e6 / window_duration_us
          window_duration_us: Size of the temporal aggregation window in microseconds (default 1e6 = 1s)
    """

    def __init__(self,
                 num_classes: int = 3,
                 input_size: Tuple[int, int] = (720, 1280),
                 time_steps: int = 8,
                 track_feature_dim: int = 128,
                 device_manager=None,
                 verbose: bool = True,
                 class_names = None,
                 mode: str = "train",
                 frame_rate: float = None,
                 window_duration_us: int = 100000):
        """
        Initialize eTraM SpikeYOLO with tracking.

        Args:
            num_classes: Number of traffic participant classes
            input_size: Input image size (height, width)
            time_steps: Number of temporal steps
            track_feature_dim: Dimension of tracking features
            device_manager: Device manager for multi-GPU
            verbose: Enable verbose logging
            class_names: Optional class names for display
            mode: "train" or "eval"
            frame_rate: Optional override; if None, computed as time_steps * 1e6 / window_duration_us
            window_duration_us: Size of the temporal aggregation window in microseconds (default 1e6 = 1s)
        """
        super().__init__()

        # Select module implementation if not provided (default to train graph)
        self.M = _select_spike_module_impl(mode)
        # Sanity log: are we using the binary (eval) modules?
        mod_name = getattr(self.M, '__name__', str(self.M))
        try:
            _qtrick = getattr(self.M.mem_update(), 'qtrick', None)
            _is_binary = (_qtrick is not None) and (_qtrick.__class__.__name__ == 'MultiSpike4')
        except Exception:
            _is_binary = False
        logger.info(f"Using binary modules: {_is_binary} (module={mod_name})")

        self.num_classes = num_classes
        self.input_size = input_size
        self.time_steps = time_steps
        # Compute effective frame rate: each timestep aggregates events into one frame
        # frame_rate = time_steps / window_duration_seconds = time_steps * 1e6 / window_duration_us
        self.window_duration_us = int(window_duration_us)
        if frame_rate is None:
            self.frame_rate = float(self.time_steps) * 1e6 / max(1.0, float(self.window_duration_us))
        else:
            self.frame_rate = float(frame_rate)
        self.track_feature_dim = track_feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        # Flag to track if weights have been initialized
        self._weights_initialized = False


        # OPTIMIZATION: Remove MS_GetT - go directly to backbone
        # Input: [B, T, H, W] = [batch, 8, 720, 1280] - already temporally binned
        self.input_layer = nn.Identity()  # Direct pass-through

        # Backbone split into stages to capture intermediate features for FPN
        # Stage 0-1: 720×1280 → 360×640 → 360×640
        self.backbone_stage0 = nn.Sequential(
            self.M.MS_DownSampling(2, 64, 7, 2, 3, True),  # 2 input (ON/OFF) → 64, 720×1280 → 360×640
            self.M.MS_AllConvBlock(64, 4, 7),  # 64 ch, mlp_ratio=4, k=7, keeps 360×640
        )

        # Stage 2-3: 360×640 → 180×320 → 180×320
        self.backbone_stage1 = nn.Sequential(
            self.M.MS_DownSampling(64, 128, 3, 2, 1, False),  # 64 → 128, 360×640 → 180×320
            self.M.MS_AllConvBlock(128, 4, 7),  # keeps 180×320
        )

        # Stage 4-5: 180×320 → 90×160 → 90×160 (P3 level)
        self.backbone_stage2 = nn.Sequential(
            self.M.MS_DownSampling(128, 256, 3, 2, 1, False)  # 128 → 256, 180×320 → 90×160
            , self.M.MS_ConvBlock(256, 3, 7),  # keeps 90×160
        )

        # Stage 6-7: 90×160 → 45×80 → 45×80 (P4 level)
        self.backbone_stage3 = nn.Sequential(
            self.M.MS_DownSampling(256, 512, 3, 2, 1, False),  # 256 → 512, 90×160 → 45×80
            self.M.MS_ConvBlock(512, 3, 7),  # keeps 45×80
        )

        # Stage 8-10: 45×80 → 23×40 → 23×40 → 23×40 (P5 level)
        self.backbone_stage4 = nn.Sequential(
            self.M.MS_DownSampling(512, 1024, 3, 2, 1, False),  # 512 → 1024, 45×80 → 23×40  (via k=3,s=2,p=1)
            self.M.MS_ConvBlock(1024, 2, 7),  # keeps 23×40
            self.M.SpikeSPPF(1024, 1024, 5)  # SPPF at 23×40
        )

        # FPN-style detection heads with upsampling/fusion
        # P5: Start from backbone output (23×40)
        self.detection_head_p5 = nn.Sequential(
            self.M.MS_StandardConv(1024, 512, 1, 1),  # Reduce channels: 1024 → 512, 23×40
        )

        # P4: Upsample P5 and fuse with backbone P4 features (45×80)
        self.p4_lateral = self.M.MS_StandardConv(512, 256, 1, 1)  # 23×40: 512 → 256 (before upsample)
        # Note: Use interpolation with specific size to match backbone P4 (45×80)
        self.p4_fusion = self.M.MS_StandardConv(256 + 512, 256, 3, 1)  # Fuse: 256 (P5↑) + 512 (P4) → 256
        self.detection_head_p4 = nn.Sequential(
            self.M.MS_StandardConv(256, 256, 3, 1),  # 45×80
        )

        # P3: Upsample P4 and fuse with backbone P3 features (90×160)
        self.p3_lateral = self.M.MS_StandardConv(256, 128, 1, 1)  # 45×80: 256 → 128 (before upsample)
        # Note: Use interpolation with specific size to match backbone P3 (90×160)
        self.p3_fusion = self.M.MS_StandardConv(128 + 256, 128, 3, 1)  # Fuse: 128 (P4↑) + 256 (P3) → 128
        self.detection_head_p3 = nn.Sequential(
            self.M.MS_StandardConv(128, 128, 3, 1),  # 90×160
        )

        # Detection with tracking capability
        if mode.lower() in ("eval", "inference", "test"):
            self.detect = SpikeDetectBinWithTracking(
                nc=num_classes,
                ch=(512, 256, 128),
                track_feature_dim=track_feature_dim,
                module_impl=self.M,
                infer_temporal=True  # decode per-timestep at inference for tracking
            )
        else:
            self.detect = SpikeDetectTrainWithTracking(
                nc=num_classes,
                ch=(512, 256, 128),
                track_feature_dim=track_feature_dim,
                module_impl=self.M,
                infer_temporal=True  # training returns raw per-timestep tensors; decode happens downstream
            )

        # Post-processing tracker. Uses ByteTracker for inference only. It is a heuristic algorithm
        self.tracker = SpikeYOLOTracker(
            nc=num_classes,
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=self.frame_rate
        )

        if verbose:
            logger.info(f"Tracker frame_rate set to {self.frame_rate:.2f} FPS (time_steps={self.time_steps}, window_duration_us={self.window_duration_us})")
            logger.info(f"eTraM SpikeYOLO with Tracking initialized:")
            logger.info(f"  Classes: {num_classes}")
            logger.info(f"  Input size: {input_size}")
            logger.info(f"  Time steps: {time_steps}")
            logger.info(f"  Tracking features: {track_feature_dim}")
            logger.info(f"  Device: {self.device}")

        # Initialize weights properly for stable training with AMP
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights properly for stable training with mixed precision.
        Uses Kaiming initialization for Conv layers and Xavier for Linear layers.
        """
        if self._weights_initialized:
            return

        logger.info("Initializing model weights for AMP-compatible training...")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for Conv layers (good for ReLU/SiLU activations)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # Xavier initialization for Linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                # Batch/Group/Layer norm initialization
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self._weights_initialized = True
        logger.info("Weight initialization complete")

    def _track_per_timestep(self, det_list: List[torch.Tensor]) -> Union[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Run ByteTrack per timestep using decoded, per-scale detections.

        Args:
            det_list: list per scale of [T, B, A, 4+C] tensors with normalized xywh and class probs.

        Returns:
            If B==1: list of length T with tensors [N_t, 4+C+2] per timestep (xywh, class_probs, conf, track_id).
            If B>1:  list of length B, each is a list length T as above.
        """
        if not det_list:
            return []
        # Infer T, B from first scale
        t0 = det_list[0]
        T, B, A0, F = t0.shape
        C = F - 4

        # Collect outputs per batch item
        outputs_per_batch: List[List[torch.Tensor]] = []

        for b in range(B):
            # Reset tracker per sequence in the batch to avoid cross-video ID bleeding
            self.tracker.reset()
            seq_outputs: List[torch.Tensor] = []
            for t in range(T):
                # Merge across scales for this (t, b)
                per_t_b = []
                for s in range(len(det_list)):
                    per_t_b.append(det_list[s][t, b])  # [A_s, 4+C]
                merged = torch.cat(per_t_b, dim=0)  # [N, 4+C]
                if merged.numel() == 0:
                    # No detections this timestep
                    seq_outputs.append(torch.zeros((0, 4 + C + 2), device=merged.device, dtype=merged.dtype))
                    continue

                # Split boxes and class probabilities
                boxes_xywh = merged[:, :4]
                class_probs = merged[:, 4:]
                # confidence as max class prob
                conf, _ = class_probs.max(dim=1, keepdim=True)

                # Build [x,y,w,h, class_probs..., conf] expected by tracker.update (which converts internally)
                det_for_tracker = torch.cat([boxes_xywh, class_probs, conf], dim=1)

                # --- Apply simple NMS before passing to ByteTrack ---
                # Convert normalized xywh to xyxy for NMS
                x, y, w, h = boxes_xywh.unbind(1)
                x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

                # Perform class-agnostic NMS using torchvision
                from torchvision.ops import nms
                keep = nms(boxes_xyxy, conf.squeeze(1), iou_threshold=0.5)
                det_for_tracker = det_for_tracker[keep]
                boxes_xywh = boxes_xywh[keep]
                class_probs = class_probs[keep]
                conf = conf[keep]

                # Run update (image_shape = self.input_size, normalized boxes OK — converter scales if needed)
                tracked = self.tracker.update(det_for_tracker, track_features=None, image_shape=self.input_size)
                seq_outputs.append(tracked)
            outputs_per_batch.append(seq_outputs)

        # If B==1, unwrap one level for convenience
        return outputs_per_batch[0] if B == 1 else outputs_per_batch

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with tracking.

        Args:
            x: Input tensor [B, T, H, W] or [B, H, W]

        Returns:
            Detection results with track_id
        """
        # Ensure input is on correct device
        x = x.to(self.device)

        # If we're in eval mode, ensure the selected module uses MultiSpike4 (binary) for mem_update
        if not self.training and hasattr(self.M, 'mem_update'):
            MultiSpike4 = getattr(self.M, 'MultiSpike4', None)
            try:
                is_bin = isinstance(self.M.mem_update().qtrick, MultiSpike4) if MultiSpike4 is not None else False
            except Exception:
                is_bin = False
            if not is_bin:
                logger.warning("Eval path is not using binary modules; build with mode='eval' if you expect binary behavior.")

        # Handle different input formats
        if x.dim() == 3:  # [B, H, W] - add time dimension
            x = x.unsqueeze(1).repeat(1, self.time_steps, 1, 1)
        elif x.dim() == 4:  # [B, T, H, W]
            pass
        else:
            raise ValueError(f"Expected input shape [B, T, H, W] or [B, H, W], got {x.shape}")

        # OPTIMIZATION: Skip MS_GetT - process directly
        # x: [B, T, H, W] = [batch, 8, 720, 1280] - already temporally binned
        # The input should already be in [B, T, 2, H, W] format with ON/OFF channels
        # If it's [B, T, H, W], we need to split into ON/OFF channels
        B, T, H, W = x.shape

        # Check if input already has channel dimension
        if x.dim() == 4:  # [B, T, H, W] - single channel, need to split into ON/OFF
            # Split accumulated event values into ON/OFF channels
            # Channel 0: Positive values (ON events) = max(0, x)
            # Channel 1: Negative values (OFF events) = max(0, -x)
            x_on = torch.clamp(x, min=0)   # ON events (positive values)
            x_off = torch.clamp(-x, min=0)  # OFF events (negative values, made positive)
            x = torch.stack([x_on, x_off], dim=2)  # [B, T, 2, H, W]
        elif x.dim() == 5:  # [B, T, C, H, W] - already has channels
            # Ensure it has 2 channels (ON/OFF)
            if x.shape[2] != 2:
                raise ValueError(f"Expected 2 channels (ON/OFF) but got {x.shape[2]} channels")
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}, expected [B, T, H, W] or [B, T, 2, H, W]")

        # channels: 0 = ON (positive events), 1 = OFF (negative events made positive)

        # Convert to [T, B, C, H, W] format expected by MS_DownSampling
        # x is now [B, T, 2, H, W] after processing
        x = x.permute(1, 0, 2, 3, 4)  # [B, T, 2, H, W] -> [T, B, 2, H, W]

        x = self.input_layer(x)        # Direct pass-through (no temporal manipulation)

        # FPN-style backbone processing: capture intermediate features
        x = self.backbone_stage0(x)  # [T, B,  2, 720, 1280] -> [T, B,  64, 360, 640]
        x = self.backbone_stage1(x)  # [T, B, 64, 360, 640]  -> [T, B, 128, 180, 320]
        x = self.backbone_stage2(x)  # [T, B,128, 180, 320]  -> [T, B, 256,  90, 160] (P3 level)
        p3_backbone = x  # Save P3 backbone features:   [T, B, 256,  90, 160]

        x = self.backbone_stage3(x)  # [T, B,256,  90, 160]  -> [T, B, 512,  45,  80] (P4 level)
        p4_backbone = x  # Save P4 backbone features:   [T, B, 512,  45,  80]

        x = self.backbone_stage4(x)  # [T, B,512,  45,  80]  -> [T, B,1024,  23,  40] (P5 level)
        p5_backbone = x  # Save P5 backbone features:   [T, B,1024,  23,  40]

        # FPN-style detection head processing with upsampling
        # P5: Start from backbone output (23×40)
        p5_features = self.detection_head_p5(p5_backbone)  # [T, B,1024, 23, 40] -> [T, B, 512, 23, 40]

        # P4: Lateral, upsample P5, fuse with P4 backbone (45×80)
        p5_lateral = self.p4_lateral(p5_features)  # [T, B, 512, 23, 40] -> [T, B, 256, 23, 40]
        _, _, _, p4_h, p4_w = p4_backbone.shape  # target: (45, 80)
        p5_up = torch.nn.functional.interpolate(
            p5_lateral.flatten(0, 1), size=(p4_h, p4_w), mode='nearest'
        ).view(p5_lateral.shape[0], p5_lateral.shape[1], p5_lateral.shape[2], p4_h, p4_w)  # [T, B, 256, 45, 80]
        p4_fused = torch.cat([p5_up, p4_backbone], dim=2)  # [T, B, 256+512, 45, 80] = [T, B, 768, 45, 80]
        p4_fused = self.p4_fusion(p4_fused)  # [T, B, 768, 45, 80] -> [T, B, 256, 45, 80]
        p4_features = self.detection_head_p4(p4_fused)  # [T, B, 256, 45, 80] -> [T, B, 256, 45, 80]

        # P3: Lateral, upsample P4, fuse with P3 backbone (90×160)
        p4_lateral = self.p3_lateral(p4_features)  # [T, B, 256, 45, 80] -> [T, B, 128, 45, 80]
        _, _, _, p3_h, p3_w = p3_backbone.shape  # target: (90, 160)
        p4_up = torch.nn.functional.interpolate(
            p4_lateral.flatten(0, 1), size=(p3_h, p3_w), mode='nearest'
        ).view(p4_lateral.shape[0], p4_lateral.shape[1], p4_lateral.shape[2], p3_h, p3_w)  # [T, B, 128, 90, 160]
        p3_fused = torch.cat([p4_up, p3_backbone], dim=2)  # [T, B, 128+256, 90, 160] = [T, B, 384, 90, 160]
        p3_fused = self.p3_fusion(p3_fused)  # [T, B, 384, 90, 160] -> [T, B, 128, 90, 160]
        p3_features = self.detection_head_p3(p3_fused)  # [T, B, 128, 90, 160] -> [T, B, 128, 90, 160]

        # Detection heads (order P5→P4→P3). In train mode: returns list of [T,B,4*reg_max + C,H,W].
        # In eval mode (binary head): returns (det_list per scale [T,B,A,4+C], track_feats per scale [B,D,H,W]).
        detections, track_features = self.detect([p5_features, p4_features, p3_features])

        # If eval head returned per-timestep decoded detections (list per scale of [T,B,A,4+C]),
        # perform ByteTrack update per timestep.
        if isinstance(self.detect, SpikeDetectBinWithTracking) and isinstance(detections, list):
            tracked_seq = self._track_per_timestep(detections)
            return tracked_seq, track_features

        return detections, track_features

    def reset_tracker(self):
        """Reset the tracker state."""
        self.tracker.reset()

    def apply_tracking(self, detections: torch.Tensor, track_features: torch.Tensor = None,
                       image_shape: Tuple[int, int] = None) -> torch.Tensor:
        """
        Apply tracking to processed detections.

        This method should be called after decoding detections (DFL decoding, NMS, filtering).

        Args:
            detections: Processed detections [N, 12] = [x, y, w, h, class_0, ..., class_7, conf]
                       or [N, 4+nc+1] = [x, y, w, h, class_0, ..., class_nc-1, conf]
            track_features: Optional tracking features [N, 128] for each detection
            image_shape: Image dimensions (height, width), defaults to self.input_size

        Returns:
            Tracked detections [N, 13] = [x, y, w, h, class_0, ..., class_7, conf, track_id]
        """
        if image_shape is None:
            image_shape = self.input_size

        # Apply tracking with features
        tracked_detections = self.tracker.update(detections, track_features, image_shape)

        return tracked_detections

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'eTraM SpikeYOLO with Tracking',
            'architecture': 'BICLab SpikeYOLO + Post-processing Tracking',
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'time_steps': self.time_steps,
            'track_feature_dim': self.track_feature_dim,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'class_names': self.class_names
        }


# def create_etram_model_with_tracking(num_classes: int = 8,
#                                    input_size: Tuple[int, int] = (720, 1280),
#                                    time_steps: int = 8,
#                                    track_feature_dim: int = 128,
#                                    verbose: bool = True) -> eTraMSpikeYOLOWithTracking:
#     """
#     Create eTraM SpikeYOLO model with tracking capability.
#
#     Args:
#         num_classes: Number of traffic participant classes
#         input_size: Input image size (height, width)
#         time_steps: Number of temporal steps
#         track_feature_dim: Dimension of tracking features
#         verbose: Enable verbose logging
#
#     Returns:
#         eTraM SpikeYOLO model with tracking
#     """
#     return eTraMSpikeYOLOWithTracking(
#         num_classes=num_classes,
#         input_size=input_size,
#         time_steps=time_steps,
#         track_feature_dim=track_feature_dim,
#         verbose=verbose
#     )


# if __name__ == "__main__":
#     # Test the tracking implementation
#     logger.info("Testing eTraM SpikeYOLO with Tracking...")
#
#     try:
#         # Create model
#         model = create_etram_model_with_tracking(
#             num_classes=8,
#             input_size=(720, 1280),
#             time_steps=8,
#             track_feature_dim=128,
#             verbose=True
#         )
#
#         # Test with dummy input
#         batch_size = 2
#         time_steps = 8
#         height, width = 720, 1280
#
#         dummy_input = torch.randn(batch_size, time_steps, height, width)
#
#         # Test forward pass
#         with torch.no_grad():
#             output = model(dummy_input)
#             logger.info(f"Output shape: {output.shape}")
#             logger.info(f"Output device: {output.device}")
#
#         # Test model info
#         info = model.get_model_info()
#         logger.info(f"Model info: {info}")
#
#         logger.info("SUCCESS:  eTraM SpikeYOLO with Tracking test completed successfully!")
#
#     except Exception as e:
#         logger.error(f"ERROR:  Test failed: {e}")
#         import traceback
#         traceback.print_exc()
#
#     def _convert_bytetracker_tracks_to_tensor(self, tracks, device: torch.device) -> torch.Tensor:
#         """Convert BYTETracker list[STrack] to our tensor format via xyxy+conf+cls+id -> SpikeYOLO format."""
#         if tracks is None or len(tracks) == 0:
#             return torch.zeros((0, 13), device=device)
#         rows = []
#         for t in tracks:
#             # Get box in xyxy
#             if hasattr(t, 'tlbr'):
#                 x1, y1, x2, y2 = map(float, t.tlbr)
#             elif hasattr(t, 'tlwh'):
#                 x, y, w, h = map(float, t.tlwh)
#                 x1, y1, x2, y2 = x, y, x + w, y + h
#             else:
#                 # Skip if no box
#                 continue
#             conf = float(getattr(t, 'score', 1.0))
#             cls_id = int(getattr(t, 'cls', 0))
#             track_id = int(getattr(t, 'track_id', -1))
#             rows.append([x1, y1, x2, y2, conf, float(cls_id), float(track_id)])
#         tracked_dets = np.asarray(rows, dtype=np.float32)
#         return self._convert_from_bytetracker_format(tracked_dets, device)
# === Shape crib sheet ===
# Backbones/heads see: [T,B,C,H,W]
# Train detect output per scale: [T,B,4*reg_max + C,H,W]
# Loss reshape per scale: [T,B,H*W,4*reg_max + C] → [T*B,A,F]
# Targets per image: [N,8] (cls, x, y, w, h, conf, track_id, ts)
# Multi‑scale is a Python list: [P5, P4, P3]