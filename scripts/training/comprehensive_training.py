#!/usr/bin/env python3
"""
Comprehensive Training Script for Traffic_Monitoring SpikeYOLO
Trains on all training data with validation monitoring and detailed logging.
"""

import os
import sys
import time
import logging
import json
import argparse
import warnings
from datetime import datetime
from typing import Optional
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress deprecation warnings from external libraries
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')
warnings.filterwarnings('ignore', message='.*deprecated.*', category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.cuda.amp import GradScaler
import torch.amp
import torch.ao.quantization as quantization
import numpy as np

# Import our modules
from src.config_loader import ConfigLoader as Config
from src.data_loader import create_ultra_low_memory_dataloader as create_dataloader
from src.etram_spikeyolo_tracking import eTraMSpikeYOLOWithTracking
from src.yolo_loss import YOLOLoss
from src.logging_utils import setup_logging as setup_logging_unified

# ---- Pruning utilities ----
def _compute_model_sparsity(model: nn.Module) -> float:
    """Compute overall sparsity using pruning masks when present (accurate during reparameterization)."""
    total_params, zero_params = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, 'weight_mask'):
                mask = m.weight_mask.detach()
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()
            elif hasattr(m, 'weight') and isinstance(m.weight, torch.Tensor):
                w = m.weight.detach()
                total_params += w.numel()
                zero_params += (w == 0).sum().item()
    if total_params == 0:
        return 0.0
    return zero_params / float(total_params)

def apply_pruning(model, pruning_config):
    """Apply pruning to the model based on configuration."""
    logger = logging.getLogger(__name__)
    
    if not pruning_config.get('enabled', False):
        logger.info("Pruning disabled")
        return model
    
    pruning_type = pruning_config.get('type', 'unstructured')
    pruning_amount = pruning_config.get('amount', 0.1)  # 10% by default
    
    logger.info(f"Applying {pruning_type} pruning with {pruning_amount*100:.1f}% sparsity")
    
    if pruning_type == 'unstructured':
        # Unstructured pruning - removes individual weights
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_amount,
        )
        
    elif pruning_type == 'structured':
        # Structured pruning - removes entire channels/filters
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=pruning_amount, n=2, dim=0)
            elif isinstance(module, nn.Linear):
                prune.ln_structured(module, name='weight', amount=pruning_amount, n=2, dim=1)

    # Count pruned parameters using masks (accurate under pruning reparameterization)
    sparsity = _compute_model_sparsity(model)
    total_params = sum(
        (m.weight_mask.numel() if hasattr(m, 'weight_mask') else (m.weight.numel() if hasattr(m, 'weight') else 0))
        for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
    )
    pruned_params = int(round(sparsity * total_params))
    logger.info(f"Pruning applied: {pruned_params}/{total_params} parameters pruned ({sparsity * 100:.2f}% sparsity)")
    # NOTE: To permanently apply masks before export/eval, call prune.remove(module, 'weight') on pruned layers.

    return model

def apply_iterative_pruning(model, epoch, total_epochs, pruning_config):
    """Apply iterative pruning during training."""
    logger = logging.getLogger(__name__)
    
    if not pruning_config.get('enabled', False):
        return model
    
    # Calculate pruning amount based on training progress
    max_sparsity = pruning_config.get('max_sparsity', 0.5)  # 50% max sparsity
    pruning_schedule = pruning_config.get('schedule', 'linear')
    
    if pruning_schedule == 'linear':
        current_sparsity = max_sparsity * (epoch / total_epochs)
    elif pruning_schedule == 'cosine':
        current_sparsity = max_sparsity * (1 - np.cos(np.pi * epoch / total_epochs)) / 2
    else:
        current_sparsity = max_sparsity * (epoch / total_epochs)

    # Apply pruning if we haven't reached target sparsity (mask-aware)
    current_model_sparsity = _compute_model_sparsity(model)

    if current_sparsity > current_model_sparsity + 1e-6:
        # Convert absolute target to fraction of remaining weights to prune now:
        # to_prune_now = (target - current) / (1 - current)
        remaining = max(1e-8, 1.0 - current_model_sparsity)
        to_prune_now = (current_sparsity - current_model_sparsity) / remaining
        to_prune_now = float(min(max(to_prune_now, 0.0), 1.0))  # clamp for safety

        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=to_prune_now,
        )

        new_sparsity = _compute_model_sparsity(model)
        logger.info(f"Epoch {epoch}: iterative pruning -> target={current_sparsity*100:.2f}% | "
              f"prev={current_model_sparsity * 100:.2f}% | now={new_sparsity * 100:.2f}% | "
              f"pruned_now={to_prune_now * 100:.2f}% of remaining")
        # NOTE: To permanently apply masks before export/eval, call prune.remove(module, 'weight') on pruned layers.

    return model

def finalize_pruning(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, 'weight_mask'):
            prune.remove(m, 'weight')  # make pruning permanent

def setup_logging(log_dir: str, log_level: str = "INFO", log_file_name: Optional[str] = None, log_format: Optional[str] = None):
    """Setup logging - wrapper around unified logging utility."""
    return setup_logging_unified(
        log_dir=log_dir,
        log_level=log_level,
        log_file_name=log_file_name,
        log_format=log_format,
        script_name="training"
    )

def prepare_model_for_qat(model, config: Config, device: torch.device):
    """Prepare model for Quantization-Aware Training (QAT)."""
    logger = logging.getLogger(__name__)
    
    use_qat = config.get('training.use_quantization', False)
    if not use_qat:
        return model
    
    logger.info("Preparing model for Quantization-Aware Training (QAT)...")
    
    # Get quantization configuration
    backend = config.get('training.quantization_backend', 'fbgemm')
    qconfig_type = config.get('training.quantization_qconfig', 'default')

    # Set quantization backend
    if backend in ('fbgemm', 'qnnpack'):
        torch.backends.quantized.engine = backend
    else:
        logger.warning(f"Unknown quantization backend: {backend}, defaulting to 'fbgemm'")
        torch.backends.quantized.engine = 'fbgemm'
    
    # Get QConfig based on type
    from torch.ao.quantization.qconfig import default_qconfig
    if qconfig_type == 'default':
        qconfig = default_qconfig
    elif qconfig_type == 'per_channel':
        # Per-channel quantization (more accurate, slower)
        from torch.ao.quantization.qconfig import default_per_channel_qconfig
        qconfig = default_per_channel_qconfig if backend == 'fbgemm' else default_qconfig
    else:
        qconfig = default_qconfig

    logger.info(f"Using quantization backend: {backend}, QConfig: {qconfig_type}")

    # Set quantization config for the model
    model.qconfig = qconfig

    # Prepare model for QAT (inserts fake quantization modules)
    # Use prepare_qat for training with fake quantization
    try:
        model = quantization.prepare_qat(model, inplace=False)
        logger.info("Model prepared for QAT - fake quantization modules inserted")
    except AttributeError:
        # Fallback to prepare if prepare_qat doesn't exist
        logger.warning("prepare_qat not available, using prepare with training=True")
        model = quantization.prepare(model, inplace=False)
        model.train()  # Ensure model is in training mode

    # Move model back to device (prepare_qat might have moved it)
    model = model.to(device)

    logger.info("Model will be trained with quantization simulation (INT8-aware) on GPU")
    logger.info("Fake quantization layers will simulate INT8 during forward pass (GPU)")
    logger.info("Gradients will flow through FP32 (full precision) during backward pass")
    logger.info("After training:")
    logger.info("  - For CPU inference: Use quantization.convert() (CPU-only)")
    logger.info("  - For GPU inference: Export to ONNX → Convert to TensorRT INT8 (GPU)")

    return model

def create_model(config: Config, device: torch.device):
    """Create and initialize the model."""
    logger = logging.getLogger(__name__)

    logger.info("Creating Traffic_Monitoring SpikeYOLO model...")

    model = eTraMSpikeYOLOWithTracking(
        num_classes=config.get_num_classes(),
        input_size=config.get_input_size(),
        time_steps=config.get_time_steps(),
        track_feature_dim=config.get_track_feature_dim(),
        class_names = config.get_class_names(),
        mode="train",
        window_duration_us=config.get_window_us()
    )

    # Move model to device
    model = model.to(device)

    # If device is CPU, ensure all model parameters and buffers are on CPU
    if device.type == 'cpu':
        model = model.cpu()
        # Double-check: ensure all parameters are on CPU
        for name, param in model.named_parameters():
            if param.is_cuda:
                param.data = param.data.cpu()
                logger.debug(f"Moved parameter {name} to CPU")
        for name, buffer in model.named_buffers():
            if buffer.is_cuda:
                buffer.data = buffer.data.cpu()
                logger.debug(f"Moved buffer {name} to CPU")

    # Prepare model for Quantization-Aware Training (QAT) if enabled
    model = prepare_model_for_qat(model, config, device)

    # Convert model to FP16 if enabled (direct FP16, not AMP)
    use_fp16 = config.get('training.use_fp16', False)
    if use_fp16 and device.type == 'cuda':
        logger.info("Converting model to FP16 (half precision) for faster training...")
        logger.warning("WARNING: FP16 may cause NaN with SpikeYOLO operations - testing carefully")
        model = model.half()  # Convert all parameters to FP16
        logger.info("Model converted to FP16 - all parameters are now FP16")
        logger.info("Inputs will also be converted to FP16 during training")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate model size based on dtype
    param_dtype = next(model.parameters()).dtype
    bytes_per_param = 2 if param_dtype == torch.float16 else 4
    model_size_mb = total_params * bytes_per_param / 1024 / 1024

    logger.info(f"Model created successfully:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model dtype: {param_dtype}")
    logger.info(f"  Model size: {model_size_mb:.2f} MB")

    return model

def warmup_learning_rate(optimizer, epoch, warmup_epochs, base_lr, start_epoch=1):
    """Apply linear warmup learning rate."""
    # Calculate adjusted epoch for warmup (account for resume from checkpoint)
    adjusted_epoch = epoch - (start_epoch - 1) if start_epoch > 1 else epoch
    if adjusted_epoch < warmup_epochs:
        lr = base_lr * (adjusted_epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def create_optimizer_and_scheduler(model, config: Config, train_loader=None):
    """Create optimizer and learning rate scheduler."""
    logger = logging.getLogger(__name__)

    # Create optimizer (SGD with momentum for better localization)
    base_lr = config.get('training.learning_rate', 0.001)
    optimizer_type = config.get('training.optimizer', 'sgd')  # 'sgd' or 'adamw'

    if optimizer_type.lower() == 'sgd':
        momentum = config.get('training.momentum', 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=momentum,
            weight_decay=config.get('training.weight_decay', 0.0005)
        )
        logger.info(f"Optimizer: SGD with momentum={momentum} (lr={base_lr})")
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=config.get('training.weight_decay', 0.0005)
        )
        logger.info(f"Optimizer: AdamW (lr={base_lr})")

    # Store base LR and warmup for later use
    config._base_lr = base_lr
    config._warmup_epochs = config.get('training.warmup_epochs', 3)

    # Create scheduler
    scheduler_type = config.get('training.lr_scheduler', 'step')
    total_epochs = config.get('training.epochs', 5)

    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('training.lr_decay_steps', 2),
            gamma=config.get('training.lr_decay', 0.5)
        )
    elif scheduler_type == 'cosine':
        # Cosine annealing scheduler with minimum learning rate
        min_lr = config.get('training.min_learning_rate', base_lr * 0.01)  # Default: 1% of base LR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_lr  # Minimum learning rate (cosine decays to this value)
        )
        logger.info(f"CosineAnnealingLR configured: T_max={total_epochs}, eta_min={min_lr}")
    elif scheduler_type == 'cyclic':
        # OneCycleLR for cyclical learning rate
        max_lr = config.get('training.max_learning_rate', base_lr * 2)
        # FIX: Calculate steps_per_epoch from train_loader if available
        steps_per_epoch = len(train_loader) if train_loader is not None else 1
        if steps_per_epoch == 1:
            logger.warning("OneCycleLR: steps_per_epoch is 1. This may not work correctly. Consider updating after dataloader creation.")

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3  # 30% for warmup phase
        )
        config._is_cyclic = True
        logger.info(f"OneCycleLR configured with {steps_per_epoch} steps per epoch")
    else:
        scheduler = None
        config._is_cyclic = False

    logger.info(f"Scheduler: {scheduler_type if scheduler else 'None'}")
    logger.info(f"Warmup epochs: {config._warmup_epochs}")

    return optimizer, scheduler

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config, batch_size=None, scaler=None, scheduler=None):
    """Train for one epoch with gradient accumulation and optional AMP."""
    logger = logging.getLogger(__name__)

    model.train()
    total_loss = 0.0
    total_samples = 0
    num_optim_steps = 0

    # Check if AMP is enabled
    use_amp = config.get('training.use_amp', False) and device.type == 'cuda' and scaler is not None
    if use_amp:
        logger.info(f"Using Automatic Mixed Precision (AMP) for epoch {epoch}")

    # Gradient accumulation configuration
    accumulation_steps = config.get('training.gradient_accumulation_steps', 10)
    actual_batch_size = batch_size if batch_size is not None else config.get('training.batch_size', 25)
    effective_batch_size = actual_batch_size * accumulation_steps

    logger.info(f"Starting epoch {epoch}")
    logger.info(f"Gradient accumulation: {accumulation_steps} steps (effective batch size: {effective_batch_size})")

    # Zero gradients at the start of epoch
    optimizer.zero_grad()

    # Profiling metrics
    total_data_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_optimizer_time = 0.0
    batch_start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # Measure data loading time
        data_time = time.time() - batch_start_time
        total_data_time += data_time
        try:
            # Move data to device (explicitly move to CPU first if device is CPU to avoid CUDA context issues)
            # Use frames - frames are [B, T, H, W] which is what the model expects
            # Raw events are no longer included in the batch to save memory
            if 'frames' not in batch or batch['frames'] is None:
                raise ValueError("Frames not found in batch. This should not happen - frames are required for model input.")
            
            # Use pre-computed frames (memory efficient)
            if device.type == 'cpu':
                events = batch['frames'].cpu().to(device)  # frames are [B, T, H, W]
            else:
                events = batch['frames'].to(device)
            
            if device.type == 'cpu':
                targets = batch['targets'].cpu().to(device)  # Move targets to device first
            else:
                targets = batch['targets'].to(device)  # Move targets to device first

            # Convert inputs to FP16 if model is FP16 (direct FP16 training)
            use_fp16 = config.get('training.use_fp16', False)
            if use_fp16 and device.type == 'cuda':
                # Check if model is FP16
                if next(model.parameters()).dtype == torch.float16:
                    events = events.half()  # Convert inputs to FP16

            event_timestamps = batch.get('event_timestamps', None)  # NEW: Get event timestamps

            # Convert targets tensor to list of tensors for each batch item
            targets_list = []
            for i in range(targets.shape[0]):
                target_tensor = targets[i]  # Get targets for batch item i
                # Ensure target_tensor is on CPU first if device is CPU
                if device.type == 'cpu':
                    target_tensor = target_tensor.cpu()
                # Filter out zero-padded targets (assuming targets with all zeros are padding)
                valid_mask = target_tensor.sum(dim=1) != 0
                if valid_mask.any():
                    targets_list.append(target_tensor[valid_mask].to(device))
                else:
                    # Create empty tensor for batch items with no targets - always use same device and dtype
                    targets_list.append(torch.zeros((0, 8), dtype=torch.float32, device=device))

            # Forward pass with optional AMP
            use_amp_this_batch = config.get('training.use_amp', False) and device.type == 'cuda'

            forward_start = time.time()
            loss = None  # Initialize loss to None
            loss_dict = None
            try:
                with torch.amp.autocast('cuda', enabled=use_amp_this_batch):
                    outputs = model(events)

                if isinstance(outputs, tuple):
                    predictions, track_features = outputs
                else:
                    predictions = outputs
                    track_features = None

                # DEBUG: Check model output for NaN/Inf BEFORE loss computation
                if isinstance(predictions, list):
                    for i, pred in enumerate(predictions):
                        if torch.isnan(pred).any() or torch.isinf(pred).any():
                            logger.error(f"ERROR: NaN/Inf in model output predictions[{i}]!")
                            logger.error(f"  Shape: {pred.shape}, NaN count: {torch.isnan(pred).sum()}, Inf count: {torch.isinf(pred).sum()}")
                            logger.error(f"  Min: {pred.min():.4f}, Max: {pred.max():.4f}, Mean: {pred.mean():.4f}")
                elif isinstance(predictions, torch.Tensor):
                    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                        logger.error(f"ERROR: NaN/Inf in model output predictions!")
                        logger.error(f"  Shape: {predictions.shape}, NaN count: {torch.isnan(predictions).sum()}, Inf count: {torch.isinf(predictions).sum()}")
                        logger.error(f"  Min: {predictions.min():.4f}, Max: {predictions.max():.4f}, Mean: {predictions.mean():.4f}")

                # Compute loss with temporal information and current epoch for adaptive IoU
                loss_dict = loss_fn(predictions, targets_list, track_features, event_timestamps=event_timestamps, current_epoch=epoch)
                # Use the weighted total loss, not sum of all values (which would double-count)
                loss = loss_dict['total_loss']

                # Normalize loss by accumulation steps (to simulate larger batch size)
                loss = loss / accumulation_steps
            except Exception as forward_error:
                # If forward pass or loss computation failed, skip this batch
                logger.error(f"Error in forward pass or loss computation at batch {batch_idx}: {forward_error}")
                logger.error(f"Forward pass traceback: {traceback.format_exc()}")
                # Clear CUDA cache if OOM error
                if "out of memory" in str(forward_error).lower() or "CUDA" in str(forward_error):
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        logger.warning("Cleared CUDA cache after OOM error in forward pass")
                
                # CRITICAL: Delete batch tensors even on forward error to free memory
                try:
                    if 'events' in locals():
                        del events
                    if 'targets' in locals():
                        del targets
                    if 'targets_list' in locals():
                        del targets_list
                    if 'batch' in locals():
                        del batch
                except NameError:
                    pass
                import gc
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                batch_start_time = time.time()
                continue

            # Check if loss was successfully computed
            if loss is None or loss_dict is None:
                logger.warning(f"Loss not computed for batch {batch_idx}, skipping...")
                batch_start_time = time.time()
                continue

            forward_time = time.time() - forward_start
            total_forward_time += forward_time

            # Check for NaN losses
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected in batch {batch_idx}, skipping...")
                batch_start_time = time.time()
                continue

            # Backward pass (accumulate gradients) with optional AMP scaling
            backward_start = time.time()
            if use_amp_this_batch:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            backward_time = time.time() - backward_start
            total_backward_time += backward_time

            # Update statistics (multiply by accumulation_steps to account for normalization)
            total_loss += loss.item() * accumulation_steps
            total_samples += len(events)

            # Perform optimizer step every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer_start = time.time()
                # Gradient clipping for training stability
                # Compute gradient norm before clipping (for monitoring)
                total_norm_before = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_before += param_norm.item() ** 2
                total_norm_before = total_norm_before ** 0.5

                if use_amp_this_batch:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Compute gradient norm after clipping (for monitoring)
                total_norm_after = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_after += param_norm.item() ** 2
                total_norm_after = total_norm_after ** 0.5

                # Log gradient norms occasionally (every 50 batches)Are we making the frame from events after loading the data into the gpu or before that ?
                if batch_idx % 50 == 0:
                    logger.info(f"  Gradient norms: before={total_norm_before:.4f}, after={total_norm_after:.4f}, "
                               f"clipped={total_norm_before > 1.0}")

                # Update model parameters
                if use_amp_this_batch:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # For OneCycleLR (cyclic scheduler), step once per optimizer step
                if scheduler is not None and getattr(config, '_is_cyclic', False):
                    scheduler.step()

                num_optim_steps += 1

                optimizer.zero_grad()  # Zero gradients for next accumulation cycle
                optimizer_time = time.time() - optimizer_start
                total_optimizer_time += optimizer_time
                
                # CRITICAL: Clear CUDA cache after optimizer step to free memory
                # This helps prevent memory fragmentation
                if device.type == 'cuda' and batch_idx % 5 == 0:  # Every 5 optimizer steps
                    torch.cuda.empty_cache()

            # Log progress (only if loss was successfully computed)
            if batch_idx % config.get('logging.print_every', 50) == 0 and 'loss' in locals() and 'loss_dict' in locals():
                # Get unique filenames in this batch
                filenames = batch.get('filenames', ['unknown'])
                unique_files = list(set(filenames))
                files_str = ', '.join(unique_files[:3])  # Show first 3 files
                if len(unique_files) > 3:
                    files_str += f" (+{len(unique_files)-3} more)"

                # Display unnormalized loss for logging (multiply by accumulation_steps)
                unnormalized_loss = loss.item() * accumulation_steps
                # Get detailed tracking loss breakdown if available
                track_loss = loss_dict.get('track_loss', 0)
                track_loss_details = getattr(loss_fn, '_last_track_loss_details', None)
                if track_loss_details and isinstance(track_loss_details, dict):
                    num_matched = track_loss_details.get('num_matched_anchors', 0)
                    num_track_ids = track_loss_details.get('num_unique_track_ids', 0)
                    track_loss_str = f"Track Loss = {track_loss:.6f} (Pos: {track_loss_details.get('positive', 0.0):.6f}, Neg: {track_loss_details.get('negative', 0.0):.6f}, Pairs: {track_loss_details.get('num_positive_pairs', 0)}+/{track_loss_details.get('num_negative_pairs', 0)}-, Matched: {num_matched}, TrackIDs: {num_track_ids})"
                else:
                    # Debug: log why details aren't available
                    if batch_idx == 0 and epoch % 10 == 0:  # Log occasionally for debugging
                        logger.debug(f"Track loss details not available: track_loss_details={track_loss_details}, hasattr={hasattr(loss_fn, '_last_track_loss_details')}")
                    track_loss_str = f"Track Loss = {track_loss:.6f}"

                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: "
                    f"Total Loss = {unnormalized_loss:.6f}, "
                    f"Box Loss = {loss_dict.get('box_loss', 0):.6f}, "
                    f"Class Loss = {loss_dict.get('cls_loss', 0):.6f}, "
                    f"{track_loss_str} | "
                    f"Accumulation: {(batch_idx + 1) % accumulation_steps}/{accumulation_steps} | "
                    f"Files: {files_str}"
                )
                # Force flush log file after each log message to ensure real-time logging
                root_logger = logging.getLogger()
                if hasattr(root_logger, '_file_handler'):
                    root_logger._file_handler.flush()

            # CRITICAL: Explicitly delete batch tensors to free memory immediately
            # This prevents CUDA memory accumulation across batches
            try:
                del events, targets, targets_list
            except NameError:
                pass
            try:
                if 'outputs' in locals():
                    del outputs
                if 'predictions' in locals():
                    del predictions
                if 'track_features' in locals():
                    del track_features
                if 'loss_dict' in locals():
                    del loss_dict
                if 'event_timestamps' in locals():
                    del event_timestamps
            except NameError:
                pass
            # Delete the entire batch dictionary to free all references
            try:
                del batch
            except NameError:
                pass
            
            # Force garbage collection every N batches to free memory
            if batch_idx % 10 == 0:  # Every 10 batches
                import gc
                gc.collect()
                # Clear CUDA cache periodically to prevent fragmentation
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Reset timer for next batch (measure data loading time)
            batch_start_time = time.time()

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Clear CUDA cache if OOM error
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    logger.warning("Cleared CUDA cache after OOM error")
            
            # CRITICAL: Explicitly delete batch tensors even on error to free memory
            try:
                if 'events' in locals():
                    del events
                if 'targets' in locals():
                    del targets
                if 'targets_list' in locals():
                    del targets_list
                if 'batch' in locals():
                    del batch
            except NameError:
                pass
            import gc
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            batch_start_time = time.time()
            continue

    # Handle leftover gradients at the end of epoch (if not evenly divisible by accumulation_steps)
    # Check if we have accumulated gradients that haven't been stepped yet
    has_unstepped_grads = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_unstepped_grads = True
            break

    if has_unstepped_grads:
        logger.info(f"Epoch {epoch}: Applying leftover gradients at end of epoch")
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Update model parameters with accumulated gradients
        optimizer.step()

        # For OneCycleLR (cyclic scheduler), step once more for this optimizer step
        if scheduler is not None and getattr(config, '_is_cyclic', False):
            scheduler.step()

        num_optim_steps += 1
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0

    # Print profiling summary
    total_time = total_data_time + total_forward_time + total_backward_time + total_optimizer_time
    num_batches = len(train_loader)

    logger.info(f"\n{'='*80}")
    logger.info(f"EPOCH {epoch} PROFILING SUMMARY:")
    logger.info(f"{'='*80}")
    logger.info(f"Total batches: {num_batches}")
    logger.info(f"Average loss: {avg_loss:.6f}, Effective batch size: {effective_batch_size}")
    logger.info(f"\nTime breakdown:")
    avg_div = max(1, num_batches)
    logger.info(f"  Data Loading:     {total_data_time:8.2f}s ({(100*total_data_time/total_time if total_time>0 else 0):5.1f}%) | Avg: {total_data_time/avg_div:.3f}s/batch")
    logger.info(f"  Forward Pass:     {total_forward_time:8.2f}s ({(100*total_forward_time/total_time if total_time>0 else 0):5.1f}%) | Avg: {total_forward_time/avg_div:.3f}s/batch")
    logger.info(f"  Backward Pass:    {total_backward_time:8.2f}s ({(100*total_backward_time/total_time if total_time>0 else 0):5.1f}%) | Avg: {total_backward_time/avg_div:.3f}s/batch")
    logger.info(f"  Optimizer Step:   {total_optimizer_time:8.2f}s ({(100*total_optimizer_time/total_time if total_time>0 else 0):5.1f}%) | Avg: {total_optimizer_time/avg_div:.3f}s/batch")
    logger.info(f"  TOTAL:            {total_time:8.2f}s")
    logger.info(f"\nBottleneck Analysis:")
    if total_data_time > total_forward_time + total_backward_time:
        logger.info(f"  ⚠️  DATA LOADING IS THE BOTTLENECK ({100*total_data_time/total_time:.1f}% of time)")
        logger.info(f"      → Increase num_workers, enable prefetch_factor, or reduce max_events_per_sample")
    elif total_forward_time > 2 * total_backward_time:
        logger.info(f"  ⚠️  FORWARD PASS IS SLOW ({100*total_forward_time/total_time:.1f}% of time)")
        logger.info(f"      → Model inference is the bottleneck (expected for complex models)")
    else:
        logger.info(f"  ✓  Reasonably balanced pipeline")
    logger.info(f"{'='*80}\n")

    return avg_loss, num_optim_steps

def validate_epoch(model, val_loader, loss_fn, device, epoch, config):
    """Validate for one epoch.

    IMPORTANT: This function does NOT perform any learning or backpropagation.
    - All operations are wrapped in torch.no_grad() to disable gradient computation
    - No optimizer.step() or loss.backward() is called
    - Model parameters are NOT updated during validation
    - Only loss computation for evaluation purposes
    """
    logger = logging.getLogger(__name__)

    # Use training mode to preserve temporal dimension for loss computation
    # This ensures we get [T, B, H*W, features] instead of averaged [B, H*W, features]
    # We'll manually disable dropout and ensure batch norm uses running stats
    # NOTE: model.train() is used ONLY to preserve temporal dimension, NOT for learning
    model.train()  # Use train mode to preserve temporal dimension
    # Disable dropout and use eval mode for batch norm
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()  # Use running stats for batch norm

    total_loss = 0.0
    total_samples = 0

    logger.info(f"Starting validation for epoch {epoch} (NO LEARNING - evaluation only)")

    # CRITICAL: torch.no_grad() disables gradient computation entirely
    # This ensures no backpropagation or parameter updates occur
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                logger.info(f"Validation batch {batch_idx}: Starting processing...")

                # Debug: Check batch contents
                logger.info(f"Batch keys: {list(batch.keys())}")
                # Use frames - raw events are no longer included in batch to save memory
                if 'frames' not in batch or batch['frames'] is None:
                    raise ValueError("Frames not found in batch. This should not happen - frames are required for model input.")
                
                logger.info(f"Frames shape: {batch['frames'].shape}")
                events_cpu = batch['frames']  # frames are [B, T, H, W]
                logger.info(f"Targets shape: {batch['targets'].shape}")

                # Check for invalid values in events/frames before device transfer
                if torch.isnan(events_cpu).any():
                    logger.error(f"NaN detected in events at batch {batch_idx}")
                    continue
                if torch.isinf(events_cpu).any():
                    logger.error(f"Inf detected in events at batch {batch_idx}")
                    continue

                # Move data to device (explicitly move to CPU first if device is CPU to avoid CUDA context issues)
                logger.info(f"Moving events to device {device}...")
                try:
                    if device.type == 'cpu':
                        events = events_cpu.cpu().to(device)
                    else:
                        events = events_cpu.to(device)

                    # Convert inputs to FP16 if model is FP16 (direct FP16 training)
                    use_fp16 = config.get('training.use_fp16', False)
                    if use_fp16 and device.type == 'cuda':
                        # Check if model is FP16
                        if next(model.parameters()).dtype == torch.float16:
                            events = events.half()  # Convert inputs to FP16

                    logger.info(f"Events moved successfully")
                except RuntimeError as cuda_error:
                    if "device-side assert" in str(cuda_error):
                        logger.error(f"CUDA device-side assert triggered. The CUDA context is now corrupted.")
                        logger.error(f"Clearing CUDA cache and aborting validation...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        # Abort validation for this epoch
                        break
                    else:
                        raise

                logger.info(f"Moving targets to device {device}...")
                if device.type == 'cpu':
                    targets = batch['targets'].cpu().to(device)  # Move targets to device first
                else:
                    targets = batch['targets'].to(device)  # Move targets to device first
                event_timestamps = batch.get('event_timestamps', None)  # NEW: Get event timestamps
                logger.info(f"Targets moved successfully")

                # Check for invalid values after device transfer
                if torch.isnan(events).any():
                    logger.error(f"NaN detected in events after device transfer at batch {batch_idx}")
                    continue
                if targets.is_floating_point() and torch.isnan(targets).any():
                    logger.error(f"NaN detected in targets after device transfer at batch {batch_idx}")
                    continue
                if targets.is_floating_point() and torch.isinf(targets).any():
                    logger.error(f"Inf detected in targets after device transfer at batch {batch_idx}")
                    continue

                # Convert targets tensor to list of tensors for each batch item
                targets_list = []
                for i in range(targets.shape[0]):
                    target_tensor = targets[i]  # Get targets for batch item i
                    # Filter out zero-padded targets (assuming targets with all zeros are padding)
                    valid_mask = target_tensor.sum(dim=1) != 0
                    if valid_mask.any():
                        targets_list.append(target_tensor[valid_mask].to(device))
                    else:
                        # Create empty tensor for batch items with no targets
                        targets_list.append(torch.zeros((0, 8), dtype=torch.float32, device=device))  # explicit dtype

                # Forward pass with optional AMP (inference mode)
                logger.info("Starting forward pass...")
                use_amp = config.get('training.use_amp', False) and device.type == 'cuda'
                try:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(events)
                    logger.info("Forward pass completed successfully")
                except Exception as e:
                    logger.error(f"Error in forward pass at batch {batch_idx}: {e}")
                    logger.error(f"Forward pass traceback: {traceback.format_exc()}")
                    continue

                if isinstance(outputs, tuple):
                    predictions, track_features = outputs
                    if isinstance(predictions, list):
                        logger.info(f"Got tuple output: predictions is list of {len(predictions)} scales")
                        # For multi-scale, use first scale for debugging
                        if len(predictions) > 0:
                            logger.info(f"  First scale shape: {predictions[0].shape}")
                    else:
                        logger.info(f"Got tuple output: predictions shape {predictions.shape}")
                else:
                    predictions = outputs
                    track_features = None
                    if isinstance(predictions, list):
                        logger.info(f"Got single output: predictions is list of {len(predictions)} scales")
                    else:
                        logger.info(f"Got single output: predictions shape {predictions.shape}")

                # Reshape predictions if needed (handle different prediction formats)
                # In training mode, we should get [T, B, H*W, features] (4D) or [B, H*W, features] (3D)
                # In eval mode (fallback), we might get [num_anchors, features] (2D)
                if isinstance(predictions, list):
                    # Multi-scale predictions - reshape each scale if needed
                    batch_size = len(targets_list)
                    reshaped_predictions = []
                    for i, pred in enumerate(predictions):
                        if pred.dim() == 2:
                            original_shape = pred.shape
                            pred = pred.unsqueeze(0)  # [1, num_anchors, features]
                            pred = pred.expand(batch_size, -1, -1)  # [B, num_anchors, features]
                            logger.warning(f"Reshaped 2D predictions scale {i} from {original_shape} to {pred.shape} (should be 3D or 4D in training mode)")
                        elif pred.dim() == 3:
                            # 3D predictions [B, num_anchors, features] - check batch size
                            if pred.shape[0] != batch_size:
                                logger.warning(f"Batch size mismatch: predictions {pred.shape[0]} vs targets {batch_size}")
                                if pred.shape[0] == 1:
                                    pred = pred.expand(batch_size, -1, -1)
                                else:
                                    pred = pred[:batch_size]
                        elif pred.dim() == 4:
                            # 4D predictions [T, B, num_anchors, features] - this is correct for training mode
                            # Loss function will handle this format
                            pass
                        reshaped_predictions.append(pred)
                    predictions = reshaped_predictions
                else:
                    # Single prediction tensor
                    if predictions.dim() == 2:
                        original_shape = predictions.shape
                        batch_size = len(targets_list)
                        predictions = predictions.unsqueeze(0)  # [1, num_anchors, features]
                        predictions = predictions.expand(batch_size, -1, -1)  # [B, num_anchors, features]
                        logger.warning(f"Reshaped 2D predictions from {original_shape} to {predictions.shape} (should be 3D or 4D in training mode)")
                    elif predictions.dim() == 4:
                        # 4D predictions [T, B, num_anchors, features] - this is correct for training mode
                        # Loss function will handle this format
                        pass

                # Check for invalid values in predictions
                if isinstance(predictions, list):
                    # Multi-scale predictions - check each scale
                    for i, pred in enumerate(predictions):
                        if torch.isnan(pred).any():
                            logger.error(f"NaN detected in predictions scale {i} at batch {batch_idx}")
                        if torch.isinf(pred).any():
                            logger.error(f"Inf detected in predictions scale {i} at batch {batch_idx}")
                else:
                    if torch.isnan(predictions).any():
                        logger.error(f"NaN detected in predictions at batch {batch_idx}")
                        continue
                    if torch.isinf(predictions).any():
                        logger.error(f"Inf detected in predictions at batch {batch_idx}")
                        continue

                # Compute loss (FOR EVALUATION ONLY - no backpropagation)
                # Loss is computed only to measure model performance, not for training
                logger.info("Computing loss (evaluation only, no backprop)...")
                try:
                    loss_dict = loss_fn(predictions, targets_list, track_features, event_timestamps=event_timestamps, current_epoch=epoch)
                    # Use the weighted total loss, not sum of all values (which would double-count)
                    loss = loss_dict['total_loss']
                    logger.info(f"Loss computed successfully: {loss.item()} (evaluation only)")

                    # Verify that loss tensor has no gradient (should be None due to torch.no_grad())
                    if loss.requires_grad:
                        logger.warning(f"WARNING: Loss tensor requires gradient! This should not happen in validation.")
                        # Force detach to prevent any accidental backpropagation
                        loss = loss.detach()
                except Exception as e:
                    logger.error(f"Error computing loss at batch {batch_idx}: {e}")
                    logger.error(f"Loss computation traceback: {traceback.format_exc()}")
                    continue

                # Check for NaN losses
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected in validation batch {batch_idx}, skipping...")
                    continue

                # Check for model parameter corruption after loss computation
                if batch_idx % 10 == 0:  # Check every 10 batches
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            logger.error(f"NaN detected in model parameter {name} after batch {batch_idx}")
                            return None
                        if torch.isinf(param).any():
                            logger.error(f"Inf detected in model parameter {name} after batch {batch_idx}")
                            return None

                # Update statistics
                total_loss += loss.item()
                total_samples += len(events)

                # Log progress
                if batch_idx % config.get('logging.print_every', 50) == 0:
                    # Get unique filenames in this batch
                    filenames = batch.get('filenames', ['unknown'])
                    unique_files = list(set(filenames))
                    files_str = ', '.join(unique_files[:3])  # Show first 3 files
                    if len(unique_files) > 3:
                        files_str += f" (+{len(unique_files)-3} more)"

                    logger.info(
                        f"Val Epoch {epoch}, Batch {batch_idx}/{len(val_loader)}: "
                        f"Total Loss = {loss.item():.6f}, "
                        f"Box Loss = {loss_dict.get('box_loss', 0):.6f}, "
                        f"Class Loss = {loss_dict.get('cls_loss', 0):.6f}, "
                        f"Track Loss = {loss_dict.get('track_loss', 0):.6f} | "
                        f"Files: {files_str}"
                    )

            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {e}")
                logger.error(f"Batch contents: {batch.keys() if 'batch' in locals() else 'Batch not available'}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue


    # Clear CUDA cache after validation to free memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache after validation")

    if len(val_loader) == 0:
        logger.warning(f"Validation loader is empty! Cannot compute validation loss.")
        logger.warning(f"This usually means validation folders don't have 3-class annotation files or data loading failed.")
        # Return None to indicate invalid validation
        return None

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    logger.info(f"Epoch {epoch} validation completed. Average loss: {avg_loss:.6f}")

    # Force flush log file
    root_logger = logging.getLogger()
    if hasattr(root_logger, '_file_handler'):
        root_logger._file_handler.flush()

    return avg_loss

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss,
                   checkpoint_dir: str, config: Config, is_best: bool = False):
    """Save model checkpoint with CUDA tensor validation."""
    logger = logging.getLogger(__name__)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get model state dict with safer approach
    logger.info("Getting model state dict...")
    try:
        # FIX: Use model.state_dict() instead of manually collecting parameters
        # This ensures all parameters and buffers (e.g., BatchNorm running stats) are included
        model_state = model.state_dict()

        # Validate state dict for NaN/Inf values
        for name, param in model_state.items():
            if isinstance(param, torch.Tensor) and param.is_floating_point():
                if torch.isnan(param).any():
                    logger.error(f"NaN detected in model parameter {name}, skipping checkpoint")
                    return None
                if torch.isinf(param).any():
                    logger.error(f"Inf detected in model parameter {name}, skipping checkpoint")
                    return None
        
        logger.info(f"Model state dict created successfully with {len(model_state)} parameters/buffers")
        
    except Exception as e:
        logger.error(f"Error getting model state: {e}")
        logger.error(f"Model state traceback: {traceback.format_exc()}")
        return None
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat(),
        'amp_enabled': config.get('training.use_amp', False)  # Track AMP state for compatibility
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        # Force flush log file after checkpoint save
        root_logger = logging.getLogger()
        if hasattr(root_logger, '_file_handler'):
            root_logger._file_handler.flush()
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        logger.error(f"Checkpoint save traceback: {traceback.format_exc()}")
        return None
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        try:
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
        except Exception as e:
            logger.error(f"Error saving best model: {e}")
    
    return checkpoint_path

def main():
    """Main training function."""
    # Load configuration first to get default values
    config = Config('config/config.yaml')
    
    parser = argparse.ArgumentParser(description='Comprehensive Traffic_Monitoring SpikeYOLO Training')
    
    # Use config values as defaults
    parser.add_argument('--epochs', type=int, 
                       default=config.get('training.epochs', 10), 
                       help='Number of training epochs (default: from config.yaml)')
    parser.add_argument('--batch_size', type=int, 
                       default=config.get('training.batch_size', 25), 
                       help='Batch size (default: from config.yaml)')
    parser.add_argument('--learning_rate', type=float, 
                       default=config.get('training.learning_rate', 0.001), 
                       help='Learning rate (default: from config.yaml)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set up logging
    # Get log file name and format from config if specified
    log_file_name = config.get('model.log_file_name', None)
    log_format = config.get('logging.format', None)
    logger, log_file = setup_logging(config.get_logs_dir(), args.log_level, log_file_name=log_file_name, log_format=log_format)
    
    logger.info("="*80)
    logger.info("TRAFFIC_MONITORING SPIKEYOLO COMPREHENSIVE TRAINING")
    logger.info("="*80)
    logger.info(f"Configuration loaded from: {config.config_path}")
    logger.info(f"Environment: vm")  # We're running on VM
    logger.info(f"Data root: {config.get_data_root()}")
    logger.info(f"Annotation dir: {config.get_annotation_dir()}")
    logger.info(f"Checkpoint dir: {config.get_checkpoint_dir()}")
    logger.info(f"Checkpoint file path: {config.get_checkpoint_path()}")
    # Log checkpoint frequency
    checkpoint_frequency = config.get('training.checkpoint_frequency', None)
    if checkpoint_frequency is None:
        checkpoint_frequency = 5
        logger.info(f"Checkpoint frequency: {checkpoint_frequency} (default, not set in config)")
    else:
        logger.info(f"Checkpoint frequency: {checkpoint_frequency} (from config)")
    
    # Set device
    # ONLY use CPU if force_cpu is explicitly set to true in config
    force_cpu = config.get('device.force_cpu', False)
    if force_cpu:
        # force_cpu is true - use CPU regardless of args.device
        device = torch.device('cpu')
        logger.info("force_cpu is set to true in config, forcing CPU usage (ignoring --device argument)")
        # Set default tensor type to CPU to prevent automatic CUDA tensor creation
        torch.set_default_tensor_type('torch.FloatTensor')
        # Disable CUDA initialization (os is already imported at top of file)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("Disabled CUDA and set default tensor type to CPU")
    elif args.device == 'auto':
        # Normal auto detection - use CUDA if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device auto-detection: {device}")
    else:
        # Use device specified in args (could be 'cpu' or 'cuda')
        device = torch.device(args.device)
        logger.info(f"Using device from command line argument: {device}")
    
    logger.info(f"Final device selection: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif device.type == 'cpu' and not force_cpu:
        logger.warning("Running on CPU - this may be slow. Set device.force_cpu=true in config/config.yaml to explicitly force CPU, or use --device cuda to use GPU if available.")
    
    # Create model
    model = create_model(config, device)
    
    # Create data loaders first (needed for OneCycleLR steps_per_epoch calculation)
    logger.info("Creating data loaders...")
    
    # When force_cpu is True, we can use workers but need to ensure all tensors are on CPU
    # Using workers will significantly speed up data loading even on CPU
    if force_cpu:
        # Use workers for faster data loading, but ensure all tensors are forced to CPU
        num_workers = config.get('data_processing.num_workers', 4)  # Use workers but limit to 4 for CPU
        logger.info(f"force_cpu=True: Using {num_workers} workers with explicit CPU tensor forcing")
    else:
        num_workers = config.get('data_processing.num_workers', 8)
    
    # Training data loader (all training folders)
    # Get max_annotations_per_class from config (primary control for class balancing)
    max_annotations_per_class = config.get('data_processing.max_annotations_per_class', None)
    
    if max_annotations_per_class is not None:
        logger.info(f"Using max_annotations_per_class={max_annotations_per_class} for class balancing")
    else:
        logger.info(f"Balancing to rarest class's count (max_annotations_per_class not set)")
        if config.get('data_processing.use_class_balanced_sampling', False):
            logger.info(f"Using max_samples_per_file={config.get('data_processing.max_samples_per_file', None)} as fallback (per-file limit)")
    
    train_loader = create_dataloader(
        data_root=config.get_data_root(),
        split='train',
        batch_size=args.batch_size,
        max_events_per_sample=config.get('data_processing.max_events_per_sample', 10000),
        num_workers=num_workers,
        shuffle=True,
        annotation_dir=config.get_annotation_dir(),
        max_samples_per_file=config.get('data_processing.max_samples_per_file', None) if max_annotations_per_class is None else None,
        targeted_training=config.get('data_processing.targeted_training', True),
        force_cpu=force_cpu,  # Pass force_cpu flag to dataloader
        num_classes=config.get_num_classes(),  # Pass number of classes from config
        use_class_balanced_sampling=config.get('data_processing.use_class_balanced_sampling', False),  # Pass class balancing flag
        min_samples_per_class=config.get('data_processing.min_samples_per_class', 1),  # Pass minimum samples per class
        max_annotations_per_class=max_annotations_per_class,  # Pass annotation limit per class
        cache_samples=config.get('data_processing.cache_samples', False),
        preload_all_samples=config.get('data_processing.preload_all_samples', False),  # Pass preloading flag
        debug_sample_loading=config.get('data_processing.debug_sample_loading', False),
        time_steps=config.get_time_steps(),
        image_height=config.get('data_processing.image_height', 720),
        image_width=config.get('data_processing.image_width', 1280),
        config=config,  # Pass config for DataLoader parameters
        time_window_us=config.get_window_us()
    )
    
    # Validation data loader (drop_last=False to keep incomplete batches for validation)
    # Validation uses the same annotation-based approach as training
    # Validation annotation limit can be configured via:
    #   1. max_annotations_per_class_validation (fixed number, takes precedence)
    #   2. validation_annotation_ratio (percentage of training max_annotations_per_class)
    #   3. validation_min_annotations_per_class (minimum when using ratio)
    val_max_annotations_per_class = config.get('data_processing.max_annotations_per_class_validation', None)
    
    if val_max_annotations_per_class is None:
        # Calculate from training annotations using ratio
        if max_annotations_per_class is not None:
            # Get ratio and minimum from config (with hardcoded defaults as fallback)
            validation_ratio = config.get('data_processing.validation_annotation_ratio', 0.2)
            validation_min = config.get('data_processing.validation_min_annotations_per_class', 100)
            raw_val_cap = max(validation_min, int(max_annotations_per_class * validation_ratio))
            val_max_annotations_per_class = min(max_annotations_per_class, raw_val_cap)
            logger.info(
                f"Validation using max_annotations_per_class={val_max_annotations_per_class} "
                f"(ratio={validation_ratio*100:.0f}% of training limit={max_annotations_per_class}, "
                f"minimum={validation_min}, capped at training limit)"
            )
        else:
            logger.info(f"Validation balancing to rarest class's count (max_annotations_per_class not set)")
            if config.get('data_processing.use_class_balanced_sampling', False):
                logger.info(f"Validation using max_samples_per_file={config.get('data_processing.max_samples_per_file', None)} as fallback (per-file limit)")
    else:
        logger.info(f"Validation using max_annotations_per_class={val_max_annotations_per_class} (fixed value from config)")
    
    if config.get('data_processing.use_class_balanced_sampling', False):
        logger.info(f"Validation will use class-balanced sampling with file diversity to ensure spread across files")
    
    val_loader = create_dataloader(
        data_root=config.get_data_root(),
        split='val',
        batch_size=args.batch_size,
        max_events_per_sample=config.get('data_processing.max_events_per_sample', 10000),
        num_workers=num_workers,
        shuffle=False,
        annotation_dir=config.get_annotation_dir(),
        max_samples_per_file=config.get('data_processing.max_samples_per_file', None) if val_max_annotations_per_class is None else None,
        targeted_training=config.get('data_processing.targeted_training', True),
        force_cpu=force_cpu,  # Pass force_cpu flag to dataloader
        num_classes=config.get_num_classes(),  # Pass number of classes from config
        drop_last=False,  # Don't drop incomplete batches for validation
        use_class_balanced_sampling=config.get('data_processing.use_class_balanced_sampling', False),  # Enable class balance + file diversity
        min_samples_per_class=config.get('data_processing.min_samples_per_class', 1),  # Pass minimum samples per class
        max_annotations_per_class=val_max_annotations_per_class,  # Pass annotation limit per class for validation
        cache_samples=config.get('data_processing.cache_samples', False),
        preload_all_samples=config.get('data_processing.preload_all_samples', False),  # Pass preloading flag (validation too)
        debug_sample_loading=config.get('data_processing.debug_sample_loading', False),
        time_steps=config.get_time_steps(),
        image_height=config.get('data_processing.image_height', 720),
        config=config,  # Pass config for DataLoader parameters
        image_width=config.get('data_processing.image_width', 1280),
        time_window_us=config.get_window_us()
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create optimizer and scheduler (now with train_loader for OneCycleLR)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, train_loader=train_loader)
    
    # Create loss function with config parameters
    loss_config = config.get('yolo_loss', {})
    # Get image dimensions from config (default: 1280x720 for event camera)
    image_width = config.get('data_processing.image_width', 1280.0)
    image_height = config.get('data_processing.image_height', 720.0)
    
    # Calculate class weights if enabled
    class_weights = None
    effective_weight_scale = 1.0  # Initialize to 1.0 (no scaling when weights not used)
    use_class_weights = config.get('yolo_loss.use_class_weights', False)
    use_class_balanced_sampling = config.get('data_processing.use_class_balanced_sampling', False)
    
    # Skip class weight calculation if we're using class-balanced sampling
    # Class-balanced sampling already ensures balanced data, so weights are redundant
    if use_class_balanced_sampling:
        logger.info("Class-balanced sampling is enabled. Skipping class weight calculation.")
        logger.info("Data is already balanced during loading, so class weights are not needed.")
        class_weights = None
        effective_weight_scale = 1.0
        use_class_weights = False
    elif use_class_weights:
        logger.info("Calculating class weights from training data...")
        try:
            from calculate_class_weights import calculate_class_frequencies_from_loader, calculate_class_weights_from_frequencies
            
            # Calculate frequencies from a sample of the training data
            num_classes = config.get_num_classes()
            
            logger.info(f"Calculating class weights for {num_classes} classes")
            
            max_samples_for_weights = config.get('yolo_loss.class_weight_calculation_samples', 5000)
            class_counts, total_samples = calculate_class_frequencies_from_loader(
                train_loader, 
                num_classes=num_classes,
                max_samples=max_samples_for_weights
            )
            
            # Check if we have any samples
            if total_samples == 0:
                logger.warning("No samples found for class weight calculation. Using uniform weights.")
                class_weights = None
                effective_weight_scale = 1.0
            else:
                # Calculate comprehensive balance metrics across all classes
                class_counts_array = class_counts.numpy()
                non_zero_counts = class_counts_array[class_counts_array > 0]
                
                # Initialize effective_weight_scale
                effective_weight_scale = 1.0
                
                if len(non_zero_counts) > 1:
                    # Metric 1: Min/Max ratio (captures worst-case imbalance)
                    min_count = non_zero_counts.min()
                    max_count = non_zero_counts.max()
                    min_max_ratio = min_count / max_count if max_count > 0 else 0
                    
                    # Metric 2: Coefficient of Variation (CV) - measures spread across ALL classes
                    # CV = std / mean, higher CV = more imbalanced
                    mean_count = non_zero_counts.mean()
                    std_count = non_zero_counts.std()
                    cv = std_count / mean_count if mean_count > 0 else float('inf')
                    
                    # Metric 3: Normalized balance ratio (considers all classes)
                    # Measures how close each class is to the mean
                    # Lower values = more balanced, Higher values = more imbalanced
                    deviations = np.abs(non_zero_counts - mean_count)
                    max_deviation = deviations.max()
                    normalized_imbalance = max_deviation / mean_count if mean_count > 0 else 1.0
                    
                    # Combined imbalance score (0.0 = perfect balance, 1.0 = extreme imbalance)
                    # Weighted combination: CV (40%), normalized_imbalance (30%), min_max_ratio inverse (30%)
                    # Normalize CV to 0-1 range (assuming CV typically ranges 0-2 for real data)
                    cv_normalized = min(cv / 2.0, 1.0)  # Cap at 1.0
                    min_max_normalized = 1.0 - min_max_ratio  # Inverse: 0 = balanced, 1 = imbalanced
                    
                    imbalance_score = 0.4 * cv_normalized + 0.3 * normalized_imbalance + 0.3 * min_max_normalized
                    imbalance_score = min(imbalance_score, 1.0)  # Cap at 1.0
                    
                    # Calculate balance ratio (for backward compatibility and logging)
                    balance_ratio = min_max_ratio
                    
                    logger.info(f"Class frequencies: {dict(enumerate(class_counts_array))}")
                    logger.info(f"Balance metrics:")
                    logger.info(f"  - Min/Max ratio: {min_max_ratio:.3f} (1.0 = perfect)")
                    logger.info(f"  - Coefficient of Variation (CV): {cv:.3f} (lower = better)")
                    logger.info(f"  - Normalized imbalance: {normalized_imbalance:.3f} (lower = better)")
                    logger.info(f"  - Combined imbalance score: {imbalance_score:.3f} (0.0 = perfect, 1.0 = extreme)")
                    
                    # Determine if class weights are needed based on imbalance score
                    # Threshold: 0.3 = moderate imbalance, use weights proportionally
                    if imbalance_score < 0.2:
                        # Excellent balance (< 20% imbalance) - no weights needed
                        logger.info(f"Excellent class balance (imbalance score: {imbalance_score:.3f}). "
                                  f"Class weights will NOT be used - training data is already balanced.")
                        class_weights = None
                        effective_weight_scale = 1.0
                    elif imbalance_score < 0.5:
                        # Moderate imbalance (20-50%) - use partial weights
                        weight_scale = (imbalance_score - 0.2) / 0.3  # Scale from 0 to 1 as imbalance increases
                        logger.info(f"Moderate class imbalance (imbalance score: {imbalance_score:.3f}). "
                                  f"Using partial class weights (scale: {weight_scale:.3f}).")
                        
                        # Calculate full weights first
                        full_weights = calculate_class_weights_from_frequencies(class_counts, method='balanced')
                        
                        # Interpolate between uniform weights (1.0) and full weights
                        uniform_weights = torch.ones(num_classes, dtype=torch.float32)
                        class_weights = uniform_weights * (1.0 - weight_scale) + full_weights * weight_scale
                        
                        # Normalize to maintain average weight = 1.0
                        weights_mean = class_weights.mean()
                        if weights_mean > 0:
                            class_weights = class_weights / weights_mean * num_classes
                        
                        effective_weight_scale = class_weights.mean() / num_classes if class_weights.mean() > 0 else 1.0
                        logger.info(f"Partial class weights (interpolated): {dict(enumerate(class_weights.numpy()))}")
                    else:
                        # Severe imbalance (> 50%) - use full weights
                        logger.warning(f"Severe class imbalance (imbalance score: {imbalance_score:.3f}). "
                                     f"Using full class weights to compensate.")
                        class_weights = calculate_class_weights_from_frequencies(class_counts, method='balanced')
                        effective_weight_scale = class_weights.mean() / num_classes if class_weights.mean() > 0 else 1.0
                        logger.info(f"Full class weights calculated: {dict(enumerate(class_weights.numpy()))}")
                    
                else:
                    # Only one class has samples - use weights
                    logger.warning("Only one class has samples. Using class weights.")
                    class_weights = calculate_class_weights_from_frequencies(class_counts, method='balanced')
                    logger.info(f"Class weights: {dict(enumerate(class_weights.numpy()))}")
                    # Calculate effective weight scale
                    avg_weight = class_weights.mean().item()
                    effective_weight_scale = avg_weight / num_classes if avg_weight > 0 else 1.0
                
                # Validate that class counts match expected number of classes
                if len(class_counts) != num_classes:
                    logger.warning(f"Class counts length ({len(class_counts)}) doesn't match expected num_classes ({num_classes}). Using uniform weights.")
                    class_weights = None
                    effective_weight_scale = 1.0
                # Validate that we have at least one non-zero class
                elif class_counts.sum() == 0:
                    logger.warning("All class counts are zero. Using uniform weights.")
                    class_weights = None
                    effective_weight_scale = 1.0
                else:
                    # Verify all expected classes are present (at least in counts, even if zero)
                    found_classes = set(i for i, count in enumerate(class_counts.numpy()) if count > 0)
                    expected_classes = set(range(num_classes))
                    missing_classes = expected_classes - found_classes
                    if missing_classes:
                        logger.warning(f"Some classes have zero samples in the training data: {sorted(missing_classes)}.")
                        if class_weights is not None:
                            logger.info("Class weights will compensate for missing classes.")
                    
                    # Ensure effective_weight_scale is set if class_weights exists
                    if class_weights is not None and effective_weight_scale == 1.0:
                        # Recalculate if not already set
                        avg_weight = class_weights.mean().item()
                        effective_weight_scale = avg_weight / num_classes if avg_weight > 0 else 1.0
        except Exception as e:
            logger.warning(f"Failed to calculate class weights: {e}. Using uniform weights.")
            class_weights = None
            effective_weight_scale = 1.0
    
    # Adjust cls_loss_weight based on effective class weight scale
    # The effective_weight_scale was calculated above based on the imbalance ratio
    # If class weights are used, we need to compensate proportionally
    # Original optimal cls_loss_weight (14.370) was tuned WITHOUT class weights
    # When class weights are applied, effective loss is multiplied by average weight
    original_cls_loss_weight = loss_config.get('cls_loss_weight', 14.370)
    
    if class_weights is not None and effective_weight_scale > 1.0:
        # Class weights ARE being used - adjust cls_loss_weight inversely proportional to weight scale
        # If effective_weight_scale = 3.0, we need to reduce cls_loss_weight by ~3x
        # This compensates for the multiplication by class weights in the loss calculation
        adjusted_cls_loss_weight = original_cls_loss_weight / effective_weight_scale
        cls_loss_weight = adjusted_cls_loss_weight
        logger.info(f"Class weights are being used (effective scale: {effective_weight_scale:.3f}). "
                   f"Adjusted cls_loss_weight: {cls_loss_weight:.3f} (from {original_cls_loss_weight:.3f})")
    else:
        # Class weights are NOT being used or have no effect (good balance) - use original optimal value
        cls_loss_weight = original_cls_loss_weight
        logger.info(f"Class weights are NOT being used (effective scale: {effective_weight_scale:.3f}). "
                   f"Using original cls_loss_weight: {cls_loss_weight:.3f}")
    
    # Multi-scale loss weights (P5, P4, P3)
    scale_weights = loss_config.get('scale_weights', None)
    if scale_weights:
        logger.info(f"Using custom multi-scale weights: P5={scale_weights[0]:.2f}, P4={scale_weights[1]:.2f}, P3={scale_weights[2]:.2f}")
    else:
        logger.info("Using default multi-scale weights: P5=0.40, P4=0.35, P3=0.25 (favoring finer scales)")
    
    loss_fn = YOLOLoss(
        num_classes=config.get_num_classes(),
        box_loss_weight=loss_config.get('box_loss_weight', 5.0),
        cls_loss_weight=cls_loss_weight,  # Use dynamically adjusted value
        track_loss_weight=loss_config.get('track_loss_weight', 0.5),
        iou_threshold=loss_config.get('iou_threshold', 0.1),
        device=str(device),
        use_focal_loss=loss_config.get('use_focal_loss', True),
        focal_alpha=loss_config.get('focal_alpha', 0.25),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        label_smoothing=loss_config.get('label_smoothing', 0.1),
        use_adaptive_iou=loss_config.get('use_adaptive_iou', False),
        adaptive_iou_start=loss_config.get('adaptive_iou_start', 0.05),
        adaptive_iou_epochs=loss_config.get('adaptive_iou_epochs', 50),
        image_width=image_width,
        image_height=image_height,
        class_weights=class_weights,  # Pass class weights to handle imbalanced data
        scale_weights=scale_weights,  # Pass multi-scale weights
        box_loss_type=loss_config.get('box_loss_type', 'ciou')  # Use CIoU by default for better localization
    )
    
    
    # Apply initial pruning if configured
    pruning_config = config.get('pruning', {})
    if pruning_config.get('enabled', False):
        model = apply_pruning(model, pruning_config)
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    # Initialize best_val_loss to infinity, but only use if we have valid validation
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_epoch = 1
    
    # Check if validation loader is empty and warn
    if len(val_loader) == 0:
        logger.warning("="*80)
        logger.warning("WARNING: Validation loader is empty (0 batches)!")
        logger.warning("This means validation cannot be performed properly.")
        logger.warning("Possible causes:")
        logger.warning("  1. Validation folders don't have 3-class annotation files (*_bbox.npy)")
        logger.warning("  2. Validation data loading is failing")
        logger.warning("  3. No annotations found matching the time windows")
        logger.warning("="*80)
    
    # Check if we should resume from checkpoint
    # Priority: --resume argument > config training.checkpoint_file
    checkpoint_file = args.resume if args.resume else config.get('training.checkpoint_file', None)
    if checkpoint_file and os.path.exists(checkpoint_file):
        logger.info(f"Resuming from checkpoint: {checkpoint_file}")
        # Always load to CPU first if force_cpu, then move to device
        map_location = 'cpu' if force_cpu else device
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        
        # Initialize tracking variables for architecture changes
        incompatible_keys = []
        missing_keys = []
        unexpected_keys = []
        
        # Load model state with compatibility handling for architecture changes
        if 'model_state_dict' in checkpoint:
            checkpoint_state = checkpoint['model_state_dict']
            model_state = model.state_dict()
            
            # Filter checkpoint state to only include compatible keys
            filtered_state = {}
            for key, value in checkpoint_state.items():
                # Skip cv5 branch if it exists (removed track_id prediction branch)
                if 'cv5' in key:
                    incompatible_keys.append(key)
                    logger.warning(f"Skipping incompatible key (removed cv5 branch): {key}")
                    continue
                
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
                    unexpected_keys.append(key)
                    logger.warning(f"Skipping unexpected key (not in current model): {key}")
            
            # Find missing keys (new parameters in current model)
            for key in model_state.keys():
                if key not in checkpoint_state:
                    missing_keys.append(key)
            
            # Load filtered state dict
            if filtered_state:
                model.load_state_dict(filtered_state, strict=False)
                logger.info(f"Model state loaded from checkpoint:")
                logger.info(f"  Loaded {len(filtered_state)} compatible parameters")
                if incompatible_keys:
                    logger.warning(f"  Skipped {len(incompatible_keys)} incompatible parameters (architecture changes)")
                if missing_keys:
                    logger.info(f"  {len(missing_keys)} new parameters initialized (architecture changes)")
                if unexpected_keys:
                    logger.warning(f"  {len(unexpected_keys)} unexpected keys in checkpoint (ignored)")
            else:
                logger.warning("No compatible parameters found in checkpoint. Initializing model from scratch.")
            
            # After loading, ensure model is on CPU if force_cpu
            if force_cpu:
                model = model.cpu()
                # Verify all parameters are on CPU
                for name, param in model.named_parameters():
                    if param.is_cuda:
                        param.data = param.data.cpu()
            
            # Log architecture change summary
            if incompatible_keys or missing_keys:
                logger.info("=" * 80)
                logger.info("ARCHITECTURE CHANGES DETECTED:")
                logger.info("  The model architecture has changed since the checkpoint was saved.")
                logger.info("  Compatible parameters were loaded, new parameters were initialized.")
                logger.info("  Training will continue with the updated architecture.")
                logger.info("=" * 80)
        
        # Load optimizer state (only if optimizer types match and AMP settings compatible)
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            try:
                # Get current optimizer type from config
                current_optimizer_type = config.get('training.optimizer', 'sgd').lower()
                
                # Check if AMP is enabled now but wasn't when checkpoint was saved
                current_amp_enabled = config.get('training.use_amp', False) and device.type == 'cuda'
                checkpoint_amp_enabled = checkpoint.get('amp_enabled', False)  # Default False for old checkpoints
                
                # Check if architecture changed (incompatible keys found)
                architecture_changed = len(incompatible_keys) > 0 or len(missing_keys) > 0
                
                # If architecture changed, skip optimizer loading (it may reference removed parameters)
                if architecture_changed:
                    logger.warning("=" * 80)
                    logger.warning("OPTIMIZER STATE SKIPPED DUE TO ARCHITECTURE CHANGES:")
                    logger.warning(f"  Architecture has changed ({len(incompatible_keys)} removed, {len(missing_keys)} new parameters)")
                    logger.warning(f"  Optimizer state may reference non-existent parameters (e.g., cv5 branch)")
                    logger.warning(f"  Skipping optimizer state loading to avoid errors")
                    logger.warning(f"  Optimizer will be reinitialized with fresh state")
                    logger.warning(f"  Training will continue from epoch {checkpoint.get('epoch', 0) + 1} with new optimizer")
                    logger.warning("=" * 80)
                # Only skip optimizer loading if going from non-AMP to AMP
                # This prevents numerical instability from loading FP32 optimizer state into AMP
                elif current_amp_enabled and not checkpoint_amp_enabled:
                    logger.warning("=" * 80)
                    logger.warning("AMP STATE MISMATCH DETECTED:")
                    logger.warning(f"  Checkpoint was trained WITHOUT AMP (FP32)")
                    logger.warning(f"  Current config has AMP ENABLED (FP16/FP32 mixed)")
                    logger.warning(f"  Skipping optimizer state to avoid numerical instability")
                    logger.warning(f"  Model weights will be loaded, but optimizer will be reinitialized")
                    logger.warning(f"  Training will continue from epoch {checkpoint.get('epoch', 0) + 1} with fresh optimizer momentum")
                    logger.warning("=" * 80)
                else:
                    # Safe to load optimizer state in these cases:
                    # 1. Both non-AMP (FP32 -> FP32)
                    # 2. Both AMP (FP16/32 -> FP16/32)
                    # 3. AMP -> non-AMP (FP16/32 -> FP32) - generally safe
                    # Check if optimizer types match by checking for momentum key
                    # SGD has momentum, AdamW doesn't
                    checkpoint_has_momentum = False
                    if checkpoint['optimizer_state_dict'].get('param_groups'):
                        checkpoint_has_momentum = 'momentum' in checkpoint['optimizer_state_dict']['param_groups'][0]
                    
                    # If checkpoint has momentum, it was saved with SGD
                    # If current optimizer is AdamW, they don't match
                    if checkpoint_has_momentum and current_optimizer_type != 'sgd':
                        logger.warning(f"Checkpoint was saved with SGD optimizer, but current config uses {current_optimizer_type.upper()}. "
                                     f"Skipping optimizer state loading - starting with fresh optimizer state.")
                    elif not checkpoint_has_momentum and current_optimizer_type == 'sgd':
                        logger.warning(f"Checkpoint was saved with AdamW optimizer, but current config uses SGD. "
                                     f"Skipping optimizer state loading - starting with fresh optimizer state.")
                    else:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Optimizer state loaded from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state from checkpoint: {e}. Starting with fresh optimizer state.")
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state loaded from checkpoint")
        
        # Resume from the next epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming from epoch {start_epoch} (checkpoint was at epoch {checkpoint['epoch']})")
            logger.info(f"Next checkpoint will be saved as: checkpoint_epoch_{start_epoch}.pth, checkpoint_epoch_{start_epoch + 1}.pth, ...")
        else:
            logger.warning("No epoch number in checkpoint, starting from epoch 1")
            start_epoch = 1
    
    # Initialize GradScaler for AMP if enabled
    use_amp = config.get('training.use_amp', False) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Automatic Mixed Precision (AMP) enabled - using FP16 for 2-3x speedup")
        logger.info(f"GradScaler initialized for gradient scaling")
    else:
        logger.info("Training in FP32 mode (AMP disabled or CPU training)")
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # Apply warmup learning rate - only if we haven't passed the warmup phase
        warmup_lr = config.get('training.warmup_epochs', 0)
        if warmup_lr > 0 and hasattr(config, '_base_lr'):
            # FIX: Check if we're past warmup epochs - if resuming from epoch > warmup_epochs, skip warmup
            if start_epoch <= warmup_lr and epoch <= warmup_lr:
                # Only apply warmup if we're still in the warmup phase
                warmup_learning_rate(optimizer, epoch, warmup_lr, config._base_lr, start_epoch)
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Warmup: Epoch {epoch}/{warmup_lr}, LR: {current_lr:.6f}")
            elif start_epoch > warmup_lr:
                # Resuming from beyond warmup - use full learning rate
                if epoch == start_epoch:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = config._base_lr
                    logger.info(f"Skipping warmup (resuming from epoch {start_epoch} > warmup_epochs {warmup_lr}). Using full LR: {config._base_lr:.6f}")
            else:
                # We've passed warmup during this training run
                if epoch == warmup_lr + 1:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = config._base_lr
                    logger.info(f"Warmup completed. Using full LR: {config._base_lr:.6f}")
        
        # Train
        train_loss, num_optim_steps = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config, batch_size=args.batch_size, scaler=scaler, scheduler=scheduler)
        
        # Apply iterative pruning
        model = apply_iterative_pruning(model, epoch, args.epochs, pruning_config)
        
        # Validate less frequently for better performance
        validation_frequency = config.get('training.validation_frequency', 1)
        validation_was_run = False
        if epoch % validation_frequency == 0 or epoch == args.epochs:
            val_loss = validate_epoch(model, val_loader, loss_fn, device, epoch, config)
            # If validation failed (empty loader), use train loss as proxy
            if val_loss is None:
                logger.warning(f"Validation failed, using train loss as proxy: {train_loss:.6f}")
                val_loss = train_loss
            else:
                validation_was_run = True  # Validation was actually performed
        else:
            val_loss = train_loss  # Use train loss as proxy
            logger.info(f"Skipping validation for epoch {epoch} (validate every {validation_frequency} epochs)")
        
        # Update scheduler
        if scheduler and num_optim_steps > 0:
            # Get minimum learning rate from config
            min_lr = config.get('training.min_learning_rate', config._base_lr * 0.01)  # Default: 1% of base LR
            
            # For cyclic LR (OneCycleLR), we step per optimizer step inside train_epoch.
            # For other schedulers, step once per epoch here.
            if not (hasattr(config, '_is_cyclic') and config._is_cyclic):
                scheduler.step()
            # Note: OneCycleLR is stepped inside train_epoch per optimizer step
            
            # Get current LR after scheduler step
            current_lr = optimizer.param_groups[0]['lr']
            
            # CRITICAL FIX: Enforce minimum learning rate IMMEDIATELY after scheduler step
            # This prevents LR from decaying to zero
            if current_lr < min_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min_lr
                current_lr = min_lr
                logger.warning(f"Learning rate below minimum ({min_lr:.6f}), resetting to minimum")
            
            # Double-check: Ensure LR is never zero
            if current_lr <= 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min_lr
                current_lr = min_lr
                logger.error(f"Learning rate was zero or negative! Forced to minimum: {min_lr:.6f}")
            
            logger.info(f"Learning rate updated to: {current_lr:.6f} (minimum: {min_lr:.6f})")
        else:
            if scheduler:
                logger.warning(
                    f"Skipping scheduler.step() for epoch {epoch} because no optimizer steps were performed "
                    f"(num_optim_steps={num_optim_steps})."
                )
        # Save checkpoint
        # Only update best_val_loss if validation was actually run and val_loss is valid
        # Skip best model tracking if validation wasn't run or val_loss is 0 (invalid)
        is_best = False
        if validation_was_run and val_loss is not None and val_loss > 0 and val_loss < best_val_loss:
            is_best = True
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.6f}")
        
        # Get checkpoint frequency from config (default: 5 epochs if None or not set)
        checkpoint_frequency = config.get('training.checkpoint_frequency', None)
        if checkpoint_frequency is None:
            checkpoint_frequency = 5
            logger.debug(f"checkpoint_frequency not set in config, using default: {checkpoint_frequency}")
        else:
            logger.debug(f"checkpoint_frequency loaded from config: {checkpoint_frequency}")
        
        # Save checkpoint based on frequency or if it's the best model or last epoch
        should_save_checkpoint = (
            epoch % checkpoint_frequency == 0 or  # Every N epochs
            is_best or  # Best model always saved
            epoch == args.epochs  # Last epoch always saved
        )
        
        if should_save_checkpoint:
            # Get checkpoint directory directly from config
            checkpoint_dir = config.get_checkpoint_dir()
            logger.info(f"Using checkpoint directory: {checkpoint_dir}")
            
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, val_loss,
                checkpoint_dir, config, is_best
            )
        else:
            logger.info(f"Skipping checkpoint save for epoch {epoch} (checkpoint_frequency={checkpoint_frequency})")
            checkpoint_path = None
        
        # Log epoch summary
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        logger.info(f"Best Val Loss: {best_val_loss:.6f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # Training completed
    total_time = time.time() - start_time
    logger.info("="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    logger.info(f"Final train loss: {train_losses[-1]:.6f}")
    logger.info(f"Final val loss: {val_losses[-1]:.6f}")
    logger.info(f"Best val loss: {best_val_loss:.6f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    
    history_file = os.path.join(config.get_logs_dir(), f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training history saved: {history_file}")
    logger.info(f"Log file: {log_file}")

if __name__ == "__main__":
    main()