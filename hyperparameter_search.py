#!/usr/bin/env python3
"""
Hyperparameter Search Script for Traffic_Monitoring SpikeYOLO
Searches optimal hyperparameters using Optuna or grid/random search.
"""

import os
import sys
import time
import logging
import json
import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

import torch
import torch.optim as optim

# Import training components
from config_loader import ConfigLoader as Config
from data_loader import create_ultra_low_memory_dataloader as create_dataloader
from yolo_loss import YOLOLoss
from comprehensive_training import (
    create_model, create_optimizer_and_scheduler, train_epoch, validate_epoch,
    save_checkpoint, warmup_learning_rate, setup_logging
)

# Try to import Optuna for advanced hyperparameter search
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Using random search instead.")


def setup_logging_for_search(log_dir: str, trial_name: str):
    """Setup logging for a specific trial."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"hyperparameter_search_{trial_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class HyperparameterSearch:
    """Hyperparameter search manager."""
    
    def __init__(self, config_path: str, search_mode: str = 'random', n_trials: int = 20):
        self.config = Config(config_path)
        self.search_mode = search_mode
        self.n_trials = n_trials
        self.results_dir = os.path.join(self.config.get_logs_dir(), 'hyperparameter_search')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.results = []
        self.best_config = None
        self.best_score = float('inf')
        
        self.logger = logging.getLogger(__name__)
    
    def define_search_space(self, trial=None):
        """Define hyperparameter search space."""
        
        if trial is not None and OPTUNA_AVAILABLE:
            # Optuna trial-based search space
            params = {
                # Learning rate
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                
                # Batch size (must be integer)
                'batch_size': trial.suggest_int('batch_size', 8, 32, step=4),
                
                # Optimizer
                'optimizer': trial.suggest_categorical('optimizer', ['sgd', 'adamw']),
                
                # Loss weights with constraints to maintain balance
                'box_loss_weight': trial.suggest_float('box_loss_weight', 5.0, 15.0),
                'cls_loss_weight': trial.suggest_float('cls_loss_weight', 5.0, 15.0),
                'track_loss_weight': trial.suggest_float('track_loss_weight', 0.05, 0.5),
                
                # IoU threshold
                'iou_threshold': trial.suggest_float('iou_threshold', 0.3, 0.6),
                
                # Learning rate scheduler
                'lr_scheduler': trial.suggest_categorical('lr_scheduler', ['step', 'cosine']),
                'lr_decay': trial.suggest_float('lr_decay', 0.3, 0.8),
                
                # Weight decay
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            }
            
            if params['optimizer'] == 'sgd':
                params['momentum'] = trial.suggest_float('momentum', 0.8, 0.95)
            
        else:
            # Random search or grid search space
            import random
            
            params = {
                # Learning rate
                'learning_rate': random.choice([1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
                
                # Batch size
                'batch_size': random.choice([8, 12, 16, 20, 24, 32]),
                
                # Optimizer
                'optimizer': random.choice(['sgd', 'adamw']),
                
                # Loss weights - focus on balanced ranges
                'box_loss_weight': random.uniform(5.0, 15.0),
                'cls_loss_weight': random.uniform(5.0, 15.0),
                'track_loss_weight': random.uniform(0.05, 0.5),
                
                # IoU threshold
                'iou_threshold': random.uniform(0.3, 0.6),
                
                # Learning rate scheduler
                'lr_scheduler': random.choice(['step', 'cosine']),
                'lr_decay': random.uniform(0.3, 0.8),
                
                # Weight decay
                'weight_decay': random.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
            }
            
            if params['optimizer'] == 'sgd':
                params['momentum'] = random.uniform(0.8, 0.95)
        
        return params
    
    def create_trial_config(self, params: Dict[str, Any]) -> Config:
        """Create a modified config with trial parameters."""
        # Create a copy of config
        trial_config = copy.deepcopy(self.config)
        
        # Update training parameters using the 'config' attribute (not '_config')
        trial_config.config['training']['learning_rate'] = params['learning_rate']
        trial_config.config['training']['batch_size'] = params['batch_size']
        trial_config.config['training']['optimizer'] = params['optimizer']
        trial_config.config['training']['lr_scheduler'] = params['lr_scheduler']
        trial_config.config['training']['lr_decay'] = params['lr_decay']
        trial_config.config['training']['weight_decay'] = params['weight_decay']
        
        if params['optimizer'] == 'sgd':
            trial_config.config['training']['momentum'] = params.get('momentum', 0.9)
        
        # Update loss weights
        trial_config.config['yolo_loss']['box_loss_weight'] = params['box_loss_weight']
        trial_config.config['yolo_loss']['cls_loss_weight'] = params['cls_loss_weight']
        trial_config.config['yolo_loss']['track_loss_weight'] = params['track_loss_weight']
        trial_config.config['yolo_loss']['iou_threshold'] = params['iou_threshold']
        
        return trial_config
    
    def run_trial(self, trial_id: int, params: Dict[str, Any], epochs: int = 3, 
                  device: str = 'cuda') -> Dict[str, Any]:
        """Run a single training trial with given hyperparameters."""
        
        trial_name = f"trial_{trial_id:03d}"
        logger = setup_logging_for_search(self.results_dir, trial_name)
        
        logger.info("="*80)
        logger.info(f"TRIAL {trial_id}: {trial_name}")
        logger.info("="*80)
        logger.info(f"Hyperparameters: {json.dumps(params, indent=2)}")
        
        start_time = time.time()
        
        try:
            # Create trial config
            trial_config = self.create_trial_config(params)
            
            # Set device
            if device == 'auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device(device)
            
            logger.info(f"Using device: {device}")
            
            # Log annotation type being used and prepare for data loaders
            use_3_class_annotations = trial_config.get('data_processing.use_3_class_annotations', False)
            num_classes = trial_config.get_num_classes()
            logger.info(f"Using {num_classes} classes ({'3-class' if use_3_class_annotations else '8-class'} annotations)")
            
            # Create model
            model = create_model(trial_config, device)
            
            # Create data loaders with support for both 3-class and 8-class annotations
            force_cpu = trial_config.get('device.force_cpu', False)
            use_class_balanced_sampling = trial_config.get('data_processing.use_class_balanced_sampling', False)
            min_samples_per_class = trial_config.get('data_processing.min_samples_per_class', 1)
            
            train_loader = create_dataloader(
                data_root=trial_config.get_data_root(),
                split='train',
                batch_size=params['batch_size'],
                max_events_per_sample=trial_config.get('data_processing.max_events_per_sample', 10000),
                num_workers=trial_config.get('data_processing.num_workers', 8),
                shuffle=True,
                annotation_dir=trial_config.get_annotation_dir() if not use_3_class_annotations else None,
                max_samples_per_file=trial_config.get('data_processing.max_samples_per_file', None),
                targeted_training=trial_config.get('data_processing.targeted_training', True),
                force_cpu=force_cpu,
                use_3_class_annotations=use_3_class_annotations,
                use_class_balanced_sampling=use_class_balanced_sampling,
                min_samples_per_class=min_samples_per_class
            )
            
            val_loader = create_dataloader(
                data_root=trial_config.get_data_root(),
                split='val',
                batch_size=params['batch_size'],
                max_events_per_sample=trial_config.get('data_processing.max_events_per_sample', 10000),
                num_workers=trial_config.get('data_processing.num_workers', 8),
                shuffle=False,
                annotation_dir=trial_config.get_annotation_dir() if not use_3_class_annotations else None,
                max_samples_per_file=trial_config.get('data_processing.max_samples_per_file', None),
                targeted_training=trial_config.get('data_processing.targeted_training', True),
                force_cpu=force_cpu,
                use_3_class_annotations=use_3_class_annotations,
                use_class_balanced_sampling=use_class_balanced_sampling,
                min_samples_per_class=min_samples_per_class
            )
            
            logger.info(f"Training batches: {len(train_loader)}")
            logger.info(f"Validation batches: {len(val_loader)}")
            
            # Create optimizer and scheduler
            optimizer, scheduler = create_optimizer_and_scheduler(model, trial_config, train_loader=train_loader)
            
            # Create loss function with support for both 3-class and 8-class annotations
            loss_config = trial_config.get('yolo_loss', {})
            
            # Get image dimensions for loss computation
            image_width = trial_config.get('data_processing.image_width', 1280.0)
            image_height = trial_config.get('data_processing.image_height', 720.0)
            
            # Get class weights if enabled (for handling class imbalance)
            class_weights = None
            use_class_weights = loss_config.get('use_class_weights', False)
            if use_class_weights:
                logger.info("Class weighting enabled - will calculate from training data")
                # Note: Class weights will be calculated during training if enabled
                # For hyperparameter search, we can skip this to save time
                # or calculate from a sample of training data
            
            # Multi-scale loss weights (use defaults for search, can be tuned later)
            scale_weights = loss_config.get('scale_weights', None)
            
            loss_fn = YOLOLoss(
                num_classes=trial_config.get_num_classes(),
                box_loss_weight=params['box_loss_weight'],
                cls_loss_weight=params['cls_loss_weight'],
                track_loss_weight=params['track_loss_weight'],
                iou_threshold=params['iou_threshold'],
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
                class_weights=class_weights,  # Pass class weights (None for now during search)
                scale_weights=scale_weights   # Pass multi-scale weights
            )
            
            logger.info(f"Loss function created with {trial_config.get_num_classes()} classes "
                       f"({'3-class' if use_3_class_annotations else '8-class'} annotations)")
            
            # Training loop
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            
            for epoch in range(1, epochs + 1):
                logger.info(f"Epoch {epoch}/{epochs}")
                
                # Warmup
                warmup_epochs = trial_config.get('training.warmup_epochs', 3)
                if warmup_epochs > 0:
                    warmup_learning_rate(optimizer, epoch, warmup_epochs, 
                                       trial_config.get('training.learning_rate', 0.001), start_epoch=1)
                
                # Train
                train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, trial_config)
                train_losses.append(train_loss)
                
                # Validate
                val_loss = validate_epoch(model, val_loader, loss_fn, device, epoch, trial_config)
                val_losses.append(val_loss)
                
                # Update best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # Scheduler step
                if scheduler and not (hasattr(trial_config, '_is_cyclic') and trial_config._is_cyclic):
                    scheduler.step()
                
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Final evaluation score (use best validation loss)
            score = best_val_loss
            
            trial_time = time.time() - start_time
            
            result = {
                'trial_id': trial_id,
                'trial_name': trial_name,
                'params': params,
                'score': score,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_losses[-1] if train_losses else None,
                'final_val_loss': val_losses[-1] if val_losses else None,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'trial_time': trial_time,
                'success': True
            }
            
            logger.info(f"Trial {trial_id} completed in {trial_time:.2f}s")
            logger.info(f"Best validation loss: {best_val_loss:.6f}")
            logger.info("="*80)
            
            return result
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'trial_id': trial_id,
                'trial_name': trial_name,
                'params': params,
                'score': float('inf'),
                'error': str(e),
                'success': False
            }
    
    def run_with_optuna(self, epochs: int = 3, device: str = 'cuda'):
        """Run hyperparameter search using Optuna."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Optuna-based search. Install with: pip install optuna")
        
        study_name = f"traffic_monitoring_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=f"sqlite:///{os.path.join(self.results_dir, f'{study_name}.db')}"
        )
        
        def objective(trial):
            params = self.define_search_space(trial)
            result = self.run_trial(trial.number, params, epochs, device)
            
            if not result['success']:
                return float('inf')
            
            self.results.append(result)
            
            # Update best if this is better
            if result['score'] < self.best_score:
                self.best_score = result['score']
                self.best_config = params.copy()
            
            return result['score']
        
        study.optimize(objective, n_trials=self.n_trials)
        
        # Save study results
        results_file = os.path.join(self.results_dir, f'{study_name}_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'trials': [trial.params for trial in study.trials],
                'values': [trial.value for trial in study.trials]
            }, f, indent=2)
        
        self.logger.info(f"Optuna study completed. Best score: {study.best_value:.6f}")
        self.logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params
    
    def run_random_search(self, epochs: int = 3, device: str = 'cuda'):
        """Run random hyperparameter search."""
        self.logger.info(f"Starting random search with {self.n_trials} trials")
        
        for trial_id in range(self.n_trials):
            params = self.define_search_space()
            result = self.run_trial(trial_id, params, epochs, device)
            self.results.append(result)
            
            # Update best if this is better
            if result['success'] and result['score'] < self.best_score:
                self.best_score = result['score']
                self.best_config = params.copy()
            
            self.logger.info(f"Trial {trial_id}/{self.n_trials} completed. "
                           f"Score: {result['score']:.6f}, Best: {self.best_score:.6f}")
        
        # Save results
        self.save_results()
        
        return self.best_config
    
    def save_results(self):
        """Save search results to file."""
        results_file = os.path.join(self.results_dir, f'search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        results_summary = {
            'search_mode': self.search_mode,
            'n_trials': len(self.results),
            'best_score': self.best_score,
            'best_config': self.best_config,
            'results': self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Also save best config as YAML
        if self.best_config:
            best_config_file = os.path.join(self.results_dir, 'best_config.yaml')
            with open(best_config_file, 'w') as f:
                yaml.dump({'best_hyperparameters': self.best_config}, f, default_flow_style=False)
            
            self.logger.info(f"Best config saved to {best_config_file}")
    
    def print_summary(self):
        """Print summary of search results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        successful_trials = [r for r in self.results if r['success']]
        
        print("\n" + "="*80)
        print("HYPERPARAMETER SEARCH SUMMARY")
        print("="*80)
        print(f"Total trials: {len(self.results)}")
        print(f"Successful trials: {len(successful_trials)}")
        print(f"Failed trials: {len(self.results) - len(successful_trials)}")
        
        if successful_trials:
            scores = [r['score'] for r in successful_trials]
            print(f"\nBest validation loss: {min(scores):.6f}")
            print(f"Worst validation loss: {max(scores):.6f}")
            print(f"Mean validation loss: {sum(scores)/len(scores):.6f}")
            
            print(f"\nBest configuration:")
            print(json.dumps(self.best_config, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for Traffic_Monitoring SpikeYOLO')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to base config file')
    parser.add_argument('--mode', type=str, default='random', 
                       choices=['random', 'optuna'],
                       help='Search mode: random or optuna')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of trials to run')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs per trial (for quick search, use 3-5)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Create search instance
    searcher = HyperparameterSearch(
        config_path=args.config,
        search_mode=args.mode,
        n_trials=args.n_trials
    )
    
    # Run search
    if args.mode == 'optuna' and OPTUNA_AVAILABLE:
        best_params = searcher.run_with_optuna(epochs=args.epochs, device=args.device)
    else:
        if args.mode == 'optuna':
            print("Warning: Optuna not available, falling back to random search")
        best_params = searcher.run_random_search(epochs=args.epochs, device=args.device)
    
    # Print summary
    searcher.print_summary()
    
    print(f"\nBest hyperparameters saved to: {searcher.results_dir}")


if __name__ == "__main__":
    main()

