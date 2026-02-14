"""
Main training script for lung segmentation.

Includes:
- MLflow experiment tracking
- Mixed precision training
- Early stopping
- Checkpointing
- Comprehensive logging
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from datetime import datetime

from src.models import get_model
from src.data.dataset import create_data_loaders
from src.utils.losses import get_loss_function
from src.utils.metrics import calculate_metrics, aggregate_metrics
from src.utils.config import load_config, save_config


class Trainer:
    """
    Training manager for lung segmentation models.
    """
    
    def __init__(self, config, experiment_name=None):
        self.config = config
        self.device = torch.device(config['experiment']['device'])
        self.experiment_name = experiment_name or config['experiment']['name']
        
        # Set random seed for reproducibility
        torch.manual_seed(config['experiment']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['experiment']['seed'])
        
        # Initialize model
        print("\n" + "="*60)
        print("INITIALIZING MODEL")
        print("="*60)
        self.model = get_model(config['model'])
        self.model = self.model.to(self.device)
        print(f"Model: {config['model']['name']}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        
        # Initialize loss function
        self.criterion = get_loss_function(config)
        
        # Initialize optimizer
        optimizer_config = config['training']['optimizer']
        if optimizer_config['type'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_config['type'].lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
        
        # Initialize scheduler
        scheduler_config = config['training']['scheduler']
        if scheduler_config['type'].lower() == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config['type'].lower() == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5
            )
        else:
            self.scheduler = None
        
        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        es_config = config['training']['early_stopping']
        self.early_stopping_enabled = es_config['enabled']
        self.patience = es_config['patience']
        self.min_delta = es_config['min_delta']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Checkpointing
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_metric = -float('inf')
        self.metric_to_track = config['validation']['metric']
        
        # Logging
        self.use_mlflow = config['logging']['use_mlflow']
        if self.use_mlflow:
            mlflow.set_tracking_uri(config['logging']['mlflow']['tracking_uri'])
            mlflow.set_experiment(self.experiment_name)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clipping']['enabled']:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clipping']['max_norm']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                
                if self.config['training']['gradient_clipping']['enabled']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clipping']['max_norm']
                    )
                
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                batch_metrics = calculate_metrics(outputs, masks)
                epoch_metrics.append(batch_metrics)
            
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{batch_metrics['dice']:.4f}"
            })
            
            # Free memory
            del images, masks, outputs, loss
            torch.cuda.empty_cache()
        
        # Aggregate metrics
        avg_loss = epoch_loss / len(train_loader)
        avg_metrics = aggregate_metrics(epoch_metrics)
        
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_metrics = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            # Calculate metrics
            batch_metrics = calculate_metrics(outputs, masks)
            val_metrics.append(batch_metrics)
            
            val_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{batch_metrics['dice']:.4f}"
            })
            
            del images, masks, outputs, loss
            torch.cuda.empty_cache()
        
        avg_loss = val_loss / len(val_loader)
        avg_metrics = aggregate_metrics(val_metrics)
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'{self.experiment_name}_last.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f'{self.experiment_name}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (dice: {metrics['dice']:.4f})")
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Start MLflow run
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params({
                    'model': self.config['model']['name'],
                    'batch_size': self.config['training']['batch_size'],
                    'learning_rate': self.config['training']['optimizer']['learning_rate'],
                    'epochs': self.config['training']['epochs'],
                    'optimizer': self.config['training']['optimizer']['type'],
                    'loss': self.config['loss']['type']
                })
                
                return self._training_loop(train_loader, val_loader)
        else:
            return self._training_loop(train_loader, val_loader)
    
    def _training_loop(self, train_loader, val_loader):
        """Internal training loop"""
        num_epochs = self.config['training']['epochs']
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            if self.use_mlflow:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_dice': train_metrics['dice'],
                    'val_dice': val_metrics['dice'],
                    'val_iou': val_metrics['iou'],
                    'learning_rate': current_lr
                }, step=epoch)
            
            # Check for best model
            metric_value = val_metrics[self.metric_to_track]
            is_best = metric_value > self.best_metric
            if is_best:
                self.best_metric = metric_value
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Early stopping
            if self.early_stopping_enabled:
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                        break
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Best {self.metric_to_track}: {self.best_metric:.4f}")
        
        return self.best_metric


def main():
    parser = argparse.ArgumentParser(description='Train lung segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--experiment_name', type=str, help='Override experiment name')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['optimizer']['learning_rate'] = args.learning_rate
    
    # Create data loaders
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    train_loader, val_loader, test_loader = create_data_loaders(
        config,
        Path(args.data_dir)
    )
    
    # Initialize trainer
    trainer = Trainer(config, args.experiment_name)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
