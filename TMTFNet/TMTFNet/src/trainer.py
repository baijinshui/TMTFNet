"""
Training engine for TMTFNet experiments.
========================================
Supports: classification, forecasting, domain adaptation.
Features: mixed precision (fp16), early stopping, LR scheduling, logging.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
from collections import defaultdict


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Trainer:
    """
    Universal trainer for TMTFNet and baselines.

    Args:
        model: nn.Module
        task: 'classification' or 'forecasting'
        device: torch.device
        lr: learning rate
        weight_decay: L2 regularization
        use_amp: use mixed precision training
        use_domain_adapt: enable domain adaptation loss
        domain_weight: weight for domain adaptation loss
    """
    def __init__(self, model, task='classification', device='cuda',
                 lr=1e-3, weight_decay=1e-4, use_amp=True,
                 use_domain_adapt=False, domain_weight=0.1):
        self.model = model.to(device)
        self.task = task
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.use_domain_adapt = use_domain_adapt
        self.domain_weight = domain_weight

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2)
        self.scaler = GradScaler(enabled=self.use_amp)

        if task == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        self.history = defaultdict(list)

    def _compute_domain_alpha(self, epoch, max_epochs):
        """Progressive domain adaptation strength."""
        p = epoch / max_epochs
        return 2. / (1. + np.exp(-10 * p)) - 1

    def train_epoch(self, loader, epoch=0, max_epochs=50):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        domain_alpha = self._compute_domain_alpha(epoch, max_epochs)

        for batch_idx, (modalities, targets) in enumerate(loader):
            modalities = [m.to(self.device, non_blocking=True) for m in modalities]
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs, aux = self.model(modalities, domain_alpha=domain_alpha)
                loss = self.criterion(outputs, targets)

                # Domain adaptation loss
                if self.use_domain_adapt and 'domain_pred' in aux:
                    # Create domain labels (0 for source, 1 for target)
                    B = targets.size(0)
                    domain_labels = torch.zeros(B, dtype=torch.long, device=self.device)
                    domain_loss = F.cross_entropy(aux['domain_pred'], domain_labels)
                    loss = loss + self.domain_weight * domain_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * targets.size(0)

            if self.task == 'classification':
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(targets.cpu().numpy())

        self.scheduler.step(epoch)
        avg_loss = total_loss / len(loader.dataset)

        metrics = {'loss': avg_loss}
        if self.task == 'classification':
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        all_representations = []

        for modalities, targets in loader:
            modalities = [m.to(self.device, non_blocking=True) for m in modalities]
            targets = targets.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs, aux = self.model(modalities)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)

            if self.task == 'classification':
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(targets.cpu().numpy())
            else:
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

            if 'representation' in aux:
                all_representations.append(aux['representation'].cpu())

        avg_loss = total_loss / len(loader.dataset)
        metrics = {'loss': avg_loss}

        if self.task == 'classification':
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
            metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro')
            metrics['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted')
            metrics['precision'] = precision_score(all_labels, all_preds, average='macro')
            metrics['recall'] = recall_score(all_labels, all_preds, average='macro')
            metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
            metrics['preds'] = all_preds
            metrics['labels'] = all_labels
        else:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            metrics['mse'] = np.mean((all_preds - all_labels) ** 2)
            metrics['mae'] = np.mean(np.abs(all_preds - all_labels))
            metrics['rmse'] = np.sqrt(metrics['mse'])
            # MAPE (avoid division by zero)
            mask = np.abs(all_labels) > 1e-6
            if mask.sum() > 0:
                metrics['mape'] = np.mean(np.abs((all_preds[mask] - all_labels[mask]) / all_labels[mask])) * 100
            else:
                metrics['mape'] = 0.0
            metrics['preds'] = all_preds
            metrics['labels'] = all_labels

        if all_representations:
            metrics['representations'] = torch.cat(all_representations, dim=0).numpy()

        return metrics

    def train(self, train_loader, val_loader, max_epochs=50, patience=10,
              save_path=None, verbose=True):
        """Full training loop with early stopping."""
        es_mode = 'min' if self.task != 'classification' else 'max'
        es_metric = 'loss' if self.task != 'classification' else 'accuracy'
        early_stopping = EarlyStopping(patience=patience, mode=es_mode)

        best_val_metrics = None
        start_time = time.time()

        for epoch in range(max_epochs):
            train_metrics = self.train_epoch(train_loader, epoch, max_epochs)
            val_metrics = self.evaluate(val_loader)

            # Log
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            if self.task == 'classification':
                self.history['train_acc'].append(train_metrics.get('accuracy', 0))
                self.history['val_acc'].append(val_metrics.get('accuracy', 0))

            if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
                lr = self.optimizer.param_groups[0]['lr']
                if self.task == 'classification':
                    print(f"  Epoch {epoch+1:3d}/{max_epochs} | "
                          f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics.get('accuracy', 0):.4f} | "
                          f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics.get('accuracy', 0):.4f} | "
                          f"LR: {lr:.6f}")
                else:
                    print(f"  Epoch {epoch+1:3d}/{max_epochs} | "
                          f"Train Loss: {train_metrics['loss']:.4f} | "
                          f"Val MSE: {val_metrics.get('mse', 0):.4f} MAE: {val_metrics.get('mae', 0):.4f} | "
                          f"LR: {lr:.6f}")

            # Early stopping
            es_score = val_metrics[es_metric]
            early_stopping(es_score, self.model)

            if early_stopping.best_score == es_score:
                best_val_metrics = {k: v for k, v in val_metrics.items()
                                   if k not in ('preds', 'labels', 'representations', 'confusion_matrix')}

            if early_stopping.early_stop:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if early_stopping.best_model_state:
            self.model.load_state_dict(early_stopping.best_model_state)

        elapsed = time.time() - start_time

        if save_path:
            torch.save(self.model.state_dict(), save_path)

        return {
            'best_val_metrics': best_val_metrics,
            'training_time': elapsed,
            'epochs_trained': epoch + 1,
            'history': dict(self.history)
        }


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_single_experiment(model_name, model_kwargs, loaders, task, device,
                          max_epochs=50, patience=10, lr=1e-3, seed=42,
                          save_dir=None, verbose=True):
    """Run a single experiment with a given model and data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    from src.models import build_model
    model = build_model(model_name, **model_kwargs)
    n_params = count_parameters(model)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} | Params: {n_params:,} | Task: {task}")
        print(f"{'='*60}")

    trainer = Trainer(model, task=task, device=device, lr=lr)
    save_path = os.path.join(save_dir, f'{model_name}.pt') if save_dir else None

    train_result = trainer.train(
        loaders['train'], loaders['val'], max_epochs=max_epochs,
        patience=patience, save_path=save_path, verbose=verbose
    )

    # Test evaluation
    test_metrics = trainer.evaluate(loaders['test'])

    # Clean metrics for JSON serialization
    result = {
        'model_name': model_name,
        'n_params': n_params,
        'task': task,
        'training_time': train_result['training_time'],
        'epochs_trained': train_result['epochs_trained'],
    }

    if task == 'classification':
        result.update({
            'test_accuracy': test_metrics['accuracy'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_f1_weighted': test_metrics['f1_weighted'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_loss': test_metrics['loss'],
            'confusion_matrix': test_metrics.get('confusion_matrix', []),
        })
    else:
        result.update({
            'test_mse': test_metrics['mse'],
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_mape': test_metrics['mape'],
            'test_loss': test_metrics['loss'],
        })

    result['history'] = train_result['history']

    # Save representations for visualization
    if 'representations' in test_metrics:
        result['_representations'] = test_metrics['representations']
        result['_labels'] = test_metrics.get('labels', None)

    return result
