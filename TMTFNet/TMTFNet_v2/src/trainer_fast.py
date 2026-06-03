"""Fast trainer that consumes ``GPUResidentBundle`` instead of DataLoader.

Numerically equivalent to ``trainer.Trainer`` -- same AdamW/cosine schedule,
same EarlyStopping, same loss, same gradient clipping -- but it iterates over
GPU-resident tensors via ``GPUBatchIterator``. No worker processes, no
collate, no pinned-memory dance.

Use only for new experiments or future re-runs. The 5-seed rebuttal matrix
in ``results/rebuttal_*`` was produced with the slow ``trainer.Trainer`` path
and must continue to use that path for protocol consistency until a complete
fast-path rerun is prepared.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import amp

from .datasets_fast import GPUBatchIterator, GPUResidentBundle, GPUSplit
from .trainer import EarlyStopping, _build_scheduler


class FastTrainer:
    def __init__(
        self,
        model,
        task,
        device,
        lr,
        weight_decay=5e-4,
        label_smoothing=0.0,
        warmup_epochs=5,
        max_epochs=100,
        use_amp=True,
        batch_size=64,
    ):
        self.model = model.to(device)
        self.task = task
        self.device = device
        self.batch_size = int(batch_size)
        self.use_amp = use_amp and device.type == "cuda"
        self.history = defaultdict(list)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = _build_scheduler(self.optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs)
        self.scaler = amp.GradScaler("cuda", enabled=self.use_amp)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if task == "classification" else nn.MSELoss()

    def _train_epoch(self, train_split: GPUSplit, generator):
        self.model.train()
        total_loss = 0.0
        n_seen = 0
        for mods, tgt in GPUBatchIterator(train_split, self.batch_size, shuffle=True, generator=generator):
            self.optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=self.use_amp):
                outputs, _ = self.model(mods)
                loss = self.criterion(outputs, tgt)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            bs = tgt.size(0)
            total_loss += float(loss.detach()) * bs
            n_seen += bs
        return {"loss": total_loss / max(1, n_seen)}

    @torch.no_grad()
    def evaluate(self, split: GPUSplit):
        self.model.eval()
        total_loss = 0.0
        n_seen = 0
        all_preds, all_labels, all_outputs = [], [], []
        for mods, tgt in GPUBatchIterator(split, self.batch_size, shuffle=False):
            with amp.autocast("cuda", enabled=self.use_amp):
                outputs, _ = self.model(mods)
                loss = self.criterion(outputs, tgt)
            bs = tgt.size(0)
            total_loss += float(loss) * bs
            n_seen += bs
            if self.task == "classification":
                all_preds.append(outputs.argmax(dim=1).cpu().numpy())
                all_labels.append(tgt.cpu().numpy())
            else:
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(tgt.cpu().numpy())
        metrics = {"loss": total_loss / max(1, n_seen)}
        if self.task == "classification":
            preds = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            metrics.update(
                {
                    "accuracy": float(accuracy_score(labels, preds)),
                    "f1_macro": float(f1_score(labels, preds, average="macro")),
                    "f1_weighted": float(f1_score(labels, preds, average="weighted")),
                    "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
                    "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
                    "confusion_matrix": confusion_matrix(labels, preds).tolist(),
                }
            )
        else:
            outputs = np.concatenate(all_outputs, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            mse = float(np.mean((outputs - labels) ** 2))
            mae = float(np.mean(np.abs(outputs - labels)))
            metrics.update({"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))})
        return metrics

    def fit(self, bundle: GPUResidentBundle, max_epochs=100, patience=15, verbose=False, seed=0):
        gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        mode = "max" if self.task == "classification" else "min"
        monitor = "accuracy" if self.task == "classification" else "loss"
        stopper = EarlyStopping(patience=patience, mode=mode)
        start = time.time()
        epoch = 0
        for epoch in range(max_epochs):
            tm = self._train_epoch(bundle.train, gen)
            vm = self.evaluate(bundle.val)
            self.scheduler.step()
            self.history["train_loss"].append(tm["loss"])
            self.history["val_loss"].append(vm["loss"])
            if self.task == "classification":
                self.history["val_acc"].append(vm["accuracy"])
            else:
                self.history["val_mse"].append(vm["mse"])
            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print(f"  ep {epoch+1:3d} | train_loss={tm['loss']:.4f} | val={vm[monitor]:.4f}")
            stopper.step(vm[monitor], self.model)
            if stopper.should_stop:
                break
        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)
        return {"training_time": float(time.time() - start), "epochs_trained": epoch + 1, "history": dict(self.history)}


def train_and_evaluate_fast(
    model,
    bundle: GPUResidentBundle,
    task,
    device,
    lr,
    max_epochs,
    patience,
    weight_decay=5e-4,
    label_smoothing=0.0,
    warmup_epochs=5,
    batch_size=64,
    seed=0,
    verbose=False,
):
    trainer = FastTrainer(
        model=model,
        task=task,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    fit = trainer.fit(bundle, max_epochs=max_epochs, patience=patience, verbose=verbose, seed=seed)
    test_metrics = trainer.evaluate(bundle.test)
    out = {"training_time": fit["training_time"], "epochs_trained": fit["epochs_trained"], "history": fit["history"]}
    out.update(test_metrics)
    return out
