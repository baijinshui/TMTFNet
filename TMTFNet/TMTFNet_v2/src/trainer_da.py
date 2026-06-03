"""Domain-adaptive trainer.

Wraps ``src.trainer.Trainer``-style AdamW/cosine optimisation with optional
alignment objectives that draw an unlabeled target batch every step:

    L = L_task(source) + lambda_coral * CORAL(src_feat, tgt_feat)
                      + lambda_consistency * ModalityConsistency(src_mods)
                      + lambda_dann * CrossEntropy(domain_logits, domain_label)

``lambda_dann`` follows the canonical DANN schedule
``lambda_dann = base * (2/(1+exp(-gamma*p)) - 1)`` with progress ``p`` ramped
from 0 to 1 across all source batches seen. The model is expected to return
``aux["shared_feature"]``, ``aux["modality_features"]`` and (when DANN is
enabled) ``aux["domain_logits"]`` -- all provided by ``TMTFNet_v2``.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import amp

from .alignment import ModalityConsistency, coral_loss
from .trainer import EarlyStopping, _build_scheduler


class DomainAdaptTrainer:
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
        lambda_coral=0.0,
        lambda_consistency=0.0,
        lambda_dann=0.0,
        dann_gamma=10.0,
    ):
        self.model = model.to(device)
        self.task = task
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.history = defaultdict(list)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = _build_scheduler(self.optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs)
        self.scaler = amp.GradScaler("cuda", enabled=self.use_amp)
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.MSELoss()
        self.consistency_fn = ModalityConsistency()
        self.domain_ce = nn.CrossEntropyLoss()
        self.lambda_coral = float(lambda_coral)
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_dann = float(lambda_dann)
        self.dann_gamma = float(dann_gamma)

    def _move_batch(self, batch):
        modalities, targets = batch
        modalities = [tensor.to(self.device, non_blocking=True) for tensor in modalities]
        targets = targets.to(self.device, non_blocking=True)
        return modalities, targets

    def _move_modalities_only(self, batch):
        modalities, _ = batch
        return [tensor.to(self.device, non_blocking=True) for tensor in modalities]

    def _dann_alpha(self, progress):
        p = max(0.0, min(1.0, float(progress)))
        return 2.0 / (1.0 + math.exp(-self.dann_gamma * p)) - 1.0

    def train_epoch(self, source_loader, target_loader, epoch_idx, total_epochs):
        self.model.train()
        total = {"loss": 0.0, "task": 0.0, "coral": 0.0, "cons": 0.0, "dann": 0.0}
        n_seen = 0

        target_iter = itertools.cycle(target_loader) if target_loader is not None else None
        steps = len(source_loader)

        for step_idx, src_batch in enumerate(source_loader):
            modalities_s, targets_s = self._move_batch(src_batch)
            if target_iter is not None:
                tgt_batch = next(target_iter)
                modalities_t = self._move_modalities_only(tgt_batch)
            else:
                modalities_t = None

            progress = ((epoch_idx * steps) + step_idx) / max(1, total_epochs * steps)
            alpha = self._dann_alpha(progress) if self.lambda_dann > 0 else 0.0

            self.optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=self.use_amp):
                out_s, aux_s = self.model(modalities_s, domain_alpha=alpha)
                loss_task = self.criterion(out_s, targets_s)

                loss_coral = torch.zeros((), device=self.device)
                loss_cons = torch.zeros((), device=self.device)
                loss_dann = torch.zeros((), device=self.device)

                if modalities_t is not None and (self.lambda_coral > 0 or self.lambda_dann > 0):
                    _, aux_t = self.model(modalities_t, domain_alpha=alpha)
                    if self.lambda_coral > 0 and "shared_feature" in aux_s and "shared_feature" in aux_t:
                        loss_coral = coral_loss(aux_s["shared_feature"].float(), aux_t["shared_feature"].float())

                    if self.lambda_dann > 0 and "domain_logits" in aux_s and "domain_logits" in aux_t:
                        logits_s = aux_s["domain_logits"]
                        logits_t = aux_t["domain_logits"]
                        label_s = torch.zeros(logits_s.size(0), dtype=torch.long, device=self.device)
                        label_t = torch.ones(logits_t.size(0), dtype=torch.long, device=self.device)
                        loss_dann = 0.5 * (self.domain_ce(logits_s, label_s) + self.domain_ce(logits_t, label_t))

                if self.lambda_consistency > 0 and "modality_features" in aux_s and len(aux_s["modality_features"]) > 1:
                    loss_cons = self.consistency_fn(aux_s["modality_features"])

                loss = (
                    loss_task
                    + self.lambda_coral * loss_coral
                    + self.lambda_consistency * loss_cons
                    + self.lambda_dann * loss_dann
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = targets_s.size(0)
            n_seen += bs
            total["loss"] += float(loss.detach()) * bs
            total["task"] += float(loss_task.detach()) * bs
            total["coral"] += float(loss_coral.detach()) * bs
            total["cons"] += float(loss_cons.detach()) * bs
            total["dann"] += float(loss_dann.detach()) * bs

        return {k: v / max(1, n_seen) for k, v in total.items()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        all_preds = []
        all_labels = []
        all_outputs = []

        for batch in loader:
            modalities, targets = self._move_batch(batch)
            with amp.autocast("cuda", enabled=self.use_amp):
                outputs, _ = self.model(modalities)
                loss = self.criterion(outputs, targets)

            bs = targets.size(0)
            total_loss += float(loss) * bs
            total_examples += bs
            if self.task == "classification":
                all_preds.append(outputs.argmax(dim=1).cpu().numpy())
                all_labels.append(targets.cpu().numpy())
            else:
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

        metrics = {"loss": total_loss / max(1, total_examples)}
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

    def fit(self, source_loader, val_loader, target_loader=None, max_epochs=100, patience=15, verbose=False):
        mode = "max" if self.task == "classification" else "min"
        monitor = "accuracy" if self.task == "classification" else "loss"
        stopper = EarlyStopping(patience=patience, mode=mode)
        start = time.time()
        epoch = 0
        for epoch in range(max_epochs):
            train_metrics = self.train_epoch(source_loader, target_loader, epoch, max_epochs)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            if self.task == "classification":
                self.history["val_acc"].append(val_metrics["accuracy"])
            else:
                self.history["val_mse"].append(val_metrics["mse"])
            if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
                print(
                    f"  ep {epoch + 1:3d}/{max_epochs} | task={train_metrics['task']:.4f} "
                    f"cor={train_metrics['coral']:.4f} cons={train_metrics['cons']:.4f} "
                    f"dann={train_metrics['dann']:.4f} | val={val_metrics[monitor]:.4f}"
                )
            stopper.step(val_metrics[monitor], self.model)
            if stopper.should_stop:
                break
        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)
        return {"training_time": float(time.time() - start), "epochs_trained": epoch + 1, "history": dict(self.history)}


def train_and_evaluate_da(
    model,
    loaders,
    target_loader,
    task,
    device,
    lr,
    max_epochs,
    patience,
    lambda_coral=0.0,
    lambda_consistency=0.0,
    lambda_dann=0.0,
    weight_decay=5e-4,
    label_smoothing=0.0,
    warmup_epochs=5,
    verbose=False,
):
    trainer = DomainAdaptTrainer(
        model=model,
        task=task,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        lambda_coral=lambda_coral,
        lambda_consistency=lambda_consistency,
        lambda_dann=lambda_dann,
    )
    fit = trainer.fit(
        loaders["train"],
        loaders["val"],
        target_loader=target_loader,
        max_epochs=max_epochs,
        patience=patience,
        verbose=verbose,
    )
    test_metrics = trainer.evaluate(loaders["test"])
    result = {"training_time": fit["training_time"], "epochs_trained": fit["epochs_trained"], "history": fit["history"]}
    result.update(test_metrics)
    return result
