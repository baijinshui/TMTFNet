import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import amp


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.should_stop = False

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def _build_scheduler(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch_idx):
        epoch = epoch_idx + 1
        if epoch <= warmup_epochs:
            warmup_start = 0.1
            progress = epoch / max(1, warmup_epochs)
            return warmup_start + (1.0 - warmup_start) * progress
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class Trainer:
    def __init__(
        self,
        model,
        task,
        device,
        lr,
        weight_decay=5e-4,
        label_smoothing=0.0,
        grad_accum_steps=1,
        warmup_epochs=5,
        max_epochs=100,
        use_amp=True,
    ):
        self.model = model.to(device)
        self.task = task
        self.device = device
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = use_amp and device.type == "cuda"
        self.history = defaultdict(list)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = _build_scheduler(self.optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs)
        self.scaler = amp.GradScaler("cuda", enabled=self.use_amp)
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.MSELoss()

    def _move_batch(self, batch):
        modalities, targets = batch
        modalities = [tensor.to(self.device, non_blocking=True) for tensor in modalities]
        targets = targets.to(self.device, non_blocking=True)
        return modalities, targets

    def train_epoch(self, loader):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_examples = 0
        all_preds = []
        all_labels = []

        for step, batch in enumerate(loader, start=1):
            modalities, targets = self._move_batch(batch)
            with amp.autocast("cuda", enabled=self.use_amp):
                outputs, _ = self.model(modalities)
                loss = self.criterion(outputs, targets) / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if step % self.grad_accum_steps == 0 or step == len(loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size * self.grad_accum_steps
            total_examples += batch_size

            if self.task == "classification":
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                labels = targets.detach().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels)

        metrics = {"loss": total_loss / max(1, total_examples)}
        if self.task == "classification":
            preds = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            metrics["accuracy"] = accuracy_score(labels, preds)
        return metrics

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
                outputs, aux = self.model(modalities)
                loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

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
            mse = np.mean((outputs - labels) ** 2)
            mae = np.mean(np.abs(outputs - labels))
            metrics.update(
                {
                    "mse": float(mse),
                    "mae": float(mae),
                    "rmse": float(np.sqrt(mse)),
                }
            )
        return metrics

    def fit(self, train_loader, val_loader, max_epochs=100, patience=15, verbose=False):
        mode = "max" if self.task == "classification" else "min"
        monitor = "accuracy" if self.task == "classification" else "loss"
        stopper = EarlyStopping(patience=patience, mode=mode)
        start_time = time.time()

        for epoch in range(max_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            if self.task == "classification":
                self.history["train_acc"].append(train_metrics["accuracy"])
                self.history["val_acc"].append(val_metrics["accuracy"])
            else:
                self.history["val_mse"].append(val_metrics["mse"])
                self.history["val_mae"].append(val_metrics["mae"])

            if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == max_epochs):
                lr = self.optimizer.param_groups[0]["lr"]
                if self.task == "classification":
                    print(
                        f"  Epoch {epoch + 1:3d}/{max_epochs} | "
                        f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                        f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
                        f"lr={lr:.6f}"
                    )
                else:
                    print(
                        f"  Epoch {epoch + 1:3d}/{max_epochs} | "
                        f"train_loss={train_metrics['loss']:.4f} | "
                        f"val_mse={val_metrics['mse']:.4f} val_mae={val_metrics['mae']:.4f} | "
                        f"lr={lr:.6f}"
                    )

            stopper.step(val_metrics[monitor], self.model)
            if stopper.should_stop:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)

        elapsed = time.time() - start_time
        return {
            "training_time": elapsed,
            "epochs_trained": epoch + 1,
            "history": dict(self.history),
        }


def train_and_evaluate_swa(
    model,
    loaders,
    task,
    device,
    lr,
    max_epochs,
    patience,
    weight_decay=5e-4,
    label_smoothing=0.0,
    grad_accum_steps=1,
    warmup_epochs=5,
    swa_start_frac=0.7,
    mixup_alpha=0.0,
    verbose=False,
):
    """Like train_and_evaluate but with Stochastic Weight Averaging (SWA) and
    optional input-level mixup. SWA averages model weights from
    ``swa_start_frac * max_epochs`` onward and replaces the final model with
    that average at evaluation time, which typically yields +0.5--1% accuracy
    on classification tasks. mixup_alpha=0.0 disables mixup."""
    import copy
    from collections import defaultdict as _dd

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _build_scheduler(optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs)
    use_amp = device.type == "cuda"
    scaler = amp.GradScaler("cuda", enabled=use_amp)
    if task == "classification":
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = nn.MSELoss()

    history = _dd(list)
    swa_state = None
    swa_count = 0
    swa_start_epoch = max(1, int(swa_start_frac * max_epochs))
    rng = np.random.default_rng(0)

    def _mix_batch(modalities, targets):
        if mixup_alpha <= 0 or task != "classification":
            return modalities, targets, None, 1.0
        lam = float(rng.beta(mixup_alpha, mixup_alpha))
        idx = torch.randperm(targets.size(0), device=device)
        mixed_modalities = [lam * m + (1 - lam) * m[idx] for m in modalities]
        return mixed_modalities, targets, idx, lam

    stopper = EarlyStopping(patience=patience, mode="max" if task == "classification" else "min")
    monitor = "accuracy" if task == "classification" else "loss"
    start = time.time()
    epoch = 0
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in loaders["train"]:
            mods, tgt = batch
            mods = [m.to(device, non_blocking=True) for m in mods]
            tgt = tgt.to(device, non_blocking=True)
            mods, tgt, idx, lam = _mix_batch(mods, tgt)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=use_amp):
                outputs, _ = model(mods)
                if idx is not None:
                    loss = lam * criterion(outputs, tgt) + (1 - lam) * criterion(outputs, tgt[idx])
                else:
                    loss = criterion(outputs, tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss) * tgt.size(0)
            train_n += tgt.size(0)
        scheduler.step()
        history["train_loss"].append(train_loss_sum / max(1, train_n))

        # eval on val
        val_metrics = _evaluate_simple(model, loaders["val"], task, device, criterion, use_amp)
        history["val_loss"].append(val_metrics["loss"])
        if task == "classification":
            history["val_acc"].append(val_metrics["accuracy"])
        else:
            history["val_mse"].append(val_metrics["mse"])

        # SWA accumulation
        if epoch + 1 >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                swa_count += 1
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        swa_state[k].mul_((swa_count - 1) / swa_count).add_(v.detach(), alpha=1 / swa_count)
                    else:
                        swa_state[k] = v.detach().clone()

        stopper.step(val_metrics[monitor], model)
        if stopper.should_stop:
            break

    if swa_state is not None:
        model.load_state_dict(swa_state)
    elif stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    test_metrics = _evaluate_simple(model, loaders["test"], task, device, criterion, use_amp,
                                     full_metrics=True)
    elapsed = time.time() - start
    out = {
        "training_time": float(elapsed),
        "epochs_trained": epoch + 1,
        "history": dict(history),
        "swa_count": int(swa_count),
        "mixup_alpha": float(mixup_alpha),
    }
    out.update(test_metrics)
    return out


def _evaluate_simple(model, loader, task, device, criterion, use_amp, full_metrics=False):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_labels, all_outputs = [], [], []
    with torch.no_grad():
        for batch in loader:
            mods, tgt = batch
            mods = [m.to(device, non_blocking=True) for m in mods]
            tgt = tgt.to(device, non_blocking=True)
            with amp.autocast("cuda", enabled=use_amp):
                outputs, _ = model(mods)
                loss = criterion(outputs, tgt)
            total_loss += float(loss) * tgt.size(0)
            n += tgt.size(0)
            if task == "classification":
                all_preds.append(outputs.argmax(dim=1).cpu().numpy())
                all_labels.append(tgt.cpu().numpy())
            else:
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(tgt.cpu().numpy())
    metrics = {"loss": total_loss / max(1, n)}
    if task == "classification":
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics["accuracy"] = float(accuracy_score(labels, preds))
        if full_metrics:
            metrics.update({
                "f1_macro": float(f1_score(labels, preds, average="macro")),
                "f1_weighted": float(f1_score(labels, preds, average="weighted")),
                "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
                "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            })
    else:
        outputs = np.concatenate(all_outputs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        mse = float(np.mean((outputs - labels) ** 2))
        mae = float(np.mean(np.abs(outputs - labels)))
        metrics.update({"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))})
    return metrics


def train_and_evaluate(
    model,
    loaders,
    task,
    device,
    lr,
    max_epochs,
    patience,
    weight_decay=5e-4,
    label_smoothing=0.0,
    grad_accum_steps=1,
    warmup_epochs=5,
    verbose=False,
):
    trainer = Trainer(
        model=model,
        task=task,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        grad_accum_steps=grad_accum_steps,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
    )
    fit_result = trainer.fit(
        loaders["train"],
        loaders["val"],
        max_epochs=max_epochs,
        patience=patience,
        verbose=verbose,
    )
    test_metrics = trainer.evaluate(loaders["test"])
    result = {
        "training_time": float(fit_result["training_time"]),
        "epochs_trained": int(fit_result["epochs_trained"]),
        "history": fit_result["history"],
    }
    result.update(test_metrics)
    return result


@torch.no_grad()
def measure_inference_time(model, loader, device, num_batches=20):
    model = model.to(device)
    model.eval()
    durations = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        modalities, _ = batch
        modalities = [tensor.to(device, non_blocking=True) for tensor in modalities]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(modalities)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(durations)) if durations else 0.0
