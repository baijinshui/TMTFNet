"""TMTFNet-Distill: train TMTFNet to mimic PatchTST predictions on PAMAP2.

Knowledge distillation from PatchTST teacher to TMTFNet student. The
teacher is loaded from the rebuttal_exp1c_har_pamap2 checkpoint (or
re-trained per seed if missing). Student loss is
``CE(student, true_label) + lambda * KL(student || softened_teacher)``
with temperature T=4 and lambda=0.5. Combined with SWA over the last 30%
of epochs and mild mixup for regularisation.

Goal: lift TMTFNet's PAMAP2 cross-subject accuracy from ~78% toward
PatchTST's 87.5% by transferring PatchTST's learned class boundaries
into TMTFNet's representation space, while keeping TMTFNet's multi-modal
fusion advantages.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.datasets import get_pamap2_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import (EarlyStopping, _build_scheduler,  # noqa: E402
                         _evaluate_simple, count_parameters, set_global_seed)

DATA_DIR = str(REPO / "data")
SEEDS = (42, 123, 456, 789, 2024)
OUT = REPO / "results" / "rebuttal_exp1c_har_pamap2_distill"
OUT.mkdir(parents=True, exist_ok=True)


def train_teacher_or_load(seed, bundle, max_epochs=60):
    """Train a PatchTST teacher (use cached if available)."""
    teacher_dir = REPO / "results" / "tsne_checkpoints"  # reuse this directory
    teacher_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = teacher_dir / f"PatchTST_pamap2_seed{seed}.pt"
    if teacher_path.exists():
        teacher = build_model("PatchTST", modality_dims=bundle.modality_dims,
                              d_model=128, n_heads=8, n_enc_layers=3, dropout=0.1,
                              n_classes=bundle.n_classes, seq_len=171,
                              patch_len=8, stride=4)
        state = torch.load(teacher_path, map_location="cuda", weights_only=False)
        teacher.load_state_dict(state["state_dict"], strict=False)
        teacher = teacher.to("cuda").eval()
        return teacher

    print(f"  training PatchTST teacher for seed={seed}...")
    set_global_seed(int(seed))
    teacher = build_model("PatchTST", modality_dims=bundle.modality_dims,
                           d_model=128, n_heads=8, n_enc_layers=3, dropout=0.1,
                           n_classes=bundle.n_classes, seq_len=171,
                           patch_len=8, stride=4).to("cuda")
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = _build_scheduler(optimizer, warmup_epochs=5, max_epochs=max_epochs)
    scaler = amp.GradScaler("cuda", enabled=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    stopper = EarlyStopping(patience=15, mode="max")
    for epoch in range(max_epochs):
        teacher.train()
        for mods, tgt in bundle.loaders["train"]:
            mods = [m.to("cuda", non_blocking=True) for m in mods]
            tgt = tgt.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=True):
                out, _ = teacher(mods)
                loss = criterion(out, tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        scheduler.step()
        val = _evaluate_simple(teacher, bundle.loaders["val"], "classification", torch.device("cuda"), criterion, True)
        stopper.step(val["accuracy"], teacher)
        if stopper.should_stop:
            break
    if stopper.best_state is not None:
        teacher.load_state_dict(stopper.best_state)
    torch.save({"state_dict": teacher.state_dict()}, teacher_path)
    teacher.eval()
    return teacher


def distill_one_seed(seed, max_epochs=120, T=4.0, lam_kd=0.5, mixup_alpha=0.2,
                     swa_start_frac=0.7):
    out_path = OUT / f"TMTFNet_distill_seed{seed}.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open() as fh:
            return json.load(fh).get("accuracy")
    set_global_seed(int(seed))
    bundle = get_pamap2_loaders(os.path.join(DATA_DIR, "PAMAP2_Dataset"),
                                 batch_size=64, num_workers=2, split_seed=int(seed))
    teacher = train_teacher_or_load(seed, bundle)

    set_global_seed(int(seed))
    student = build_model("TMTFNet_v2", modality_dims=bundle.modality_dims,
                           d_model=64, n_heads=8, n_enc_layers=2,
                           modality_dropout=0.1, n_classes=bundle.n_classes,
                           dropout=0.1, seq_len=171).to("cuda")
    n_params = count_parameters(student)

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = _build_scheduler(optimizer, warmup_epochs=5, max_epochs=max_epochs)
    scaler = amp.GradScaler("cuda", enabled=True)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    kld = nn.KLDivLoss(reduction="batchmean")
    rng = np.random.default_rng(int(seed))
    history = defaultdict(list)
    swa_state = None
    swa_count = 0
    swa_start_epoch = max(1, int(swa_start_frac * max_epochs))
    stopper = EarlyStopping(patience=20, mode="max")
    t0 = time.time()
    epoch = 0
    for epoch in range(max_epochs):
        student.train()
        train_loss_sum = 0.0
        train_n = 0
        for mods, tgt in bundle.loaders["train"]:
            mods = [m.to("cuda", non_blocking=True) for m in mods]
            tgt = tgt.to("cuda", non_blocking=True)
            # mixup
            if mixup_alpha > 0:
                lam = float(rng.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(tgt.size(0), device="cuda")
                mods = [lam * m + (1 - lam) * m[idx] for m in mods]
            else:
                lam = 1.0; idx = None

            with torch.no_grad():
                t_logits, _ = teacher(mods)
                t_soft = F.softmax(t_logits / T, dim=-1)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=True):
                s_logits, _ = student(mods)
                s_log_soft = F.log_softmax(s_logits / T, dim=-1)
                if idx is None:
                    loss_ce = ce(s_logits, tgt)
                else:
                    loss_ce = lam * ce(s_logits, tgt) + (1 - lam) * ce(s_logits, tgt[idx])
                loss_kd = kld(s_log_soft, t_soft) * (T * T)
                loss = loss_ce + lam_kd * loss_kd

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            train_loss_sum += float(loss.detach()) * tgt.size(0)
            train_n += tgt.size(0)
        scheduler.step()
        history["train_loss"].append(train_loss_sum / max(1, train_n))

        val = _evaluate_simple(student, bundle.loaders["val"], "classification",
                                torch.device("cuda"), ce, True)
        history["val_acc"].append(val["accuracy"])

        if epoch + 1 >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
                swa_count = 1
            else:
                swa_count += 1
                for k, v in student.state_dict().items():
                    if v.dtype.is_floating_point:
                        swa_state[k].mul_((swa_count - 1) / swa_count).add_(v.detach(), alpha=1 / swa_count)
                    else:
                        swa_state[k] = v.detach().clone()

        stopper.step(val["accuracy"], student)
        if stopper.should_stop:
            break

    if swa_state is not None:
        student.load_state_dict(swa_state)
    elif stopper.best_state is not None:
        student.load_state_dict(stopper.best_state)

    test = _evaluate_simple(student, bundle.loaders["test"], "classification",
                             torch.device("cuda"), ce, True, full_metrics=True)
    elapsed = time.time() - t0
    result = {
        **test,
        "training_time": float(elapsed),
        "epochs_trained": epoch + 1,
        "history": dict(history),
        "swa_count": int(swa_count),
        "lam_kd": float(lam_kd),
        "T": float(T),
        "mixup_alpha": float(mixup_alpha),
        "seed": int(seed),
        "model_name": "TMTFNet_distill",
        "n_params": int(n_params),
    }
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    return result["accuracy"]


def main():
    print(f"started: {time.strftime('%F %T')}")
    accs = []
    for seed in SEEDS:
        acc = distill_one_seed(seed)
        accs.append(acc)
        print(f"  PAMAP2 distill seed={seed} acc={acc:.4f}")
    arr = np.asarray(accs, dtype=np.float64)
    print(f"\n=== PAMAP2 TMTFNet-Distill 5-seed ===")
    print(f"mean={arr.mean()*100:.2f}, std={arr.std(ddof=1)*100:.2f}")
    print(f"vs PatchTST baseline 87.53 ± 1.44")
    if arr.mean() > 0.8753:
        print(">>> WIN <<<")
    print(f"finished: {time.strftime('%F %T')}")


if __name__ == "__main__":
    main()
