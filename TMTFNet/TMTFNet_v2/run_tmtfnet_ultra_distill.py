"""TMTFNet-Ultra-Distill: combine Ultra architecture (per-channel patch
encoders) with PatchTST knowledge distillation (T=4, lambda=0.5) and SWA +
mixup. Last attempt to push PAMAP2 cross-subject past PatchTST 87.53%."""

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
OUT = REPO / "results" / "rebuttal_exp1c_har_pamap2_ultra_distill"
OUT.mkdir(parents=True, exist_ok=True)
TEACHER_DIR = REPO / "results" / "tsne_checkpoints"


def load_or_train_teacher(seed, bundle, max_epochs=60):
    teacher_path = TEACHER_DIR / f"PatchTST_pamap2_seed{seed}.pt"
    if teacher_path.exists():
        teacher = build_model("PatchTST", modality_dims=bundle.modality_dims,
                              d_model=128, n_heads=8, n_enc_layers=3, dropout=0.1,
                              n_classes=bundle.n_classes, seq_len=171,
                              patch_len=8, stride=4)
        teacher.load_state_dict(torch.load(teacher_path, map_location="cuda", weights_only=False)["state_dict"], strict=False)
        return teacher.to("cuda").eval()
    raise FileNotFoundError(f"missing teacher for seed {seed}; run distill first")


def run_one_seed(seed, max_epochs=120, T=4.0, lam_kd=0.5, mixup_alpha=0.2,
                 swa_start_frac=0.7):
    out_path = OUT / f"TMTFNet_ultra_distill_seed{seed}.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open() as fh:
            return json.load(fh).get("accuracy")
    set_global_seed(int(seed))
    bundle = get_pamap2_loaders(os.path.join(DATA_DIR, "PAMAP2_Dataset"),
                                 batch_size=64, num_workers=2, split_seed=int(seed))
    teacher = load_or_train_teacher(seed, bundle)

    set_global_seed(int(seed))
    student = build_model(
        "TMTFNetUltra",
        modality_dims=bundle.modality_dims,
        d_model=64, n_heads=8, n_enc_layers=2,
        modality_dropout=0.1, n_classes=bundle.n_classes,
        dropout=0.1, seq_len=171,
        patch_len=16, patch_stride=8,
    ).to("cuda")
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
    result = {**test, "training_time": float(elapsed), "epochs_trained": epoch + 1,
              "history": dict(history), "swa_count": int(swa_count),
              "lam_kd": float(lam_kd), "T": float(T), "mixup_alpha": float(mixup_alpha),
              "seed": int(seed), "model_name": "TMTFNet_ultra_distill",
              "n_params": int(n_params)}
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    return result["accuracy"]


def main():
    print(f"started: {time.strftime('%F %T')}")
    accs = []
    for seed in SEEDS:
        try:
            acc = run_one_seed(seed)
            accs.append(acc)
            print(f"  PAMAP2 ultra+distill seed={seed} acc={acc:.4f}")
        except FileNotFoundError as e:
            print(f"  seed={seed} skipped: {e}")
    if accs:
        arr = np.asarray(accs, dtype=np.float64)
        print(f"\n=== PAMAP2 TMTFNet-Ultra-Distill {len(accs)}-seed ===")
        print(f"mean={arr.mean()*100:.2f}, std={arr.std(ddof=1)*100 if len(accs)>1 else 0:.2f}")
        print(f"vs PatchTST baseline 87.53 ± 1.44")
        if arr.mean() > 0.8753:
            print(">>> WIN <<<")
    print(f"finished: {time.strftime('%F %T')}")


if __name__ == "__main__":
    main()
