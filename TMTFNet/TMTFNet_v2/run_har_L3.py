"""Run TMTFNet on UCI HAR with L=3 (instead of default L=2).

The Sec.IV-I sensitivity sweep at single seed=42 reported 94.74% accuracy at
L=3, which would beat the strongest deep baseline PatchTST (94.18%) under
the same protocol. We re-run all 5 seeds with L=3 to see whether the
single-seed advantage holds under the full statistical protocol. Output
files are written under results/rebuttal_exp1a_har_uci_L3/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.datasets import get_har_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import (count_parameters, set_global_seed,  # noqa: E402
                         train_and_evaluate)

OUT = REPO / "results" / "rebuttal_exp1a_har_uci_L3"
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = str(REPO / "data")
SEEDS = (42, 123, 456, 789, 2024)


def main():
    print(f"started: {time.strftime('%F %T')}")
    accs = []
    for seed in SEEDS:
        path = OUT / f"TMTFNet_v2_L3_seed{seed}.json"
        if path.exists() and path.stat().st_size > 0:
            with path.open() as fh:
                accs.append(json.load(fh).get("accuracy"))
            print(f"  seed={seed} cached acc={accs[-1]:.4f}")
            continue
        set_global_seed(int(seed))
        bundle = get_har_loaders(DATA_DIR, batch_size=64, num_workers=2, seq_len=128, split_seed=int(seed))
        set_global_seed(int(seed))
        model = build_model(
            "TMTFNet_v2",
            modality_dims=bundle.modality_dims,
            d_model=64, n_heads=8, n_enc_layers=3,    # <-- L=3 (was 2)
            modality_dropout=0.1, seq_len=128,
            n_classes=bundle.n_classes, dropout=0.1,
        )
        n_params = count_parameters(model)
        t0 = time.time()
        result = train_and_evaluate(
            model=model, loaders=bundle.loaders, task="classification",
            device=torch.device("cuda"), lr=5e-4,
            max_epochs=100, patience=15, label_smoothing=0.1,
        )
        result.update({"seed": int(seed), "model_name": "TMTFNet_v2_L3", "n_params": int(n_params),
                       "wall_time": time.time() - t0})
        with path.open("w") as fh:
            json.dump(result, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        accs.append(result["accuracy"])
        print(f"  seed={seed} acc={result['accuracy']:.4f} (n_params={n_params}, t={result['wall_time']:.1f}s)")

    import numpy as np
    arr = np.asarray(accs, dtype=np.float64)
    print(f"\n=== L=3 5-seed result ===")
    print(f"mean={arr.mean()*100:.2f}, std={arr.std(ddof=1)*100:.2f}")
    print(f"vs PatchTST baseline 94.18 ± 0.47")
    if arr.mean() > 0.9418:
        print(">>> WIN: TMTFNet (L=3) > PatchTST <<<")
    else:
        print(">>> still trails PatchTST <<<")
    print(f"finished: {time.strftime('%F %T')}")


if __name__ == "__main__":
    main()
