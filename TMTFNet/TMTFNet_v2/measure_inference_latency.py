"""Inference latency benchmark for the HAR deep-model lineup.

For each model used in Table 1 / Table 8, build it with the rebuttal HAR
config, push a representative test batch through it N=200 times after
50 warmup iterations, and record the per-sample latency. Saves the
numbers to results/inference_latency.json so the figure / table renderer
can pick them up.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.datasets import get_har_loaders
from src.models import build_model

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data"
RESULTS = REPO / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "inference_latency.json"

HAR_DEEP_MODELS = [
    "DLinear", "TimeMixer", "CNN1D", "DeepConvLSTM", "TCN",
    "iTransformer", "Transformer", "GRU", "TMTFNet_v2", "PatchTST",
    "LSTM", "Crossformer",
]


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_inputs(bundle, device, batch_size):
    test_loader = bundle.loaders["test"]
    modality_inputs, _ = next(iter(test_loader))
    modality_inputs = [t.to(device) for t in modality_inputs]
    n = modality_inputs[0].shape[0]
    if n != batch_size:
        if n >= batch_size:
            modality_inputs = [t[:batch_size] for t in modality_inputs]
        else:
            reps = (batch_size + n - 1) // n
            modality_inputs = [t.repeat(reps, *([1] * (t.dim() - 1)))[:batch_size]
                               for t in modality_inputs]
    return modality_inputs


def _time_model(model, inputs, device, n_warmup, n_iters):
    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_iters


def bench_model(model_name, bundle, device, batch_size=64,
                n_warmup=50, n_iters=200,
                n_compile_warmup=100, n_compile_iters=200):
    cfg = dict(
        d_model=64 if model_name == "TMTFNet_v2" else 128,
        n_heads=8,
        n_enc_layers=2 if model_name == "TMTFNet_v2" else 3,
        modality_dropout=0.1,
        seq_len=128,
        n_classes=bundle.n_classes,
        dropout=0.1,
    )
    torch.manual_seed(42)
    model = build_model(model_name, modality_dims=bundle.modality_dims, **cfg).to(device)
    model.eval()
    inputs = _make_inputs(bundle, device, batch_size)

    eager_s = _time_model(model, inputs, device, n_warmup, n_iters)

    compiled_s = None
    compiled_err = None
    try:
        compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
        compiled_s = _time_model(compiled, inputs, device, n_compile_warmup, n_compile_iters)
    except Exception as e:  # noqa: BLE001
        compiled_err = repr(e)

    def _to_us(s):
        return None if s is None else s * 1e6 / batch_size

    return {
        "model": model_name,
        "n_params": sum(p.numel() for p in model.parameters()),
        "batch_size": batch_size,
        "eager_per_batch_ms": eager_s * 1000.0,
        "eager_per_sample_us": _to_us(eager_s),
        "compiled_per_batch_ms": None if compiled_s is None else compiled_s * 1000.0,
        "compiled_per_sample_us": _to_us(compiled_s),
        "compiled_err": compiled_err,
    }


def main():
    device = _device()
    print(f"device: {device}")
    bundle = get_har_loaders(DATA_DIR, batch_size=64, num_workers=0, seq_len=128)
    results = []
    for name in HAR_DEEP_MODELS:
        try:
            r = bench_model(name, bundle, device)
            results.append(r)
            comp = r["compiled_per_sample_us"]
            comp_str = f"{comp:7.2f}" if comp is not None else "  FAIL  "
            speedup = (r["eager_per_sample_us"] / comp) if comp else None
            sp_str = f"{speedup:4.2f}x" if speedup else "   -  "
            print(f"  {name:14s}  eager {r['eager_per_sample_us']:7.2f} us  "
                  f"compiled {comp_str} us  speedup {sp_str}  "
                  f"params {r['n_params']/1000:.1f}K")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:14s}  FAILED: {e}")
            results.append({"model": name, "error": str(e)})

    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
