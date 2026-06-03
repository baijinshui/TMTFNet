"""Collect every rebuttal summary.json into one flat JSON that the manuscript
revisions can consult. Writes:

* ``results/latex_numbers.json`` -- one model-centric dict per experiment,
  with ``mean``/``std``/``n_seeds`` for each metric. The manuscript rewrite
  looks up numbers here instead of pulling from individual files.
* ``results/latex_numbers.md`` -- human-readable Markdown mirror, easier to
  eyeball while drafting tables.

Run this after any partial set of experiments finish; it tolerates missing
summaries and just emits what is available.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "latex_numbers.json"
OUT_MD = RESULTS / "latex_numbers.md"


def _load(path):
    try:
        with path.open() as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _normalise_summary(summary):
    out = {}
    for model, stats in summary.items():
        if model.startswith("_") or not isinstance(stats, dict):
            continue
        entry = {"n_seeds": stats.get("n_seeds"), "n_params": stats.get("n_params")}
        for metric in ("accuracy", "f1_macro", "f1_weighted", "precision", "recall", "mse", "mae", "training_time"):
            val = stats.get(metric)
            if isinstance(val, dict):
                entry[metric] = {"mean": val.get("mean"), "std": val.get("std")}
        out[model] = entry
    return out


def _collect_single_summary(exp):
    path = RESULTS / f"rebuttal_{exp}" / "summary.json"
    summary = _load(path)
    if summary is None:
        return None
    return _normalise_summary(summary)


def _collect_split_summaries(exp):
    out = {}
    for path in sorted((RESULTS).glob(f"rebuttal_{exp}__*/summary.json")):
        label = path.parent.name.replace(f"rebuttal_{exp}__", "")
        summary = _load(path)
        if summary:
            out[label] = _normalise_summary(summary)
    return out or None


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = {}
    data["exp1a_har_uci"] = _collect_single_summary("exp1a_har_uci")
    data["exp1b_har_classical"] = _load(RESULTS / "rebuttal_exp1b_har_classical" / "all_seeds.json")
    data["exp1c_har_pamap2"] = _collect_single_summary("exp1c_har_pamap2")
    data["exp2_forecast"] = {
        ds: {h: _collect_single_summary(f"exp2_forecast_{ds.lower()}_pred{h}") for h in (24, 48, 96)}
        for ds in ("ETTh1", "ETTh2", "ETTm1")
    }
    data["exp3_cross_forecast"] = _collect_split_summaries("exp3_cross_forecast_pred24")
    data["exp3_da_forecast"] = _collect_split_summaries("exp3_cross_da_pred24")
    data["exp4_ablation"] = _collect_split_summaries("exp4_ablation")
    data["exp5_sensitivity"] = _load(RESULTS / "rebuttal_exp5_sensitivity" / "summary.json")

    with OUT_JSON.open("w") as fh:
        json.dump(data, fh, indent=2)
    print(f"wrote {OUT_JSON}")

    with OUT_MD.open("w") as fh:
        fh.write("# Rebuttal experimental numbers (5-seed mean ± std)\n\n")
        _write_classification(fh, "UCI HAR", data.get("exp1a_har_uci"))
        fh.write("## Classical HAR baselines (UCI HAR, 5 seeds)\n\n")
        _write_classical(fh, data.get("exp1b_har_classical"))
        _write_classification(fh, "PAMAP2 (cross-subject)", data.get("exp1c_har_pamap2"))
        _write_forecast(fh, data.get("exp2_forecast"))
        _write_cross(fh, "Cross-domain forecasting (H=24)", data.get("exp3_cross_forecast"))
        _write_cross(fh, "Cross-domain DA baselines (H=24)", data.get("exp3_da_forecast"))
        _write_cross(fh, "Ablation (HAR & ETTh1->ETTh2)", data.get("exp4_ablation"))
    print(f"wrote {OUT_MD}")


def _pct(v):
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}"


def _fmt_mean_std(entry, key, pct=False):
    if not entry or key not in entry:
        return "--"
    mean = entry[key].get("mean")
    std = entry[key].get("std") or 0.0
    if mean is None:
        return "--"
    if pct:
        return f"{mean*100:.2f} ± {std*100:.2f}"
    return f"{mean:.4f} ± {std:.4f}"


def _write_classification(fh, title, summary):
    if not summary:
        return
    fh.write(f"## {title} classification (5 seeds)\n\n")
    fh.write("| Model | Params (K) | Accuracy (%) | F1-Macro (%) | Train (s) |\n")
    fh.write("|---|---:|---:|---:|---:|\n")
    for model, stats in sorted(summary.items(), key=lambda kv: -(kv[1].get("accuracy") or {}).get("mean", 0)):
        params = stats.get("n_params") or 0
        fh.write(
            f"| {model} | {params/1000:.1f} "
            f"| {_fmt_mean_std(stats, 'accuracy', pct=True)} "
            f"| {_fmt_mean_std(stats, 'f1_macro', pct=True)} "
            f"| {_fmt_mean_std(stats, 'training_time')} |\n"
        )
    fh.write("\n")


def _write_classical(fh, summary):
    if not summary:
        fh.write("(classical summary missing)\n\n")
        return
    fh.write("| Classifier | Accuracy (%) | F1-Macro (%) | Train (s) |\n|---|---:|---:|---:|\n")
    for model, stats in summary.items():
        fh.write(
            f"| {model} | {_fmt_mean_std(stats, 'accuracy', pct=True)} "
            f"| {_fmt_mean_std(stats, 'f1_macro', pct=True)} "
            f"| {_fmt_mean_std(stats, 'training_time')} |\n"
        )
    fh.write("\n")


def _write_forecast(fh, summary):
    if not summary:
        return
    fh.write("## ETT forecasting (MSE, 5 seeds)\n\n")
    for ds, per_h in summary.items():
        fh.write(f"### {ds}\n\n| Model | H=24 | H=48 | H=96 |\n|---|---:|---:|---:|\n")
        all_models = sorted({m for h in per_h.values() if h for m in h})
        for model in all_models:
            row = [model]
            for h in (24, 48, 96):
                stats = (per_h.get(h) or {}).get(model)
                row.append(_fmt_mean_std(stats, "mse") if stats else "--")
            fh.write("| " + " | ".join(row) + " |\n")
        fh.write("\n")


def _write_cross(fh, title, summary):
    if not summary:
        return
    fh.write(f"## {title}\n\n")
    for label, per_model in summary.items():
        fh.write(f"### {label}\n\n| Model | MSE | MAE | Train (s) |\n|---|---:|---:|---:|\n")
        for model, stats in sorted(per_model.items()):
            fh.write(
                f"| {model} | {_fmt_mean_std(stats, 'mse')} "
                f"| {_fmt_mean_std(stats, 'mae')} "
                f"| {_fmt_mean_std(stats, 'training_time')} |\n"
            )
        fh.write("\n")


if __name__ == "__main__":
    main()
