"""Reuse previously-computed experiment JSONs so the 5-seed rebuttal does not
have to retrain every single (model, seed) pair from scratch.

For each *existing* per-seed result file, rename/copy it to the corresponding
``results/rebuttal_<experiment>/`` directory using the naming convention the
rebuttal runner checks (``{model}_seed{seed}.json``). Running the rebuttal
runner afterwards will skip those seeds and only train the *missing* seeds
(typically 789 and 2024, which were not in the original 3-seed protocol).

Source -> destination mapping:

    final_exp1_har/TMTFNet_v2_seed{S}_lr0p0005.json
        -> rebuttal_exp1a_har_uci/TMTFNet_v2_seed{S}.json
    exp1_har/{Model}_seed{S}_lr0p0005.json
        -> rebuttal_exp1a_har_uci/{Model}_seed{S}.json
    final_exp2_etth1/H{P}/TMTFNet_v2_seed{S}.json
        -> rebuttal_exp2_forecast_etth1_pred{P}/TMTFNet_v2_seed{S}.json
    exp2_etth1/H{P}/{Model}_seed{S}.json
        -> rebuttal_exp2_forecast_etth1_pred{P}/{Model}_seed{S}.json
    final_exp3_cross_domain/*TMTFNet_v2_cross_seed{S}_lr0p001.json
        -> rebuttal_exp3_cross_forecast_pred24/h1_to_h2_TMTFNet_v2_seed{S}.json

Usage::

    python migrate_existing_results.py --dry-run   # just print mapping
    python migrate_existing_results.py             # actually copy
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"


def _ensure(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy(src, dst, dry_run):
    _ensure(dst)
    if dst.exists():
        print(f"  SKIP (already exists): {dst.relative_to(ROOT)}")
        return
    action = "(dry) " if dry_run else ""
    print(f"  {action}COPY {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    if not dry_run:
        shutil.copy2(src, dst)


def _standardize_har_baselines(dry_run):
    print("== HAR baseline seeds (exp1_har) ==")
    src_dir = ROOT / "exp1_har"
    if not src_dir.is_dir():
        return
    dst_dir = ROOT / "rebuttal_exp1a_har_uci"
    for path in sorted(src_dir.glob("*_seed*_lr0p0005.json")):
        m = re.match(r"(?P<model>.+)_seed(?P<seed>\d+)_lr0p0005\.json$", path.name)
        if not m:
            continue
        model = m.group("model")
        seed = m.group("seed")
        if model == "TMTFNet_v2":
            continue  # use final_ version below
        _copy(path, dst_dir / f"{model}_seed{seed}.json", dry_run)


def _standardize_har_tmtfnet(dry_run):
    print("== HAR TMTFNet seeds (final_exp1_har) ==")
    src_dir = ROOT / "final_exp1_har"
    dst_dir = ROOT / "rebuttal_exp1a_har_uci"
    if not src_dir.is_dir():
        return
    for path in sorted(src_dir.glob("TMTFNet_v2_seed*_lr0p0005.json")):
        m = re.match(r"TMTFNet_v2_seed(?P<seed>\d+)_lr0p0005\.json$", path.name)
        if not m:
            continue
        _copy(path, dst_dir / f"TMTFNet_v2_seed{m.group('seed')}.json", dry_run)


def _standardize_etth1(dry_run):
    print("== ETTh1 forecasting (exp2_etth1 + final_exp2_etth1) ==")
    for horizon in (24, 48, 96):
        dst_dir = ROOT / f"rebuttal_exp2_forecast_etth1_pred{horizon}"
        for src_sub in ("exp2_etth1", "final_exp2_etth1"):
            src_dir = ROOT / src_sub / f"H{horizon}"
            if not src_dir.is_dir():
                continue
            for path in sorted(src_dir.glob("*_seed*.json")):
                m = re.match(r"(?P<model>.+?)_seed(?P<seed>\d+)(?:_lr[0-9p]+)?\.json$", path.name)
                if not m:
                    continue
                _copy(path, dst_dir / f"{m.group('model')}_seed{m.group('seed')}.json", dry_run)


def _standardize_cross_domain(dry_run):
    print("== cross-domain forecasting (final_exp3_cross_domain) ==")
    src_dir = ROOT / "final_exp3_cross_domain"
    dst_dir = ROOT / "rebuttal_exp3_cross_forecast_pred24"
    if not src_dir.is_dir():
        return
    for path in sorted(src_dir.glob("*_cross_seed*_lr*.json")):
        m = re.match(r"(?P<model>.+)_cross_seed(?P<seed>\d+)_lr[0-9p]+\.json$", path.name)
        if not m:
            continue
        _copy(path, dst_dir / f"h1_to_h2_{m.group('model')}_seed{m.group('seed')}.json", dry_run)


def _standardize_ablation(dry_run):
    print("== ablation (final_exp4_ablation) ==")
    base = ROOT / "final_exp4_ablation"
    if not base.is_dir():
        return
    dst_dir = ROOT / "rebuttal_exp4_ablation"
    for subdir, prefix in [("har", "har"), ("cross_forecast", "cross_h1_to_h2")]:
        src_dir = base / subdir
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.glob("*_seed*_lr*.json")):
            m = re.match(r"(?P<model>TMTFNet_v2(?:_\w+)?)_seed(?P<seed>\d+)_lr[0-9p]+\.json$", path.name)
            if not m:
                continue
            _copy(path, dst_dir / f"{prefix}_{m.group('model')}_seed{m.group('seed')}.json", dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _standardize_har_baselines(args.dry_run)
    _standardize_har_tmtfnet(args.dry_run)
    _standardize_etth1(args.dry_run)
    _standardize_cross_domain(args.dry_run)
    _standardize_ablation(args.dry_run)


if __name__ == "__main__":
    main()
