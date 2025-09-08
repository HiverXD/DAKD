# temp_at_check.py
"""
Quick checker that the saved AT bank (p=2) matches on-the-fly AT from the teacher
for the same dataset indices.

Usage example
-------------
python temp_at_check.py \
  --config config.yaml \
  --config experiments/02_soft_kd/configs/cifar10.yaml \
  --device cuda \
  --split train \
  --max_samples 1024 \
  --tolerance 2e-3

It prints per-layer relative errors and asserts they are within tolerance.
The bank path defaults to: src/model/ckpts/AT/{dataset}{|_val}.pt

Notes
-----
- Requires that AT banks were generated via `python -m src.AT_collection ...`.
- Loader uses WithIndex and shuffle=False to match the bank's address space.
- We only compare TEACHER layers (keys: teacher_layer1..4 by default).
"""
from __future__ import annotations
import argparse
import os
from typing import List, Dict

import torch
import torch.nn as nn

from src.utils.config import load_config, expand_softkd_templates
from src.data.data_setup import loaders
from src.data.indexed import WithIndex, collate_with_index
from src.model.load_teacher import load_backbone_and_classifier
from src.AT_collection import ATBankReader, at_p2, _TapManager, _vectorize_map


def _default_taps() -> List[str]:
    return ["layer1", "layer2", "layer3", "layer4"]


def _bank_path(dataset: str, split: str) -> str:
    suffix = "_val" if split.lower().startswith("val") or split.lower().startswith("valid") else ""
    return os.path.join("src/model/ckpts/AT", f"{dataset}{suffix}.pt")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="train", choices=["train","val","valid","validation"]) 
    ap.add_argument("--tolerance", type=float, default=2e-3)  # allow small diff due to fp16 storage
    ap.add_argument("--max_samples", type=int, default=1024)
    ap.add_argument("--teacher_taps", nargs="*", default=None,
                    help="Module names to tap, default: layer1 layer2 layer3 layer4")
    ap.add_argument("--bank_path", default=None)
    args = ap.parse_args()

    # Load & expand config
    cfg = load_config(args.config[0], args.config[1:])
    cfg = expand_softkd_templates(cfg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = cfg["data"]["dataset"]
    num_classes = cfg["model"]["num_classes"]
    taps = args.teacher_taps or _default_taps()

    # Prepare loaders (follow AT_collection pattern)
    train_dict, val_dict = loaders()
    if args.split.lower().startswith("train"):
        base = train_dict[ds]
    else:
        base = val_dict[ds]

    loader = torch.utils.data.DataLoader(
        WithIndex(base.dataset),
        batch_size=min(getattr(base, "batch_size", 128), args.max_samples),
        shuffle=False,  # keep deterministic order for direct idx addressing
        num_workers=getattr(base, "num_workers", 4),
        pin_memory=True,
        collate_fn=collate_with_index,
    )

    # Teacher
    tch = cfg["model"]["teacher"]
    bp = tch.get("backbone_path")
    cp = tch.get("classifier_path")
    if bp is None or cp is None:
        bp = tch["template_backbone"].format(dataset=ds)
        cp = tch["template_classifier"].format(dataset=ds)
    teacher = load_backbone_and_classifier(bp, cp, tch["arch"], num_classes).to(device).eval()

    # Bank path
    bank_path = args.bank_path or _bank_path(ds, args.split)
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"AT bank file not found: {bank_path}")
    bank = ATBankReader(bank_path, device=device)

    # Hook and compare for up to max_samples
    tap = _TapManager(teacher, taps)

    total = 0
    layer_relerrs: Dict[str, float] = {f"teacher_{t}": 0.0 for t in taps}

    try:
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            idx = idx.to(device)
            _ = teacher(x)  # forward to fill hooks
            feats = tap.pop_all()  # dict[layer] -> (B,C,H,W) on CPU

            B = x.size(0)
            total += B
            for lname, h in feats.items():
                with torch.no_grad():
                    amap = at_p2(h)                        # (B,H,W) float32 (CPU)
                    normalize = bool(bank.meta.get("l2_normalize", False))
                    avec = _vectorize_map(amap, normalize)  # (B,H*W) float32 (CPU)
                    bank_vec = bank.get("teacher", lname, idx)  # (B,H*W) float32 (device)
                    live_vec = avec.to(device)
                    # relative error per batch
                    num = (live_vec - bank_vec).norm(p=2)
                    den = (live_vec.norm(p=2) + 1e-8)
                    rel = (num / den).item()
                    layer_relerrs[f"teacher_{lname}"] += rel * B
            if total >= args.max_samples:
                break
    finally:
        tap.close()

    # Average and report
    print("=== AT bank vs on-the-fly teacher check ===")
    ok = True
    for k in sorted(layer_relerrs.keys()):
        avg_rel = layer_relerrs[k] / max(1, total)
        print(f"{k}: rel_err={avg_rel:.3e}")
        if avg_rel > args.tolerance:
            ok = False
    print(f"checked_samples={total}, tolerance={args.tolerance}")
    if not ok:
        raise AssertionError("AT bank mismatch exceeds tolerance for at least one layer.")
    print("OK: AT bank matches teacher outputs within tolerance.")


if __name__ == "__main__":
    main()
