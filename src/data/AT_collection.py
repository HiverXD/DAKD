#src/AT_collection.py
"""
Attention-Transfer (AT) collection utilities

Goals
-----
- Iterate over train/valid loaders and collect AT (p=2) vectors from
  specific intermediate blocks of teacher and student models.
- Store results to Torch .pt files under:
    src/model/ckpts/AT/{dataset}.pt
    src/model/ckpts/AT/{dataset}_val.pt
- Keep strict idx alignment with WithIndex-based loaders (like logit bank).
- Save only AT(p=2) to minimize size (raw features are NOT stored).
- Provide a simple reader API similar to a bank: get(model_key, layer_key, idx).

Default tap points
------------------
- Teacher: stages 2, 4, 6, 8  (mapped to ["layer1","layer2","layer3","layer4"], see notes)
- Student: stages 1, 2, 3, 4  (mapped to ["layer1","layer2","layer3","layer4"]) 

note on layer naming
--------------------
This module collects from module attribute paths (e.g., "layer1", "layer2").
If you specifically require teacher taps at (2,4,6,8), set teacher_taps to
["layer1","layer2","layer3","layer4"]. For torchvision ResNet-style
architectures, these correspond to stage outputs and typically match
student stages 1..4 in spatial resolution.
"""
from __future__ import annotations
import os
import time
from typing import Dict, List, Optional, Tuple

import argparse
import torch
import torch.nn as nn
from src.model.load_teacher import load_backbone_and_classifier, load_student_imagenet
from src.data.data_setup import loaders
from src.data.indexed import WithIndex, collate_with_index
from src.utils.config import load_config, expand_softkd_templates

# -----------------------------
# Helpers: module resolution & hooks
# -----------------------------

def _resolve_module(root: nn.Module, dotted_path: str) -> nn.Module:
    mod = root
    for part in dotted_path.split("."):
        if not hasattr(mod, part):
            raise AttributeError(f"Module has no attribute '{part}' while resolving '{dotted_path}'")
        mod = getattr(mod, part)
    if not isinstance(mod, nn.Module):
        raise TypeError(f"Resolved object at '{dotted_path}' is not an nn.Module: {type(mod)}")
    return mod


class _TapManager:
    """Registers forward-hooks on given module attribute paths and captures outputs.

    Captured outputs are detached tensors on CPU. Use `.pop_all()` after each
    forward to retrieve a dict[layer_key] -> tensor(B, C, H, W).
    """
    def __init__(self, model: nn.Module, tap_names: List[str]):
        self.model = model
        self.tap_names = list(tap_names)
        self._buffers: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        for name in self.tap_names:
            module = _resolve_module(self.model, name)
            handle = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)

    def _make_hook(self, key: str):
        def _hook(_mod, _inp, out):
            # Ensure tensor output
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not torch.is_tensor(out):
                raise TypeError(f"Hooked output at '{key}' is not a Tensor: {type(out)}")
            self._buffers[key] = out.detach().to("cpu")
        return _hook

    def pop_all(self) -> Dict[str, torch.Tensor]:
        out = self._buffers
        self._buffers = {}
        return out

    def close(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()


# -----------------------------
# AT (p=2) computation
# -----------------------------

def at_p2(h: torch.Tensor) -> torch.Tensor:
    """Compute Attention Transfer map with p=2.

    A^2 = sum_c |h_c|^2 over channel dimension.
    Input:  h : (B, C, H, W)
    Return: (B, H, W) float32
    """
    if h.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got {tuple(h.shape)}")
    # Compute in float32 for stability
    h32 = h.to(dtype=torch.float32)
    # Sum of squares across channels
    a = (h32 * h32).sum(dim=1)
    return a  # (B, H, W)


def _vectorize_map(m: torch.Tensor, l2_normalize: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """Flatten spatial map to (B, H*W) and (optional) L2-normalize per sample."""
    B = m.size(0)
    v = m.reshape(B, -1)
    if l2_normalize:
        norms = v.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        v = v / norms
    return v


# -----------------------------
# Bank writer/reader
# -----------------------------

class ATBankWriter:
    """Accumulate AT vectors for multiple layers and save as a .pt bank.

    This writer keeps CPU float32 buffers during collection and converts to
    float16 for disk when saving. It supports arbitrary per-layer lengths (H*W).
    """
    def __init__(self, total_samples: int):
        self.n = int(total_samples)
        self._stores: Dict[str, torch.Tensor] = {}   # key -> (N, D) float32
        self._shapes: Dict[str, Tuple[int, int]] = {}  # key -> (H, W)

    def ensure_storage(self, key: str, H: int, W: int):
        D = H * W
        if key not in self._stores:
            self._stores[key] = torch.empty((self.n, D), dtype=torch.float32)
            self._shapes[key] = (H, W)
        else:
            # Validate consistent shapes across batches
            H0, W0 = self._shapes[key]
            if (H0, W0) != (H, W):
                raise ValueError(f"AT shape mismatch for '{key}': existing {(H0,W0)} vs new {(H,W)}")

    @torch.no_grad()
    def write_batch(self, key: str, idx: torch.Tensor, vec: torch.Tensor):
        """Place vectors into rows given by idx (idx on CPU or CUDA, vec on CPU)."""
        if vec.dim() != 2:
            raise ValueError("Expected 2D vec (B, D)")
        if idx.dim() != 1:
            idx = idx.view(-1)
        # Move idx to CPU long
        idx_cpu = idx.detach().to("cpu", dtype=torch.long)
        if idx_cpu.min() < 0 or idx_cpu.max() >= self.n:
            raise IndexError(f"Index out of range in write_batch: min={idx_cpu.min().item()}, max={idx_cpu.max().item()}, n={self.n}")
        # Assign
        self._stores[key][idx_cpu] = vec  # float32

    def save(self, path: str, meta: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert to fp16 on disk
        pack: Dict[str, torch.Tensor] = {}
        for k, buf in self._stores.items():
            pack[k] = buf.to(dtype=torch.float16)
        obj = {
            "at": pack,                    # dict[key] = (N, D) half
            "meta": {
                **meta,
                "n": self.n,
                "shapes": self._shapes,   # dict[key] -> (H, W)
                "created_at": time.time(),
                "dtype": "float16",
                "p": 2,
            },
        }
        torch.save(obj, path)


class ATBankReader:
    """Reader for the saved AT bank .pt.

    Provides `.get(model_key, layer_key, idx)` -> (B, D) float32 on device.
    Keys are composed as f"{model_key}_{layer_key}".
    Example: model_key="teacher", layer_key="layer2" => key "teacher_layer2".
    """
    def __init__(self, path: str, device: Optional[torch.device] = None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"AT bank not found: {path}")
        self.path = path
        self.device = device or torch.device("cpu")
        self._obj = torch.load(self.path, map_location="cpu")
        self._at: Dict[str, torch.Tensor] = self._obj["at"]
        self.meta: Dict = self._obj.get("meta", {})
        self.n = int(self.meta.get("n", 0))

    def keys(self) -> List[str]:
        return list(self._at.keys())

    def get(self, model_key: str, layer_key: str, idx: torch.Tensor) -> torch.Tensor:
        key = f"{model_key}_{layer_key}"
        if key not in self._at:
            raise KeyError(f"AT bank missing key '{key}'. Available: {self.keys()}")
        mat_half = self._at[key]  # (N, D) half on CPU
        # Gather rows
        idx_cpu = idx.detach().to("cpu", dtype=torch.long)
        if idx_cpu.min() < 0 or idx_cpu.max() >= mat_half.size(0):
            raise IndexError(f"Index out of range in AT get: min={idx_cpu.min().item()}, max={idx_cpu.max().item()}, n={mat_half.size(0)})")
        out = mat_half[idx_cpu].to(dtype=torch.float32, device=self.device)
        return out  # (B, D) float32


# -----------------------------
# Collection main routines
# -----------------------------

@torch.no_grad()
def collect_at_for_loader(
    model: nn.Module,
    model_key: str,
    tap_names: List[str],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    writer: ATBankWriter,
    l2_normalize: bool = True,
):
    """Iterate over loader and write AT(p=2) vectors into writer for given taps.

    The loader must yield (x, y, idx). AT vectors are placed at rows given by idx
    (strict idx alignment).
    """
    if len(tap_names) == 0:
        return
    tap = _TapManager(model, tap_names)
    try:
        model_was_training = model.training
        model.eval()
        for batch in loader:
            if len(batch) < 3:
                raise RuntimeError("Loader must yield (x, y, idx). Wrap dataset with WithIndex.")
            x, _y, idx = batch[0], batch[1], batch[2]
            x = x.to(device, non_blocking=True)
            # Forward (hooks capture)
            _ = model(x)
            feats = tap.pop_all()  # dict[layer_name] -> (B, C, H, W)
            for lname, h in feats.items():
                if h.dim() != 4:
                    raise ValueError(f"Captured feature for '{lname}' is not 4D: {tuple(h.shape)}")
                B, C, H, W = h.shape
                writer.ensure_storage(key=f"{model_key}_{lname}", H=H, W=W)
                amap = at_p2(h)                # (B, H, W) float32
                avec = _vectorize_map(amap, l2_normalize=l2_normalize)  # (B, H*W)
                writer.write_batch(key=f"{model_key}_{lname}", idx=idx, vec=avec)
        # Restore mode
        if model_was_training:
            model.train()
    finally:
        tap.close()


def collect_and_save_at(
    dataset_key: str,
    split: str,
    teacher: Optional[nn.Module],
    student: Optional[nn.Module],
    loader: torch.utils.data.DataLoader,
    out_dir: str = "src/model/ckpts/AT",
    teacher_taps: Optional[List[str]] = None,
    student_taps: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    l2_normalize: bool = True,
    extra_meta: Optional[Dict] = None,
) -> str:
    """Collect AT(p=2) for given split and save a single .pt bank file.

    Returns the saved file path.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if teacher_taps is None:
        teacher_taps = ["layer1", "layer2", "layer3", "layer4"]
    if student_taps is None:
        student_taps = ["layer1", "layer2", "layer3", "layer4"]

    # Prepare writer
    n = len(loader.dataset)
    writer = ATBankWriter(total_samples=n)

    if teacher is not None and len(teacher_taps) > 0:
        collect_at_for_loader(
            model=teacher, model_key="teacher", tap_names=teacher_taps,
            loader=loader, device=device, writer=writer, l2_normalize=l2_normalize,
        )

    if student is not None and len(student_taps) > 0:
        collect_at_for_loader(
            model=student, model_key="student", tap_names=student_taps,
            loader=loader, device=device, writer=writer, l2_normalize=l2_normalize,
        )

    # Build output path
    suffix = "_val" if split.lower().startswith("val") or split.lower().startswith("valid") else ""
    fname = f"{dataset_key}{suffix}.pt"
    out_path = os.path.join(out_dir, fname)

    # Save
    meta = {
        "dataset": dataset_key,
        "split": split,
        "teacher_taps": teacher_taps,
        "student_taps": student_taps,
        "l2_normalize": l2_normalize,
    }
    if extra_meta:
        meta.update(extra_meta)
    writer.save(out_path, meta=meta)
    return out_path


def collect_and_save_at_for_splits(
    dataset_key: str,
    teacher: Optional[nn.Module],
    student: Optional[nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    out_dir: str = "src/model/ckpts/AT",
    teacher_taps: Optional[List[str]] = None,
    student_taps: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    l2_normalize: bool = True,
    extra_meta: Optional[Dict] = None,
) -> Tuple[str, str]:
    """Collect AT(p=2) for both train and val splits, returning (train_path, val_path)."""
    tr_path = collect_and_save_at(
        dataset_key=dataset_key, split="train",
        teacher=teacher, student=student, loader=train_loader,
        out_dir=out_dir, teacher_taps=teacher_taps, student_taps=student_taps,
        device=device, l2_normalize=l2_normalize, extra_meta=extra_meta,
    )
    va_path = collect_and_save_at(
        dataset_key=dataset_key, split="val",
        teacher=teacher, student=student, loader=val_loader,
        out_dir=out_dir, teacher_taps=teacher_taps, student_taps=student_taps,
        device=device, l2_normalize=l2_normalize, extra_meta=extra_meta,
    )
    return tr_path, va_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True,
                    help="config 파일 경로 여러 개 (--config root.yaml --config ds.yaml)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # config 불러오기
    cfg = load_config(args.config[0], args.config[1:])
    cfg = expand_softkd_templates(cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = cfg["data"]["dataset"]
    num_classes = cfg["model"]["num_classes"]

    train_loaders, valid_loaders = loaders()

    ds = cfg["data"]["dataset"]
    tr = train_loaders[ds]
    va = valid_loaders[ds]

    # WithIndex 래핑 (shuffle=False, logit bank와 같은 규약)
    from src.data.indexed import WithIndex, collate_with_index
    train_loader = torch.utils.data.DataLoader(
        WithIndex(tr.dataset),
        batch_size=tr.batch_size, shuffle=False,
        num_workers=tr.num_workers, pin_memory=True,
        collate_fn=collate_with_index,
    )
    val_loader = torch.utils.data.DataLoader(
        WithIndex(va.dataset),
        batch_size=va.batch_size, shuffle=False,
        num_workers=va.num_workers, pin_memory=True,
        collate_fn=collate_with_index,
    )

    # (2) teacher 경로: 우선 확장된 값(backbone_path/classifier_path)을 쓰되,
    #     없으면 템플릿에서 dataset을 format 해서 만들도록 fallback
    tch = cfg["model"]["teacher"]
    bp  = tch.get("backbone_path")
    cp  = tch.get("classifier_path")
    if bp is None or cp is None:
        # 템플릿 기반 fallback (build_banks.py와 동일한 방식)
        bp = tch["template_backbone"].format(dataset=ds)
        cp = tch["template_classifier"].format(dataset=ds)

    from src.model.load_teacher import load_backbone_and_classifier, load_student_imagenet

    teacher = load_backbone_and_classifier(
        bp, cp, tch["arch"], cfg["model"]["num_classes"]
    ).to(device).eval()

    # (3) student 경로: 이 리포는 student에 backbone_path를 두지 않고 ckpt만 씀
    #     → load_student_imagenet에 ckpt를 넘겨야 함
    std = cfg["model"]["student"]
    student = load_student_imagenet(
        std["ckpt"], cfg["model"]["num_classes"]
    ).to(device).eval()

    # AT bank 생성
    tr_path, va_path = collect_and_save_at_for_splits(
        dataset_key=ds,
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir="src/model/ckpts/AT",
        device=device,
    )
    print("AT bank saved:", tr_path, va_path)

if __name__ == "__main__":
    main()