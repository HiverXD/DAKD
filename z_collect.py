# z_collect.py
"""
Z-space collector:
- 데이터셋별(split별)로 최대 N(클래스 균등) 샘플을 수집
- 모델 4종(teacher, student, student_kd, student_rkd) 각각에 대해, 지정 레이어에서 훅으로 h(B,C,H,W) 캡처
- 스타일 3종(AAP/GAP, AT(sum of squares), MHRP)로 변환 → per-sample L2 정규화 → HDF5에 append 저장
- logits/margin 저장 (옵션)
- 수집 후 layer×class 프로토타입 μ 저장

요구사항:
- experiments/analysis/config.yaml 존재
- src/data/data_setup.py: loaders()
- src/data/indexed.py: WithIndex, collate_with_index
- 체크포인트 경로 템플릿은 config에 명시 (runs/.../best.pt)

실행:
    python z_collect.py --cfg experiments/analysis/config.yaml
"""

from __future__ import annotations
import os, sys, time, math, json, yaml, argparse, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import h5py
from tqdm import tqdm

# -----------------------------
# Path / imports (repo root assumed)
# -----------------------------
REPO = Path(__file__).resolve().parents[1]  # tools/ -> repo root
sys.path.append(str(REPO))
sys.path.append(str(REPO / "src"))

# ---- repo modules (확인 완료) ----
from src.data.data_setup import loaders as data_loaders  # (train_dict, test_dict) 반환
from src.data.indexed import WithIndex, collate_with_index

# torchvision resnets (체크포인트는 state_dict만 저장됨: train_soft_kd.py 참고)
try:
    from torchvision.models import resnet18, resnet34
except Exception as e:
    import torchvision
    resnet18 = torchvision.models.resnet18
    resnet34 = torchvision.models.resnet34


# -----------------------------
# Config
# -----------------------------
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -----------------------------
# Utils: model build & state dict normalize
# -----------------------------
def _strip_prefix(k: str) -> str:
    for p in ("module.", "ema.", "backbone.", "encoder."):
        if k.startswith(p):
            return k[len(p):]
    return k

def _normalize_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {_strip_prefix(k): v for k, v in sd.items()}

def _extract_state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj

def build_resnet(arch: str, num_classes: int) -> nn.Module:
    a = arch.lower()
    if a in ("resnet18", "r18"):
        m = resnet18(num_classes=num_classes)
    elif a in ("resnet34", "r34"):
        m = resnet34(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m

def load_classifier_from_ckpt(arch: str, ckpt_path: str | Path, num_classes: int, device: torch.device) -> nn.Module:
    m = build_resnet(arch, num_classes).to(device)
    if not Path(ckpt_path).exists():
        print(f"[WARN] checkpoint not found: {ckpt_path} (initialized random weights)")
        return m.eval()

    # PyTorch >=2.4 권장 플래그. 구버전 호환을 위해 try/fallback
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(str(ckpt_path), map_location="cpu")

    def _extract_state_dict(obj):
        if isinstance(obj, dict) and "state_dict" in obj:
            return obj["state_dict"]
        return obj

    sd = _normalize_state_dict(_extract_state_dict(obj))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[load] {Path(ckpt_path).name}: missing={len(missing)} unexpected={len(unexpected)}")
    return m.eval()

# -----------------------------
# ResNet block enumeration & hook
# -----------------------------
def enumerate_basicblocks(resnet: nn.Module) -> List[nn.Module]:
    blocks = []
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        lyr = getattr(resnet, name)
        for b in lyr:
            blocks.append(b)
    return blocks

def ordinal_to_indices_0based(ordinals_1based: List[int]) -> List[int]:
    return [o-1 for o in ordinals_1based]

# -----------------------------
# Transforms: AAP / AT / MHRP
# -----------------------------
@torch.no_grad()
def to_gap(h: torch.Tensor, l2: bool=True) -> torch.Tensor:
    # (B,C,H,W) -> (B,C); compute in float32 for stability/compat with AMP
    x = h.float().mean(dim=(2,3))
    return F.normalize(x, p=2, dim=1) if l2 else x

@torch.no_grad()
def to_at(h: torch.Tensor, target_hw: Optional[Tuple[int,int]], reduce: str="sum_sq", l2: bool=True) -> torch.Tensor:
    # (B,C,H,W) -> (B,H*W); do resize BEFORE flatten; compute in float32
    H, W = h.shape[-2], h.shape[-1]
    hf = h.float()
    if target_hw is not None and (H, W) != target_hw:
        hf = F.interpolate(hf, size=target_hw, mode="bilinear", align_corners=False, antialias=True)
    if reduce == "sum_abs":
        a = hf.abs().sum(dim=1)     # (B,Ht,Wt)
    else:
        a = (hf ** 2).sum(dim=1)
    a = a.flatten(1)                # (B,Ht*Wt)
    return F.normalize(a, p=2, dim=1) if l2 else a

def make_Q(C: int, k: int, seed: int, device: torch.device, qr_mode: str="reduced") -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    A = torch.randn(C, k, generator=gen, device=device, dtype=torch.float32)
    Q, R = torch.linalg.qr(A, mode=qr_mode)  # (C,k) float32
    diag = torch.sign(torch.diag(R))
    Q = Q * diag
    return Q  # float32

@torch.no_grad()
def to_mhrp_from_gap(gap_x: torch.Tensor, C_in: int, k: int, seeds: List[int],
                     device: torch.device, pre_norm=True, post_norm=True,
                     agg="mean", qr_mode="reduced") -> torch.Tensor:
    """
    입력: GAP (B,C) — ANY dtype (half/float), 내부는 float32로 처리
    출력: (B,k) float32 (writer가 최종 FP16로 저장)
    """
    x = gap_x.float()                           # promote to float32
    if pre_norm:
        x = F.normalize(x, p=2, dim=1)

    outs = []
    for s in seeds:
        Q = make_Q(C_in, k, s, device=device, qr_mode=qr_mode)  # float32
        z = x @ Q                                               # float32 @ float32
        if post_norm:
            z = F.normalize(z, p=2, dim=1)
        outs.append(z)

    if agg == "concat":
        z_out = torch.cat(outs, dim=1)
    else:
        z_out = torch.stack(outs, dim=0).mean(dim=0)
        if post_norm:
            z_out = F.normalize(z_out, p=2, dim=1)

    return z_out  # float32

# -----------------------------
# HDF5 Writer
# -----------------------------
class H5Writer:
    def __init__(self, h5_path: str | Path, dataset: str, split: str, compression="gzip", chunk_rows=256):
        self.h5 = h5py.File(str(h5_path), "a")
        if "meta" not in self.h5:
            g = self.h5.create_group("meta")
            g.attrs["dataset"] = dataset
            g.attrs["split"] = split
            g.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            g.attrs["normalize"] = "l2"
            g.attrs["dtype"] = "float16"
        self.compression = compression
        self.chunk_rows = int(chunk_rows)

    def _grp(self, path: str) -> h5py.Group:
        return self.h5.require_group(path)

    def _append_arr(self, grp: h5py.Group, name: str, arr: np.ndarray):
        arr = np.asarray(arr)
        if name not in grp:
            maxshape = (None,) + arr.shape[1:]
            chunks = (min(self.chunk_rows, max(arr.shape[0],1)),) + arr.shape[1:]
            ds = grp.create_dataset(name, data=arr, maxshape=maxshape, chunks=chunks, compression=self.compression)
        else:
            ds = grp[name]
            ds.resize((ds.shape[0] + arr.shape[0],) + ds.shape[1:])
            ds[-arr.shape[0]:] = arr

    def append_features(self, model_tag: str, style: str, layer_idx: int,
                        emb: torch.Tensor, class_id: torch.Tensor, sample_id: torch.Tensor,
                        sub: Optional[str]=None):
        base = f"/features/{model_tag}/{style}/layer_{layer_idx}"
        if sub: base = f"{base}/{sub}"
        g = self._grp(base)
        emb_np = emb.detach().cpu().to(torch.float16).numpy()
        cls_np = class_id.detach().cpu().numpy().astype(np.int16).reshape(-1,1)
        sid_np = sample_id.detach().cpu().numpy().astype(np.int32).reshape(-1,1)
        self._append_arr(g, "emb", emb_np)
        self._append_arr(g, "class_id", cls_np)
        self._append_arr(g, "sample_id", sid_np)

    def append_logits(self, model_tag: str, logits: torch.Tensor, labels: Optional[torch.Tensor]):
        base = f"/logits/{model_tag}"
        g = self._grp(base)
        lg = logits.detach().cpu().to(torch.float16).numpy()
        self._append_arr(g, "logits", lg)
        if labels is not None:
            y = labels.detach().cpu().numpy()
            rows = np.arange(lg.shape[0])
            gt = lg[rows, y]
            mask = np.ones_like(lg, dtype=bool); mask[rows, y] = False
            max_others = (lg.astype(np.float32)[mask].reshape(lg.shape[0], -1)).max(axis=1)
            margin = (gt.astype(np.float32) - max_others).reshape(-1,1).astype(np.float16)
            self._append_arr(g, "margin", margin)

    def save_proto(self, model_tag: str, style: str, layer_idx: int, mu: np.ndarray):
        base = f"/prototypes/{model_tag}/{style}/layer_{layer_idx}"
        g = self._grp(base)
        if "mu" in g: del g["mu"]
        g.create_dataset("mu", data=mu.astype(np.float32), compression=self.compression)

    def save_Q(self, model_tag: str, layer_idx: int, head_idx: int, Q: torch.Tensor, seed: int):
        base = f"/Q/{model_tag}/layer_{layer_idx}/head_{head_idx}"
        g = self._grp(base)
        if "Q" in g: del g["Q"]
        g.create_dataset("Q", data=Q.detach().cpu().to(torch.float32).numpy(), compression=self.compression)
        g.attrs["seed"] = int(seed)

    def close(self):
        self.h5.close()

# -----------------------------
# Helper: AT target sizes (teacher policy)
# -----------------------------
@torch.no_grad()
def get_teacher_hw_per_layer(model: nn.Module, tap_indices: List[int], sample_input: torch.Tensor, device: torch.device) -> Dict[int, Tuple[int,int]]:
    model.eval().to(device)
    blocks = enumerate_basicblocks(model)
    caches: Dict[int, torch.Size] = {}
    hooks = []
    def mk_cb(li):
        def _cb(m, i, o): caches[li] = o.shape  # (B,C,H,W)
        return _cb
    for li in tap_indices:
        hooks.append(blocks[li].register_forward_hook(mk_cb(li)))
    _ = model(sample_input.to(device))
    for h in hooks: h.remove()
    sizes = {li: (int(shp[2]), int(shp[3])) for li, shp in caches.items()}
    return sizes

# -----------------------------
# Class-balanced sampling
# -----------------------------
def class_balanced_mask(labels: torch.Tensor, counters: np.ndarray, target_per_class: int) -> torch.Tensor:
    y_np = labels.detach().cpu().numpy()
    keep = np.zeros_like(y_np, dtype=bool)
    for i, y in enumerate(y_np):
        if counters[y] < target_per_class:
            keep[i] = True
            counters[y] += 1
    return torch.from_numpy(keep)

# -----------------------------
# Prototypes
# -----------------------------
def compute_and_save_prototypes(h5_path: str | Path, model_tags: List[str], styles: List[str], num_classes: int):
    with h5py.File(str(h5_path), "a") as h5:
        for mt in model_tags:
            for st in styles:
                grp = h5.get(f"/features/{mt}/{st}")
                if grp is None: continue
                for layer_name in grp.keys():
                    g = grp[layer_name]
                    if "emb" not in g:
                        if "mean" in g and "emb" in g["mean"]:
                            g = g["mean"]
                        else:
                            continue
                    emb = g["emb"][:]                 # (N, K)
                    cls = g["class_id"][:].reshape(-1)
                    K = emb.shape[1]
                    mu = np.zeros((num_classes, K), dtype=np.float32)
                    for c in range(num_classes):
                        idx = np.where(cls==c)[0]
                        if len(idx) > 0:
                            mu[c] = emb[idx].astype(np.float32).mean(axis=0)
                    li = int(layer_name.split("_")[-1])
                    dst = f"/prototypes/{mt}/{st}/layer_{li}"
                    if dst in h5: del h5[dst]
                    gg = h5.require_group(dst)
                    gg.create_dataset("mu", data=mu, compression="gzip")
                    gg.attrs["num_classes"] = int(num_classes)
                    gg.attrs["dim"] = int(K)
                    print(f"[proto] {mt}/{st}/layer_{li} -> {mu.shape}")

# -----------------------------
# Build loaders (using src/data/data_setup.loaders)
# -----------------------------
def build_loader_for(dataset_key: str, split: str, batch_size: int, root: str, num_workers: int, pin_memory: bool, download: bool) -> DataLoader:
    """
    data_setup.loaders()는 (train_dict, test_dict) 딕셔너리 반환.
    각 dict에서 dataset_key를 골라, .dataset을 WithIndex로 감싸고 새 DataLoader 생성.
    """
    
    train_dict, test_dict = data_loaders(batch_size=batch_size, root=root, num_workers=num_workers, pin_memory=pin_memory, download=download)

    if split not in {"train", "valid", "val", "test"}:
        raise ValueError("split must be in {train, valid/val, test}")
    split_key = "train" if split=="train" else "test"

    dd = train_dict if split_key=="train" else test_dict
    if dataset_key not in dd:
        raise KeyError(f"Dataset '{dataset_key}' not found in loaders(). Available: {list(dd.keys())}")

    base_ds = dd[dataset_key].dataset  # 원본 Dataset
    ds = WithIndex(base_ds)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split_key=="train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_with_index,
        drop_last=False,
    )
    return loader

def infer_num_classes(dataset_key) -> int:
    num_classes = {"cifar10": 10, "cifar100": 100, "stl10": 10, "tiny_imagenet": 200}
    return num_classes[dataset_key]

# -----------------------------
# Main collect routine (single dataset/split)
# -----------------------------
@torch.no_grad()
def collect_for_dataset_split(cfg: dict, dataset_key: str, split: str):
    device = torch.device(cfg.get("compute", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg.get("compute", {}).get("amp", True))

    # Storage
    h5_tpl = cfg.get("storage", {}).get("h5_path_tpl", "runs/z_store/{dataset}_{split}.h5")
    h5_path = Path(h5_tpl.format(dataset=dataset_key, split=split))
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    writer = H5Writer(h5_path, dataset_key, split,
                      compression=cfg.get("storage", {}).get("compression", "gzip"),
                      chunk_rows=int(cfg.get("storage", {}).get("chunk_rows", 256)))

    # Loader
    dcfg = cfg.get("data", {})
    loader = build_loader_for(
        dataset_key=dataset_key,
        split=split,
        batch_size=int(dcfg.get("dataloader", {}).get("batch_size", 128)),
        root=dcfg.get("root", "dataset"),
        num_workers=int(dcfg.get("dataloader", {}).get("num_workers", 4)),
        pin_memory=bool(dcfg.get("dataloader", {}).get("pin_memory", True)),
        download=bool(dcfg.get("download", False)),
    )
    num_classes = infer_num_classes(dataset_key)

    # Sampling
    max_per_split = int(dcfg.get("max_per_split", 1000))
    target_per_class = max(1, max_per_split // num_classes)
    counters = np.zeros(num_classes, dtype=int)
    class_balanced = bool(dcfg.get("class_balanced", True))

    # Models (4개)
    mcfgs = cfg.get("models", {})
    models_def = []  # list of (tag, arch, model, tap_idx, ckpt_path)
    for tag in ["teacher", "student", "student_kd", "student_rkd"]:
        if tag not in mcfgs: 
            continue
        m = mcfgs[tag]
        arch = m["arch"]
        ckpt_tpl = m["ckpt_tpl"]
        ckpt_path = ckpt_tpl.format(dataset=dataset_key)
        tap_idx = ordinal_to_indices_0based(m["tap_blocks_1based"])
        model = load_classifier_from_ckpt(arch, ckpt_path, num_classes, device)
        models_def.append((tag, arch, model, tap_idx, ckpt_path))

    # AT sizes from teacher (resize_policy == "teacher")
    at_cfg = cfg.get("transforms", {}).get("at", {})
    at_enabled = bool(at_cfg.get("enabled", True))
    resize_policy = at_cfg.get("resize_policy", "teacher")
    teacher_sizes: Dict[int, Tuple[int,int]] = {}
    if at_enabled and resize_policy == "teacher":
        # one tiny batch
        for images, labels, idxs in loader:
            sample = images[:min(2, images.shape[0])].to(device)
            tdef = next(((t,a,m,ti,cp) for (t,a,m,ti,cp) in models_def if t=="teacher"), None)
            if tdef is not None:
                tag, arch, model, tap_idx, _ = tdef
                teacher_sizes = get_teacher_hw_per_layer(model, tap_idx, sample, device)
            break
        print("[AT] teacher target HW per layer:", teacher_sizes)

    # MHRP params
    mhrp_cfg = cfg.get("transforms", {}).get("mhrp", {})
    mhrp_enabled = bool(mhrp_cfg.get("enabled", True))
    mhrp_heads = int(mhrp_cfg.get("heads", 4))
    mhrp_seeds = list(mhrp_cfg.get("seeds", [1,2,3,4]))[:mhrp_heads]
    mhrp_k = int(mhrp_cfg.get("k", 256))
    qr_mode = mhrp_cfg.get("qr_mode", "reduced")
    pre_norm = bool(mhrp_cfg.get("pre_norm", True))
    post_norm = bool(mhrp_cfg.get("post_norm", True))
    agg = mhrp_cfg.get("agg", "mean")
    store_Q = bool(mhrp_cfg.get("store_Q", False))

    # AAP config
    aap_cfg = cfg.get("transforms", {}).get("aap", {})
    aap_enabled = bool(aap_cfg.get("enabled", True))
    aap_l2 = bool(aap_cfg.get("l2_normalize", True))

    # Style list
    styles = []
    if aap_enabled: styles.append("aap")
    if at_enabled:  styles.append("at")
    if mhrp_enabled: styles.append("mhrp")

    # Hooks per model
    latest_h: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
    hooks = {}
    for (tag, arch, model, tap_idx, _) in models_def:
        blocks = enumerate_basicblocks(model)
        hooks[tag] = []
        for li in tap_idx:
            def make_cb(tg, layer_i):
                def _cb(m, i, o):
                    latest_h[tg][layer_i] = o.detach()
                return _cb
            hooks[tag].append(blocks[li].register_forward_hook(make_cb(tag, li)))

    # Loop
    every = int(cfg.get("logging", {}).get("every_n_batches", 10))
    seen = 0
    for bidx, batch in enumerate(tqdm(loader, desc=f"{dataset_key}/{split} collecting")):
        images, labels, idxs = batch  # collate_with_index 보장: (x, y, idx)
        # class-balanced selection
        if class_balanced:
            mask = class_balanced_mask(labels, counters, target_per_class)
            if mask.sum().item() == 0:
                continue
            images, labels, idxs = images[mask], labels[mask], idxs[mask]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        # Done?
        if counters.sum() >= target_per_class * num_classes:
            break

        # For each model
        for (tag, arch, model, tap_idx, _) in models_def:
            latest_h[tag].clear()
            # forward once (AMP)
            with torch.autocast(device_type=device.type, dtype=torch.float16) if (amp and device.type=="cuda") else torch.no_grad():
                logits = model(images)
            # logits/margin
            if bool(cfg.get("save", {}).get("logits", True)):
                writer.append_logits(tag, logits, labels if bool(cfg.get("save", {}).get("margin", True)) else None)

            # each tapped layer
            for li in tap_idx:
                if li not in latest_h[tag]:
                    continue
                h = latest_h[tag][li]  # (B,C,H,W)
                B, C, H, W = h.shape

                # AAP
                if aap_enabled:
                    gap = to_gap(h, l2=aap_l2)      # (B,C)
                    writer.append_features(tag, "aap", li, gap, labels, idxs)

                # AT
                if at_enabled:
                    target_hw = None
                    pol = at_cfg.get("resize_policy", "teacher")
                    if pol == "teacher":
                        target_hw = teacher_sizes.get(li, None)
                    elif pol == "fixed" and at_cfg.get("fixed_hw"):
                        fhw = at_cfg.get("fixed_hw")
                        target_hw = (int(fhw[0]), int(fhw[1]))
                    elif pol == "larger":
                        # simple fallback: keep current (H,W); 더 정교한 larger 정책 필요시 teacher_sizes와 비교 추가 가능
                        target_hw = (H, W)
                    at_vec = to_at(h, target_hw, reduce=at_cfg.get("reduce","sum_sq"),
                                   l2=bool(at_cfg.get("l2_normalize", True)))
                    writer.append_features(tag, "at", li, at_vec, labels, idxs)

                # MHRP (from GAP)
                if mhrp_enabled:
                    gap_for_rp = to_gap(h, l2=pre_norm)  # pre_norm 설정 반영
                    z = to_mhrp_from_gap(gap_for_rp, C_in=C, k=mhrp_k, seeds=mhrp_seeds,
                                         device=device, pre_norm=pre_norm, post_norm=post_norm,
                                         agg=agg, qr_mode=qr_mode)
                    writer.append_features(tag, "mhrp", li, z, labels, idxs)
                    if store_Q:
                        for hi, seed in enumerate(mhrp_seeds):
                            Q = make_Q(C, mhrp_k, seed, device=device, qr_mode=qr_mode)
                            writer.save_Q(tag, li, hi, Q, seed)

        seen += images.shape[0]
        if (bidx+1) % every == 0:
            print(f"[{dataset_key}/{split}] seen={seen} / target={target_per_class * num_classes}")

        if counters.sum() >= target_per_class * num_classes:
            print(f"[{dataset_key}/{split}] reached target {counters.sum()}")
            break

    # Prototypes
    if bool(cfg.get("save", {}).get("prototypes", {}).get("enabled", True)):
        styles_to_save = cfg.get("save", {}).get("prototypes", {}).get("styles", ["aap","at","mhrp"])
        compute_and_save_prototypes(h5_path, [t for (t,_,_,_,_) in models_def], styles_to_save, num_classes)

    # remove hooks
    for tag, hs in hooks.items():
        for h in hs:
            try: h.remove()
            except: pass

    writer.close()
    print(f"[DONE] {dataset_key}/{split} -> {h5_path}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="experiments/analysis/config.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)

    # seed
    seed = int(cfg.get("compute", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    datasets = cfg.get("data", {}).get("datasets", [])
    splits = cfg.get("data", {}).get("splits", ["train","valid"])
    # valid 명칭 통일: data_setup는 test loader가 "val" 역할
    splits = ["train" if s=="train" else "valid" if s=="valid" else s for s in splits]
    splits = [s if s!="valid" else "valid" for s in splits]  # 그대로 둠

    # 내부적으로 build_loader_for에서 split=="valid"를 "test"로 매핑하므로 그대로 전달
    for ds in datasets:
        for sp in splits:
            collect_for_dataset_split(cfg, ds, sp)

if __name__ == "__main__":
    main()
