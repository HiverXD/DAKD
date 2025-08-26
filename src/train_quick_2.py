# src/train_quick_2.py
import os, json, math, copy, time, argparse, contextlib
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import load_config, expand_softkd_templates, save_config_used
from src.data.soft_targets import TeacherLogitsBank
from src.losses.soft_kd import SoftTargetKDLoss
from src.losses.compute.compute_relational_kd_loss_quick_2 import make_compute_relational_kd_loss

# ---------- helpers ----------
DATASET_NUM_CLASSES = {
    "cifar10": 10, "cifar100": 100, "stl10": 10, "tiny_imagenet": 200,
}

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _jsonl_write(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _trimmed_mean(xs: List[float], p: float = 0.05) -> float:
    if not xs: return float("nan")
    xs = sorted(xs)
    k = int(len(xs) * p)
    xs = xs[k:len(xs)-k] if len(xs) > 2*k else xs
    return float(sum(xs) / max(1, len(xs)))

def _median(xs: List[float]) -> float:
    if not xs: return float("nan")
    xs = sorted(xs)
    n = len(xs)
    if n % 2: return float(xs[n//2])
    return float(0.5*(xs[n//2-1] + xs[n//2]))

# Try robust feature extraction for torchvision resnet18/34
def extract_feats_and_logits(model: nn.Module, x: torch.Tensor):
    # fast path: torchvision resnet structure
    m = model
    try:
        y = m.conv1(x); y = m.bn1(y); y = m.relu(y); y = m.maxpool(y)
        y = m.layer1(y); y = m.layer2(y); y = m.layer3(y); y = m.layer4(y)
        y = m.avgpool(y)
        feats = torch.flatten(y, 1)
        logits = m.fc(feats)
        return feats, logits
    except Exception:
        pass
    # fallback: capture input to fc via a forward-pre hook
    feats_buf = {}
    def _hook(mod, inp):
        feats_buf["z"] = inp[0].detach()
    handle = None
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and name.endswith("fc"):
            handle = mod.register_forward_pre_hook(_hook)
            break
    logits = m(x)
    if handle is not None:
        handle.remove()
    feats = feats_buf.get("z", None)
    if feats is None:
        raise RuntimeError("Could not extract penultimate features; please implement a model-specific path.")
    return feats, logits

# import build utilities from 02 pipeline
# (local import to avoid eager import side-effects)
def _imports_from_02():
    from src.train_soft_kd import build_dataloaders_single, build_teacher, build_student, build_optimizer_scheduler
    return build_dataloaders_single, build_teacher, build_student, build_optimizer_scheduler

# ---------- training core ----------
def run_quick2_for_dataset(root_cfg: Dict, dataset: str, epochs: int = 3,
                           out_root: str = "runs/_quick_2",
                           a: float = None, T: float = None,
                           eps: float = 1e-6, b_clip: float = 1e3):
    cfg = copy.deepcopy(root_cfg)
    cfg["data"]["dataset"] = dataset
    cfg["model"]["num_classes"] = DATASET_NUM_CLASSES[dataset]
    # KD on
    cfg["kd"]["enable"] = True
    if a is not None: cfg["kd"]["alpha"] = float(a)
    if T is not None: cfg["kd"]["temperature"] = float(T)
    # force simple schedule: finetune-only for speed (3 epochs)
    cfg["train"]["linear_probe"]["enable"] = False
    cfg["train"]["finetune"]["enable"] = True
    cfg["train"]["finetune"]["epochs"] = int(epochs)
    # resolve teacher/student/template paths
    cfg = expand_softkd_templates(cfg)

    # outputs
    out_dir = _ensure_dir(os.path.join(out_root, dataset))
    save_config_used(cfg, os.path.join(out_dir, "config_used.yaml"))
    bi_step_path = os.path.join(out_dir, "bi_per_step.jsonl")
    ep_metrics_path = os.path.join(out_dir, "metrics_per_epoch.jsonl")

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    precision = cfg.get("precision", "fp32")
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

    # imports from 02
    build_dataloaders_single, build_teacher, build_student, build_optimizer_scheduler = _imports_from_02()
    train_loader, val_loader = build_dataloaders_single(cfg)  # val_loader는 여기선 사용하지 않음

    # teacher/student
    teacher = build_teacher(cfg, device)
    student = build_student(cfg, device)

    # logits bank for train
    num_samples = len(train_loader.dataset)
    bank = TeacherLogitsBank(
        path=cfg["kd"].get("cache_path", cfg["kd"]["cache_template"].format(dataset=dataset)),
        num_samples=num_samples, num_classes=cfg["model"]["num_classes"],
        device=device
    )
    if not bank.exists():
        # 캐시 구축: 반드시 shuffle=False 로더 사용 (함수 내부에서 처리하는 구현이라면 생략)
        # 여기서는 02의 내부 구현을 따른다고 가정
        bank.build(teacher, loader=train_loader)

    # losses
    kd_loss = SoftTargetKDLoss(alpha=cfg["kd"]["alpha"], temperature=cfg["kd"]["temperature"])
    compute_rkd = make_compute_relational_kd_loss(center=True, offdiag_only=True)

    # optimizer/scheduler
    optim, sched = build_optimizer_scheduler(cfg, student)

    global_step = 0
    for epoch in range(1, epochs+1):
        student.train()
        t0 = time.time()
        b_list = []
        ce_sum = kd_sum = rkd_sum = base_sum = 0.0
        n_seen = 0

        for batch in train_loader:
            # WithIndex loader: (idx, x, y)
            x, y, idx = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(precision == "fp16")):
                # student
                s_feat, s_logits = extract_feats_and_logits(student, x)

                # teacher logits from bank (no grad)
                t_logits = bank.get(idx)  # (B, C), likely fp32 tensor already on device

                # teacher feats on-the-fly (no grad)
                with torch.no_grad():
                    t_feat, _ = extract_feats_and_logits(teacher, x)

                # losses
                loss_ce = F.cross_entropy(s_logits, y)
                loss_soft, parts_kd = kd_loss(s_logits, t_logits, y)  # returns (total, dict) if implemented that way
                if isinstance(loss_soft, tuple):  # compatibility
                    loss_soft, parts_kd = loss_soft

                loss_base = (1.0 - cfg["kd"]["alpha"]) * loss_ce + cfg["kd"]["alpha"] * loss_soft
                loss_rkd, parts_rkd = compute_rkd(s_feat, t_feat)

                # bi (clipped)
                bi = 0.3 * (loss_base.detach() / (loss_rkd.detach() + eps))
                bi = torch.clamp(bi, min=1e-3, max=b_clip)
                bi_scalar = float(bi.cpu().item())
                b_list.append(bi_scalar)

                # optimize with base loss only (quick measurement)
                loss = loss_base

            optim.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            # sched (per-step cosine warmup 지원 시)
            if sched is not None and getattr(sched, "step_per_iteration", False):
                sched.step()

            # logging per-step
            ce_scalar = float(loss_ce.detach().cpu().item())
            kd_scalar = float(loss_soft.detach().cpu().item())
            rkd_scalar = float(loss_rkd.detach().cpu().item())
            base_scalar = float(loss_base.detach().cpu().item())

            _jsonl_write(bi_step_path, {
                "epoch": epoch, "step": global_step,
                "train_loss_ce": ce_scalar,
                "train_loss_kd": kd_scalar,
                "train_loss_base": base_scalar,
                "train_loss_rkd": rkd_scalar,
                "b_i": bi_scalar,
                "lr": float(optim.param_groups[0]["lr"]),
            })

            ce_sum += ce_scalar
            kd_sum += kd_scalar
            rkd_sum += rkd_scalar
            base_sum += base_scalar
            n_seen += 1
            global_step += 1

        # epoch scheduler
        if sched is not None and not getattr(sched, "step_per_iteration", False):
            sched.step()

        # epoch summary
        elapsed = time.time() - t0
        b_mean = float(sum(b_list) / max(1, len(b_list)))
        b_med  = _median(b_list)
        b_tmean = _trimmed_mean(b_list, p=0.05)

        _jsonl_write(ep_metrics_path, {
            "epoch": epoch, "time_sec": elapsed,
            "train_loss_ce": ce_sum / max(1, n_seen),
            "train_loss_kd": kd_sum / max(1, n_seen),
            "train_loss_base": base_sum / max(1, n_seen),
            "train_loss_rkd": rkd_sum / max(1, n_seen),
            "b_mean": b_mean, "b_median": b_med, "b_trimmed_mean": b_tmean,
            "num_steps": n_seen,
        })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", action="append", required=True,
                        help="config.yaml (root) and optional overrides (like 02)")
    parser.add_argument("--datasets", nargs="*", default=["cifar10","cifar100","stl10","tiny_imagenet"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", type=str, default="runs/_quick_2")
    parser.add_argument("--alpha", type=float, default=None, help="override kd.alpha (a)")
    parser.add_argument("--temperature", type=float, default=None, help="override kd.temperature (T)")
    args = parser.parse_args()
    root, overrides = args.config[0], args.config[1:]
    cfg = load_config(root, overrides)
    for ds in args.datasets:
        run_quick2_for_dataset(cfg, dataset=ds, epochs=args.epochs,
                               out_root=args.out, a=args.alpha, T=args.temperature)

if __name__ == "__main__":
    main()
