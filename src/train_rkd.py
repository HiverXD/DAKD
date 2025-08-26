# src/train_rkd.py
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

# ---- data ----
from src.data.data_setup import loaders
from src.data.indexed import WithIndex, collate_with_index
from src.data.soft_targets import TeacherLogitsBank
from src.data.final_embeddings import TeacherEmbeddingBank

# ---- models / loaders ----
from src.model.load_teacher import load_backbone_and_classifier, load_student_imagenet

# ---- utils ----
from src.utils.config import load_config, expand_softkd_templates, pretty
from src.utils.schedulers import BetaScheduler, BetaScheduleCfg

# ---- losses ----
from src.losses.soft_kd import SoftTargetKDLoss
from src.losses.relational_kd import cos_gram_mse
from src.losses.compute.compute_relational_kd_loss import compute_rkd_loss

# ----------------------------
# Config CLI (root + overrides)
# ----------------------------
def load_cfg_from_cli() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True,
                        help="first is root config, the rest are overrides")
    args = parser.parse_args()
    root, overrides = args.config[0], args.config[1:]
    cfg = load_config(root, overrides)
    cfg = expand_softkd_templates(cfg)  # teachers.json/템플릿 경로 자동 확장 (soft_kd와 동일)
    return cfg

# ----------------------------
# Dataloaders (WithIndex; (x,y,idx))
# ----------------------------
def build_dataloaders_single(cfg):
    """data_setup.loaders()에서 원하는 dataset 하나만 골라, .dataset을 꺼내 WithIndex로 감싸 새 DataLoader 생성"""
    ds_key = cfg["data"]["dataset"]
    train_dict, val_dict = loaders(
        batch_size=cfg["data"]["batch_size"],
        root=cfg["data"]["root"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        download=cfg["data"].get("download", False),
    )
    if ds_key not in train_dict or ds_key not in val_dict:
        raise KeyError(f"Dataset '{ds_key}' not found. Available: {list(train_dict.keys())}")

    base_train = train_dict[ds_key].dataset
    base_val   = val_dict[ds_key].dataset

    train_ds = WithIndex(base_train)
    val_ds   = WithIndex(base_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        persistent_workers=(cfg["data"]["num_workers"] > 0),
        collate_fn=collate_with_index,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        persistent_workers=(cfg["data"]["num_workers"] > 0),
        collate_fn=collate_with_index,
    )
    return train_loader, val_loader

# ----------------------------
# Models
# ----------------------------
def build_teacher(cfg, device: torch.device):
    nc  = int(cfg["model"]["num_classes"])
    arc = cfg["model"]["teacher"]["arch"]
    tbb = cfg["model"]["teacher"]["backbone_path"]
    tcl = cfg["model"]["teacher"]["classifier_path"]
    teacher = load_backbone_and_classifier(tbb, tcl, arc, nc, map_location="cpu").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher

def build_student(cfg, device: torch.device):
    nc   = int(cfg["model"]["num_classes"])
    sckp = cfg["model"]["student"]["ckpt"]
    student = load_student_imagenet(sckp, nc, map_location="cpu").to(device)
    return student

# ----------------------------
# Optim / Sched (AdamW 기본, cosine)  — soft_kd와 동일
# ----------------------------
def build_optimizer_scheduler(cfg, model):
    opt_cfg = cfg["train"]["optimizer"]
    name = opt_cfg["name"].lower()
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg.get("weight_decay", 0.05))

    if name == "adamw":
        optim = AdamW(model.parameters(), lr=lr,
                      betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
                      eps=float(opt_cfg.get("eps", 1e-8)),
                      weight_decay=wd)
    elif name == "sgd":
        optim = SGD(model.parameters(), lr=lr,
                    momentum=float(opt_cfg.get("momentum", 0.9)),
                    weight_decay=wd, nesterov=True)
    else:
        optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # soft_kd와 동일한 스케줄 키/형식 사용
    sch_cfg = cfg["train"]["scheduler"]  # <-- 기존 lr_scheduler가 아님
    warm   = int(sch_cfg.get("warmup_epochs", 0))
    min_lr = float(sch_cfg.get("min_lr", 1e-6))

    # 총 epoch 계산: enable된 구간만 합산
    total_epochs = 0
    if cfg["train"]["linear_probe"]["enable"]:
        total_epochs += int(cfg["train"]["linear_probe"]["epochs"])
    if cfg["train"]["finetune"]["enable"]:
        total_epochs += int(cfg["train"]["finetune"]["epochs"])
    total_epochs = max(1, total_epochs)  # 0으로 나누기 안전장치

    # warmup + cosine to min_lr
    def lr_lambda(epoch):
        if epoch < warm:
            return float(epoch + 1) / max(1, warm)
        if total_epochs <= warm:
            return min_lr / lr
        t = (epoch - warm) / max(1, total_epochs - warm)
        floor = (min_lr / lr)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda)
    return optim, scheduler


# ----------------------------
# Utilities: feature hooks
# ----------------------------
class _FCInHook:
    """fc 모듈의 입력(=최종 임베딩)을 캡처하는 forward hook"""
    def __init__(self, fc: nn.Module):
        self.handle = fc.register_forward_hook(self.hook)
        self.buf = None
    def hook(self, module, inputs, output):
        x = inputs[0]
        # (B, D)
        self.buf = x.detach()
    def close(self):
        self.handle.remove()

@torch.no_grad()
def _extract_feats_for_bank(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # teacher 모델에서 임베딩만 추출 (fc 입력 캡처)
    assert hasattr(model, "fc"), "Expected a torchvision ResNet-like model with .fc"
    hk = _FCInHook(model.fc)
    _ = model(x)  # forward once
    feats = hk.buf
    hk.close()
    return feats

def forward_feats_and_logits(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """학생 모델을 한 번만 forward 해서 (feat, logits)을 함께 얻음"""
    hk = _FCInHook(model.fc)
    logits = model(x)
    feats = hk.buf
    hk.close()
    return feats, logits

# ----------------------------
# Epoch Loops
# ----------------------------
def train_one_epoch_lp(student, teacher, bank_logits, loader, device, optimizer, kd_loss_fn, log_interval=50, use_amp=False, scaler: torch.cuda.amp.GradScaler | None = None):
    """LP: RKD 비활성화, base KD만. BN eval 고정 가정."""
    student.train()
    tbar = tqdm(loader, desc="train[LP]", leave=False)
    total_loss_sum = ce_sum = kd_sum = correct = seen = 0

    for step, batch in enumerate(tbar):
        if len(batch) == 3: x, y, idx = batch
        else: x, y, idx = batch[0], batch[1], None
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            s_logits = student(x)
            if bank_logits is not None and idx is not None:
                t_logits = bank_logits.get(idx)
            else:
                with torch.no_grad():
                    t_logits = teacher(x)
            total, parts = kd_loss_fn(s_logits, t_logits, y)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        # metrics
        bs = y.size(0)
        total_loss_sum += float(parts["train_loss"]) * bs
        ce_sum += float(parts["train_loss_ce"]) * bs
        kd_sum += float(parts["train_loss_kd"]) * bs
        pred = s_logits.detach().argmax(1)
        correct += int((pred == y).sum())
        seen += bs

    return {
        "train_loss": total_loss_sum / max(1, seen),
        "train_loss_ce": ce_sum / max(1, seen),
        "train_loss_kd": kd_sum / max(1, seen),
        "train_loss_rkd": 0.0,
        "train_loss_rkd_weighted": 0.0,
        "train_acc": correct / max(1, seen),
    }

@torch.no_grad()
def eval_one_epoch_lp(student, teacher, loader, device, kd_loss_fn):
    """LP: 검증에서도 RKD 없음 (로그엔 0 기록)"""
    student.eval()
    tbar = tqdm(loader, desc="val[LP]", leave=False)
    total_loss_sum = ce_sum = kd_sum = correct = seen = 0

    for batch in tbar:
        if len(batch) == 3: x, y, _ = batch
        else: x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        s_logits = student(x)
        t_logits = teacher(x)
        total, parts = kd_loss_fn(s_logits, t_logits, y)

        bs = y.size(0)
        total_loss_sum += float(parts["train_loss"]) * bs
        ce_sum += float(parts["train_loss_ce"]) * bs
        kd_sum += float(parts["train_loss_kd"]) * bs
        pred = s_logits.detach().argmax(1)
        correct += int((pred == y).sum())
        seen += bs

    return {
        "val_loss": total_loss_sum / max(1, seen),
        "val_loss_ce": ce_sum / max(1, seen),
        "val_loss_kd": kd_sum / max(1, seen),
        "val_loss_rkd": 0.0,
        "val_loss_rkd_weighted": 0.0,
        "val_acc": correct / max(1, seen),
    }

def train_one_epoch_ft(student, teacher, bank_logits, bank_feats, loader, device, optimizer, kd_loss_fn,
                       beta_sched: BetaScheduler, epoch_idx: int, log_interval=50, use_amp=False, scaler: torch.cuda.amp.GradScaler | None = None):
    """FT: base KD + RKD(beta). bank_feats는 train split만 사용."""
    student.train()
    tbar = tqdm(loader, desc="train[FT]", leave=False)
    total_loss_sum = ce_sum = kd_sum = rkd_sum = rkdw_sum = correct = seen = 0
    steps = len(loader)

    for step, batch in enumerate(tbar):
        if len(batch) == 3: x, y, idx = batch
        else: x, y, idx = batch[0], batch[1], None
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        beta = float(beta_sched.value(epoch_idx, step))
        with torch.cuda.amp.autocast(enabled=use_amp):
            s_feats, s_logits = forward_feats_and_logits(student, x)
            if bank_logits is not None and idx is not None:
                t_logits = bank_logits.get(idx)
            else:
                with torch.no_grad():
                    t_logits = teacher(x)
            base_total, parts = kd_loss_fn(s_logits, t_logits, y)

        # RKD (skip when beta==0 for speed)
        if beta > 0.0:
            if bank_feats is not None and idx is not None:
                t_feats = bank_feats.get(idx)  # (B, D)
            else:
                with torch.no_grad():
                    t_feats = _extract_feats_for_bank(teacher, x)
            rkd = compute_rkd_loss(s_feats, t_feats, beta)
            total = base_total + rkd["loss_rkd_weighted"]
        else:
            rkd = {"loss_rkd": torch.tensor(0.0, device=device),
                   "loss_rkd_weighted": torch.tensor(0.0, device=device)}
            total = base_total

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        # metrics
        bs = y.size(0)
        total_loss_sum += float(parts["train_loss"]) * bs + float(rkd["loss_rkd_weighted"].detach()) * bs
        ce_sum += float(parts["train_loss_ce"]) * bs
        kd_sum += float(parts["train_loss_kd"]) * bs
        rkd_sum += float(rkd["loss_rkd"].detach()) * bs
        rkdw_sum += float(rkd["loss_rkd_weighted"].detach()) * bs
        pred = s_logits.detach().argmax(1)
        correct += int((pred == y).sum())
        seen += bs

    return {
        "train_loss": total_loss_sum / max(1, seen),
        "train_loss_ce": ce_sum / max(1, seen),
        "train_loss_kd": kd_sum / max(1, seen),
        "train_loss_rkd": rkd_sum / max(1, seen),
        "train_loss_rkd_weighted": rkdw_sum / max(1, seen),
        "train_acc": correct / max(1, seen),
    }

@torch.no_grad()
def eval_one_epoch_ft(student, teacher, loader, device, kd_loss_fn, beta_for_log: float):
    """FT: 검증은 base KD + (옵션) RKD 측정. 로깅 편의상 beta_for_log를 받아 weighted 값도 보고."""
    student.eval()
    tbar = tqdm(loader, desc="val[FT]", leave=False)
    total_loss_sum = ce_sum = kd_sum = rkd_sum = rkdw_sum = correct = seen = 0

    for batch in tbar:
        if len(batch) == 3: x, y, _ = batch
        else: x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        s_feats, s_logits = forward_feats_and_logits(student, x)
        t_logits = teacher(x)  # val은 on-the-fly
        base_total, parts = kd_loss_fn(s_logits, t_logits, y)

        # RKD (on-the-fly)
        t_feats = _extract_feats_for_bank(teacher, x)
        rkd = compute_rkd_loss(s_feats, t_feats, beta_for_log if beta_for_log>0 else 0.0)

        total = base_total + (rkd["loss_rkd_weighted"] if beta_for_log>0 else 0.0)

        bs = y.size(0)
        total_loss_sum += float(base_total.detach()) * bs + (float(rkd["loss_rkd_weighted"].detach()) if beta_for_log>0 else 0.0) * bs
        ce_sum += float(parts["train_loss_ce"]) * bs
        kd_sum += float(parts["train_loss_kd"]) * bs
        rkd_sum += float(rkd["loss_rkd"].detach()) * bs
        rkdw_sum += float(rkd["loss_rkd_weighted"].detach()) * bs if beta_for_log>0 else 0.0
        pred = s_logits.detach().argmax(1)
        correct += int((pred == y).sum())
        seen += bs

    return {
        "val_loss": total_loss_sum / max(1, seen),
        "val_loss_ce": ce_sum / max(1, seen),
        "val_loss_kd": kd_sum / max(1, seen),
        "val_loss_rkd": rkd_sum / max(1, seen),
        "val_loss_rkd_weighted": rkdw_sum / max(1, seen),
        "val_acc": correct / max(1, seen),
    }

# ----------------------------
# Main
# ----------------------------
def main():
    cfg = load_cfg_from_cli()
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    use_amp = str(cfg.get("precision", "fp32")).lower() == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 출력 경로: runs/<experiment.name>/<dataset>
    exp_name = cfg["experiment"]["name"]
    ds_key   = cfg["data"]["dataset"]
    out_dir  = Path(cfg["output"]["dir"]) / exp_name / ds_key
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(pretty(cfg), encoding="utf-8")

    # data / models
    train_loader, val_loader = build_dataloaders_single(cfg)
    teacher = build_teacher(cfg, device)
    student = build_student(cfg, device)

    # banks (train-only)
    bank_logits = None
    if cfg.get("kd", {}).get("enable", True) and cfg["kd"].get("cache_logits", True):
        bank_logits = TeacherLogitsBank(
            cfg["kd"]["cache_path"],
            num_samples=len(train_loader.dataset),
            num_classes=int(cfg["model"]["num_classes"]),
            device=device,
        )
        if not bank_logits.exists():
            cache_loader = DataLoader(
                train_loader.dataset,
                batch_size=cfg["data"]["batch_size"],
                shuffle=False,
                num_workers=cfg["data"]["num_workers"],
                pin_memory=cfg["data"].get("pin_memory", True),
                collate_fn=collate_with_index,
            )
            bank_logits.build(teacher, cache_loader)

    bank_feats = None
    if cfg.get("rkd", {}).get("enable", True):
        embed_path = cfg["rkd"].get("embed_path") or cfg["rkd"].get("embed_template", "src/model/ckpts/final_embeddings/{dataset}_fp16.pt").format(dataset=ds_key)
        feat_dim = int(cfg["rkd"].get("feat_dim", student.fc.in_features))
        bank_feats = TeacherEmbeddingBank(
            embed_path,
            num_samples=len(train_loader.dataset),
            feat_dim=feat_dim,
            device=device,
        )
        if not bank_feats.exists():
            cache_loader = DataLoader(
                train_loader.dataset,
                batch_size=cfg["data"]["batch_size"],
                shuffle=False,
                num_workers=cfg["data"]["num_workers"],
                pin_memory=cfg["data"].get("pin_memory", True),
                collate_fn=collate_with_index,
            )
            bank_feats.build(teacher, cache_loader, feature_extractor=_extract_feats_for_bank)

    kd_loss = SoftTargetKDLoss(alpha=float(cfg["kd"]["alpha"]), temperature=float(cfg["kd"]["temperature"]))
    optim, sched = build_optimizer_scheduler(cfg, student)
    log_interval = int(cfg.get("logging", {}).get("log_interval", 50))

    jsonl_path = out_dir / "metrics_per_epoch.jsonl"
    best_acc = -1.0
    global_epoch = 0

    # ---- Linear Probe ----
    if cfg["train"]["linear_probe"]["enable"]:
        # Freeze backbone, fc only
        for p in student.parameters(): p.requires_grad_(False)
        for p in student.fc.parameters(): p.requires_grad_(True)
        num_epochs = int(cfg["train"]["linear_probe"]["epochs"])
        for e in range(num_epochs):
            global_epoch += 1
            t0 = time.perf_counter()
            tr = train_one_epoch_lp(student, teacher, bank_logits, train_loader, device, optim, kd_loss, log_interval, use_amp, scaler)
            va = eval_one_epoch_lp(student, teacher, val_loader, device, kd_loss)
            if sched is not None: sched.step()
            time_sec = time.perf_counter() - t0
            lr = float(optim.param_groups[0]["lr"])

            line = {
                "epoch": global_epoch,
                "time_sec": time_sec,
                "beta": 0.0,
                "train_loss": tr["train_loss"],
                "train_loss_ce": tr["train_loss_ce"],
                "train_loss_kd": tr["train_loss_kd"],
                "train_loss_rkd": tr["train_loss_rkd"],
                "train_loss_rkd_weighted": tr["train_loss_rkd_weighted"],
                "train_acc": tr["train_acc"],
                "val_loss": va["val_loss"],
                "val_loss_ce": va["val_loss_ce"],
                "val_loss_kd": va["val_loss_kd"],
                "val_loss_rkd": va["val_loss_rkd"],
                "val_loss_rkd_weighted": va["val_loss_rkd_weighted"],
                "val_acc": va["val_acc"],
                "lr": lr,
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")

            torch.save(student.state_dict(), out_dir / "checkpoints" / "last.pt")
            if va["val_acc"] > best_acc:
                best_acc = va["val_acc"]
                torch.save(student.state_dict(), out_dir / "checkpoints" / "best.pt")
                (out_dir / "best_metrics.json").write_text(
                    json.dumps({"epoch": global_epoch, "val_acc": best_acc}), encoding="utf-8"
                )

    # ---- Finetune ----
    if cfg["train"]["finetune"]["enable"]:
        for p in student.parameters(): p.requires_grad_(True)

        # Beta schedule
        bs_cfg = cfg.get("rkd", {}).get("beta_schedule", {})
        bcfg = BetaScheduleCfg(
            start_epoch=int(bs_cfg.get("start_epoch", 21)),
            end_epoch=int(bs_cfg.get("end_epoch", 30)),
            base=float(bs_cfg.get("base", 50.0)),
            stepwise=bool(bs_cfg.get("stepwise", True)),
        )
        beta_sched = BetaScheduler(bcfg, steps_per_epoch=max(1, len(train_loader)))

        num_epochs = int(cfg["train"]["finetune"]["epochs"])
        for e in range(num_epochs):
            global_epoch += 1
            t0 = time.perf_counter()
            tr = train_one_epoch_ft(student, teacher, bank_logits, bank_feats, train_loader, device, optim, kd_loss, beta_sched, global_epoch, log_interval, use_amp, scaler)
            current_beta = beta_sched.value(global_epoch, len(train_loader)-1)  # epoch 끝에서의 beta
            va = eval_one_epoch_ft(student, teacher, val_loader, device, kd_loss, beta_for_log=float(current_beta))
            if sched is not None: sched.step()
            time_sec = time.perf_counter() - t0
            lr = float(optim.param_groups[0]["lr"])

            line = {
                "epoch": global_epoch,
                "time_sec": time_sec,
                "beta": float(current_beta),
                "train_loss": tr["train_loss"],
                "train_loss_ce": tr["train_loss_ce"],
                "train_loss_kd": tr["train_loss_kd"],
                "train_loss_rkd": tr["train_loss_rkd"],
                "train_loss_rkd_weighted": tr["train_loss_rkd_weighted"],
                "train_acc": tr["train_acc"],
                "val_loss": va["val_loss"],
                "val_loss_ce": va["val_loss_ce"],
                "val_loss_kd": va["val_loss_kd"],
                "val_loss_rkd": va["val_loss_rkd"],
                "val_loss_rkd_weighted": va["val_loss_rkd_weighted"],
                "val_acc": va["val_acc"],
                "lr": lr,
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")

            torch.save(student.state_dict(), out_dir / "checkpoints" / "last.pt")
            if va["val_acc"] > best_acc:
                best_acc = va["val_acc"]
                torch.save(student.state_dict(), out_dir / "checkpoints" / "best.pt")
                (out_dir / "best_metrics.json").write_text(
                    json.dumps({"epoch": global_epoch, "val_acc": best_acc}), encoding="utf-8"
                )

if __name__ == "__main__":
    main()
