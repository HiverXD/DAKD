# src/train_soft_kd.py
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from src.data.data_setup import loaders  # 첨부 파일 기준: loaders()가 (train_loaders, test_loaders) 딕셔너리 반환
from src.data.indexed import WithIndex, collate_with_index
from src.data.soft_targets import TeacherLogitsBank
from src.losses.soft_kd import SoftTargetKDLoss
from src.model.load_teacher import load_backbone_and_classifier, load_student_imagenet
from src.utils.config import load_config, expand_softkd_templates, pretty  # 첨부 파일 기준


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
    cfg = expand_softkd_templates(cfg)  # teachers.json/템플릿 경로 자동 확장
    return cfg


# ----------------------------
# Data: pick single dataset & rewrap with indices
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
# Optim / Sched (AdamW 기본, cosine)
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

    sch_cfg = cfg["train"]["scheduler"]
    warm = int(sch_cfg.get("warmup_epochs", 0))
    min_lr = float(sch_cfg.get("min_lr", 1e-6))

    total_epochs = 0
    if cfg["train"]["linear_probe"]["enable"]:
        total_epochs += int(cfg["train"]["linear_probe"]["epochs"])
    if cfg["train"]["finetune"]["enable"]:
        total_epochs += int(cfg["train"]["finetune"]["epochs"])

    def lr_lambda(epoch):
        if epoch < warm:
            return float(epoch + 1) / max(1, warm)
        if total_epochs <= warm:
            return min_lr / lr
        t = (epoch - warm) / max(1, total_epochs - warm)
        return (min_lr / lr) + (1 - (min_lr / lr)) * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda)
    return optim, scheduler


# ----------------------------
# KD helpers
# ----------------------------
@torch.no_grad()
def _teacher_logits_from_teacher(teacher, x):
    return teacher(x)

def _teacher_logits_from_bank(bank: TeacherLogitsBank, idx):
    return bank.get(idx)

def _step_kd(kd_loss_fn: SoftTargetKDLoss, student_logits, teacher_logits, targets):
    total, parts = kd_loss_fn(student_logits, teacher_logits, targets)
    # parts: {"train_loss_ce", "train_loss_kd", "train_loss"}
    return total, float(parts["train_loss_ce"]), float(parts["train_loss_kd"])


# ----------------------------
# One epoch train / eval (tqdm + 001 포맷용 통계)
# ----------------------------
def train_one_epoch(student, teacher, bank, loader, device, optimizer, kd_loss_fn, log_interval=50):
    student.train()
    tbar = tqdm(loader, desc="train", leave=False)
    total_loss_sum = ce_sum = kd_sum = correct = seen = 0

    for step, batch in enumerate(tbar):
        if len(batch) == 3: x, y, idx = batch
        else: x, y, idx = batch[0], batch[1], None
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        s_logits = student(x)
        if bank is not None and idx is not None:
            t_logits = _teacher_logits_from_bank(bank, idx)
        else:
            with torch.no_grad():
                t_logits = _teacher_logits_from_teacher(teacher, x)

        loss_total, loss_ce, loss_kd = _step_kd(kd_loss_fn, s_logits, t_logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()

        bs = x.size(0)
        seen += bs
        total_loss_sum += float(loss_total) * bs
        ce_sum += float(loss_ce) * bs
        kd_sum += float(loss_kd) * bs
        correct += (s_logits.argmax(1) == y).sum().item()

        if (step + 1) % log_interval == 0 or (step + 1) == len(loader):
            tbar.set_postfix({
                "loss": total_loss_sum / seen,
                "ce": ce_sum / seen,
                "kd": kd_sum / seen,
                "acc": correct / max(1, seen),
            })

    return {
        "train_loss": total_loss_sum / max(1, seen),
        "train_loss_ce": ce_sum / max(1, seen),
        "train_loss_kd": kd_sum / max(1, seen),
        "train_acc": correct / max(1, seen),
    }


@torch.no_grad()
def eval_one_epoch(student, teacher, loader, device, kd_loss_fn):
    student.eval()
    tbar = tqdm(loader, desc="val", leave=False)
    total_loss_sum = ce_sum = kd_sum = correct = seen = 0

    for batch in tbar:
        if len(batch) == 3: x, y, _ = batch
        else: x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        s_logits = student(x)
        t_logits = _teacher_logits_from_teacher(teacher, x)  # val은 bank 대신 teacher on-the-fly

        loss_total, loss_ce, loss_kd = _step_kd(kd_loss_fn, s_logits, t_logits, y)

        bs = x.size(0)
        seen += bs
        total_loss_sum += float(loss_total) * bs
        ce_sum += float(loss_ce) * bs
        kd_sum += float(loss_kd) * bs
        correct += (s_logits.argmax(1) == y).sum().item()

        tbar.set_postfix({
            "loss": total_loss_sum / seen,
            "ce": ce_sum / seen,
            "kd": kd_sum / seen,
            "acc": correct / max(1, seen),
        })

    return {
        "val_loss": total_loss_sum / max(1, seen),
        "val_loss_ce": ce_sum / max(1, seen),
        "val_loss_kd": kd_sum / max(1, seen),
        "val_acc": correct / max(1, seen),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = load_cfg_from_cli()
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

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

    # logits cache (train 전용)
    bank = None
    if cfg["kd"].get("cache_logits", True):
        bank = TeacherLogitsBank(
            cfg["kd"]["cache_path"],
            num_samples=len(train_loader.dataset),
            num_classes=int(cfg["model"]["num_classes"]),
            device=device,
        )
        if not bank.exists():
            cache_loader = DataLoader(
                train_loader.dataset,
                batch_size=cfg["data"]["batch_size"],
                shuffle=False,
                num_workers=cfg["data"]["num_workers"],
                pin_memory=cfg["data"].get("pin_memory", True),
                collate_fn=collate_with_index,
            )
            bank.build(teacher, cache_loader)

    kd_loss = SoftTargetKDLoss(alpha=float(cfg["kd"]["alpha"]), temperature=float(cfg["kd"]["temperature"]))
    optim, sched = build_optimizer_scheduler(cfg, student)
    log_interval = int(cfg.get("logging", {}).get("log_interval", 50))

    jsonl_path = out_dir / "metrics_per_epoch.jsonl"
    best_acc = -1.0
    global_epoch = 0

    # ---- Linear Probe ----
    if cfg["train"]["linear_probe"]["enable"]:
        for p in student.parameters(): p.requires_grad_(False)
        for p in student.fc.parameters(): p.requires_grad_(True)
        num_epochs = int(cfg["train"]["linear_probe"]["epochs"])
        for e in range(num_epochs):
            global_epoch += 1
            t0 = time.perf_counter()
            tr = train_one_epoch(student, teacher, bank, train_loader, device, optim, kd_loss, log_interval)
            va = eval_one_epoch(student, teacher, val_loader, device, kd_loss)
            if sched is not None: sched.step()
            time_sec = time.perf_counter() - t0
            lr = float(optim.param_groups[0]["lr"])

            # 001 포맷 + KD 추가 항목 (한 줄)
            line = {
                "epoch": global_epoch,
                "time_sec": time_sec,
                "train_loss": tr["train_loss"],
                "train_loss_ce": tr["train_loss_ce"],
                "train_loss_kd": tr["train_loss_kd"],
                "train_acc": tr["train_acc"],
                "val_loss": va["val_loss"],
                "val_loss_ce": va["val_loss_ce"],
                "val_loss_kd": va["val_loss_kd"],
                "val_acc": va["val_acc"],
                "lr": lr,
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")

            # ckpt
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
        num_epochs = int(cfg["train"]["finetune"]["epochs"])
        for e in range(num_epochs):
            global_epoch += 1
            t0 = time.perf_counter()
            tr = train_one_epoch(student, teacher, bank, train_loader, device, optim, kd_loss, log_interval)
            va = eval_one_epoch(student, teacher, val_loader, device, kd_loss)
            if sched is not None: sched.step()
            time_sec = time.perf_counter() - t0
            lr = float(optim.param_groups[0]["lr"])

            line = {
                "epoch": global_epoch,
                "time_sec": time_sec,
                "train_loss": tr["train_loss"],
                "train_loss_ce": tr["train_loss_ce"],
                "train_loss_kd": tr["train_loss_kd"],
                "train_acc": tr["train_acc"],
                "val_loss": va["val_loss"],
                "val_loss_ce": va["val_loss_ce"],
                "val_loss_kd": va["val_loss_kd"],
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
