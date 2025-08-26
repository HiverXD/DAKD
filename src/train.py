# src/train.py
from __future__ import annotations
import argparse, json, math, os, random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# --- project modules ---
from data.data_setup import loaders
from utils.trainer import Trainer
from model.backbones import (
    build_classifier,           # (ckpt_path, num_classes, freeze_backbone, device) -> nn.Module
    export_backbone_from_classifier,
)

# -------------------------------
# Config helpers
# -------------------------------
def _reuse_head_flag(cfg, who: str) -> bool:
    """
    who: "student" | "teacher"
    우선순위: train.<who>.reuse_linear_head -> train.reuse_linear_head -> True(기본)
    """
    tr = cfg.get("train", {})
    if who in tr and isinstance(tr[who], dict) and "reuse_linear_head" in tr[who]:
        return bool(tr[who]["reuse_linear_head"])
    if "reuse_linear_head" in tr:
        return bool(tr["reuse_linear_head"])
    return True


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_config(root_cfg: str, overrides: List[str]) -> dict:
    cfg_path = Path(root_cfg)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for ov in overrides or []:
        ovp = Path(ov)
        if not ovp.exists():
            raise FileNotFoundError(f"Override not found: {ov}")
        _deep_update(cfg, yaml.safe_load(ovp.read_text(encoding="utf-8")))
    return cfg

# -------------------------------
# Repro
# -------------------------------
def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# -------------------------------
# Optim/Sched builders (AdamW + warmup+cosine)
# -------------------------------
def build_optimizer(model: nn.Module, cfg: dict) -> AdamW:
    opt_cfg = cfg["train"]["optimizer"]
    wd = float(opt_cfg.get("weight_decay", 0.0))
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or n.endswith(".bias") or "bn" in n.lower() or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    return AdamW(groups,
                 lr=float(opt_cfg.get("lr", 3e-4)),
                 betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
                 eps=float(opt_cfg.get("eps", 1e-8)))

def build_scheduler(optimizer, cfg: dict, total_epochs: int):
    sch_cfg = cfg["train"]["scheduler"]
    warmup = int(sch_cfg.get("warmup_epochs", 0))
    base_lr = float(cfg["train"]["optimizer"]["lr"])
    min_lr  = float(sch_cfg.get("min_lr", 1e-6))

    def lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch + 1) / max(1, warmup)
        t = (epoch - warmup) / max(1, total_epochs - warmup)
        min_factor = min_lr / base_lr
        return min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * t))

    return LambdaLR(optimizer, lr_lambda)

# -------------------------------
# Teacher registry & helpers
# -------------------------------
DATASET2NUM = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "tiny_imagenet": 200,
}

def _ckpt_root(cfg) -> Path:
    return Path(cfg.get("paths", {}).get("ckpt_root", "src/model/ckpts"))

def _teachers_registry_path(cfg) -> Path:
    return _ckpt_root(cfg) / "teachers.json"

def _load_teachers_registry(cfg) -> dict:
    p = _teachers_registry_path(cfg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}

def _save_teachers_registry(cfg, reg: dict):
    p = _teachers_registry_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2), encoding="utf-8")

def _template_paths_for_ds(cfg, ds_key: str):
    tmpl_bb  = cfg["model"]["teacher"].get("template_backbone",  "src/model/ckpts/resnet34_{dataset}_backbone.pt")
    tmpl_cls = cfg["model"]["teacher"].get("template_classifier","src/model/ckpts/resnet34_{dataset}_classifier.pt")
    return Path(tmpl_bb.format(dataset=ds_key)), Path(tmpl_cls.format(dataset=ds_key))

def _export_and_register_teacher(model: nn.Module, cfg: dict, ds_key: str):
    """Export teacher weights to ckpt_root with dataset-specific names and update teachers.json"""
    # best 가중치로 덮어쓰기 (Trainer가 best_state_dict를 제공한다고 가정)
    bb_path, cls_path = _template_paths_for_ds(cfg, ds_key)
    bb_path.parent.mkdir(parents=True, exist_ok=True)
    cls_path.parent.mkdir(parents=True, exist_ok=True)

    # 전체 분류기 포함 가중치
    torch.save({"arch": cfg["model"]["teacher"]["arch"], "state_dict": model.state_dict()}, cls_path)

    # 백본만(FC 제거) — KD에서 feature 추출용
    export_backbone_from_classifier(model, cfg["model"]["teacher"]["arch"], str(bb_path))

    reg = _load_teachers_registry(cfg)
    reg[ds_key] = {"classifier": str(cls_path), "backbone": str(bb_path)}
    _save_teachers_registry(cfg, reg)
    print(f"[OK] Registered teacher for {ds_key}\n - {cls_path}\n - {bb_path}")

def _load_frozen_teacher_for_ds(cfg, ds_key: str, num_classes: int, device: torch.device) -> nn.Module:
    """02+ 단계: 등록된 교사 ckpt를 찾아 분류기 포함 모델을 읽고 동결"""
    bb_path, cls_path = _template_paths_for_ds(cfg, ds_key)
    reg = _load_teachers_registry(cfg)
    if cls_path.exists():
        use = cls_path
    elif reg.get(ds_key, {}).get("classifier") and Path(reg[ds_key]["classifier"]).exists():
        use = Path(reg[ds_key]["classifier"])
    else:
        # 마지막 fallback (ImageNet backbone에서 새 fc 붙인 비학습 교사) — 01이 안 돌았을 때만
        use = Path(cfg["model"]["teacher"].get("ckpt", _ckpt_root(cfg) / "ResNet34.pt"))
        print(f"[WARN] Teacher classifier for '{ds_key}' not found; fallback to {use}")
    teacher = build_classifier(str(use), num_classes, freeze_backbone=True, device=str(device))
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()
    return teacher

# -------------------------------
# Student helpers
# -------------------------------
def _student_ckpt_path(cfg) -> str:
    student = cfg.get("model", {}).get("student", {})
    if "ckpt" in student and student["ckpt"]:
        return student["ckpt"]
    # fallback to default naming
    return str(_ckpt_root(cfg) / "ResNet18.pt")

def build_student_for_stage(cfg: dict, device: torch.device, stage: str, num_classes: int) -> nn.Module:
    ckpt_path = _student_ckpt_path(cfg)
    freeze = (stage == "linear_probe")
    model = build_classifier(ckpt_path, num_classes, freeze_backbone=freeze, device=str(device))
    return model

# -------------------------------
# (Optional) KD loss factory (02+에서 사용)
# -------------------------------
def make_compute_loss_ce():
    def compute_loss(model, batch, mode, step, epoch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits
    return compute_loss

def make_compute_loss_kd(teacher: nn.Module, T: float = 4.0, alpha: float = 0.5, device: str = "cuda"):
    def compute_loss(model, batch, mode, step, epoch, _device):
        x, y = batch
        x = x.to(device); y = y.to(device)
        s_logits = model(x)
        ce = F.cross_entropy(s_logits, y)
        with torch.no_grad():
            t_logits = teacher(x)
        log_p_s = F.log_softmax(s_logits / T, dim=1)
        p_t     = F.softmax(t_logits / T, dim=1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        loss = alpha * ce + (1.0 - alpha) * kl
        return loss, s_logits
    return compute_loss

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=[])
    parser.add_argument("--role", type=str, choices=["teacher", "student", "both"], default="student",
                        help="01에서는 teacher / 02+에서는 student 권장")
    args = parser.parse_args()

    cfg = load_config(args.cfg, args.override)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # data
    train_loaders, test_loaders = loaders(
        batch_size=cfg["data"]["batch_size"],
        root=cfg["data"]["root"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        download=cfg["data"]["download"],
    )
    ds_key = cfg["data"]["dataset"]
    if ds_key not in train_loaders:
        raise ValueError(f"Unknown dataset key: {ds_key}")
    num_classes = cfg.get("model", {}).get("num_classes", DATASET2NUM[ds_key])
    train_loader, test_loader = train_loaders[ds_key], test_loaders[ds_key]

    # output dirs
    exp_name = cfg.get("experiment", {}).get("name", "exp")
    out_root = Path(cfg["output"]["dir"]) / exp_name / ds_key
    student_lp_dir = out_root / "student" / "linear_probe"
    student_ft_dir = out_root / "student" / "finetune"
    teacher_lp_dir = out_root / "teacher" / "linear_probe"
    teacher_ft_dir = out_root / "teacher" / "finetune"
    for d in [student_lp_dir, student_ft_dir, teacher_lp_dir, teacher_ft_dir]:
        d.mkdir(parents=True, exist_ok=True)
        # config snapshot for reproducibility
        (d / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # precision / logging
    use_amp = (cfg.get("precision", "fp32") != "fp32")
    log_interval = int(cfg.get("logging", {}).get("log_interval", 50))

    # ---------------- Teacher (01에서만 1회) ----------------
    if args.role in ("teacher", "both"):
        # 1) Teacher Linear Probe (백본 동결, 분류기만 학습)
        lp_head_state = None
        if cfg["train"]["teacher"]["linear_probe"]["enable"]:
            teacher_lp = build_classifier(
                cfg["model"]["teacher"]["ckpt"], num_classes,
                freeze_backbone=True, device=str(device)
            )
            opt_tlp = build_optimizer(teacher_lp, cfg)
            sch_tlp = build_scheduler(
                opt_tlp, cfg, total_epochs=int(cfg["train"]["teacher"]["linear_probe"]["epochs"])
            )
            t_trainer_lp = Trainer(
                model=teacher_lp, optimizer=opt_tlp, scheduler=sch_tlp, schedule_mode="epoch",
                device=device, train_loader=train_loader, test_loader=test_loader,
                epochs=int(cfg["train"]["teacher"]["linear_probe"]["epochs"]),
                output_dir=teacher_lp_dir, amp=use_amp, compute_loss=make_compute_loss_ce(),
                log_interval=log_interval,
            )
            t_trainer_lp.fit()

            # (선택) finetune에서 이어받기 위해 LP로 학습된 fc 가중치 보관
            if getattr(t_trainer_lp, "best_state_dict", None) is not None:
                # best로 모델 업데이트
                teacher_lp.load_state_dict(t_trainer_lp.best_state_dict, strict=True)
            lp_head_state = {k.replace("fc.", ""): v.cpu()
                            for k, v in teacher_lp.fc.state_dict().items()}  # {"weight":..., "bias":...}

        # 2) Teacher Finetune (백본 포함 전체 학습)
        if cfg["train"]["teacher"]["finetune"]["enable"]:
            teacher = build_classifier(
                cfg["model"]["teacher"]["ckpt"], num_classes,
                freeze_backbone=False, device=str(device)
            )

            # (옵션) LP에서 학습한 head를 이어받아 초기화
            if lp_head_state is not None and cfg["train"]["teacher"].get("reuse_linear_head", True):
                with torch.no_grad():
                    teacher.fc.load_state_dict(lp_head_state, strict=False)

            opt_t = build_optimizer(teacher, cfg)
            sch_t = build_scheduler(
                opt_t, cfg, total_epochs=int(cfg["train"]["teacher"]["finetune"]["epochs"])
            )
            t_trainer = Trainer(
                model=teacher, optimizer=opt_t, scheduler=sch_t, schedule_mode="epoch",
                device=device, train_loader=train_loader, test_loader=test_loader,
                epochs=int(cfg["train"]["teacher"]["finetune"]["epochs"]),
                output_dir=teacher_ft_dir, amp=use_amp, compute_loss=make_compute_loss_ce(),
                log_interval=log_interval,
            )
            t_trainer.fit()

            # best 가중치로 모델 업데이트 후 ckpts에 고정 이름으로 export + registry 업데이트
            if getattr(t_trainer, "best_state_dict", None) is not None:
                teacher.load_state_dict(t_trainer.best_state_dict, strict=True)
            _export_and_register_teacher(teacher, cfg, ds_key)

    # ---------------- Student ----------------
    if args.role in ("student", "both"):
        lp_head_state = None

        # Linear Probe (백본 동결, fc만 학습)
        if cfg["train"]["linear_probe"]["enable"]:
            student_lp = build_student_for_stage(cfg, device, "linear_probe", num_classes)
            opt = build_optimizer(student_lp, cfg)
            sch = build_scheduler(opt, cfg, total_epochs=int(cfg["train"]["linear_probe"]["epochs"]))
            s_trainer_lp = Trainer(
                model=student_lp, optimizer=opt, scheduler=sch, schedule_mode="epoch",
                device=device, train_loader=train_loader, test_loader=test_loader,
                epochs=int(cfg["train"]["linear_probe"]["epochs"]),
                output_dir=student_lp_dir, amp=use_amp, compute_loss=make_compute_loss_ce(),
                log_interval=log_interval,
            )
            s_trainer_lp.fit()

            # LP best 기준으로 모델을 업데이트한 뒤 fc만 추출해 보관
            if getattr(s_trainer_lp, "best_state_dict", None) is not None:
                student_lp.load_state_dict(s_trainer_lp.best_state_dict, strict=True)
            lp_head_state = {k.replace("fc.", ""): v.cpu()
                            for k, v in student_lp.fc.state_dict().items()}

        # Finetune (백본 포함 전체 학습)
        if cfg["train"]["finetune"]["enable"]:
            student_ft = build_student_for_stage(cfg, device, "finetune", num_classes)

            # LP에서 학습된 head를 이어받아 초기화 (스위치로 제어)
            if lp_head_state is not None and _reuse_head_flag(cfg, "student"):
                with torch.no_grad():
                    student_ft.fc.load_state_dict(lp_head_state, strict=False)

            opt = build_optimizer(student_ft, cfg)
            sch = build_scheduler(opt, cfg, total_epochs=int(cfg["train"]["finetune"]["epochs"]))
            s_trainer_ft = Trainer(
                model=student_ft, optimizer=opt, scheduler=sch, schedule_mode="epoch",
                device=device, train_loader=train_loader, test_loader=test_loader,
                epochs=int(cfg["train"]["finetune"]["epochs"]),
                output_dir=student_ft_dir, amp=use_amp, compute_loss=make_compute_loss_ce(),
                log_interval=log_interval,
            )
            s_trainer_ft.fit()


    print("[DONE] role:", args.role, " dataset:", ds_key)

if __name__ == "__main__":
    main()
