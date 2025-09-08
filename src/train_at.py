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

from src.data.attention_maps import activation_to_attention, vectorize_attention_map
from src.data.AT_collection import ATBankReader  # (N,D) 벡터 뱅크 리더
from src.losses.attention_transfer import VectorizedATLoss

def register_resnet_block_hooks(model, student_layers=(1,2,3,4)):
    feats = {}
    handles = []
    blocks = {1:model.layer1, 2:model.layer2, 3:model.layer3, 4:model.layer4}
    for li in student_layers:
        def _make(li):
            def hook(_m, _in, out): feats[li] = out
            return hook
        handles.append(blocks[li].register_forward_hook(_make(li)))
    return feats, handles

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

def build_kd_loss_fn(cfg: dict) -> SoftTargetKDLoss:
    kd_cfg = cfg.get("kd", {})
    T = float(kd_cfg.get("temperature", 1.0))
    # gamma_soft_kd 최우선, 없으면 gamma/alpha 하위호환
    gamma = kd_cfg.get("gamma_soft_kd", kd_cfg.get("gamma", None))
    alpha = kd_cfg.get("alpha", None)
    return SoftTargetKDLoss(temperature=T, gamma=gamma, alpha=alpha)

def _derive_val_cache_path(train_path: str) -> str:
    # 예: foo_fp16.pt -> foo_val_fp16.pt
    if train_path.endswith(".pt"):
        base = train_path[:-3]
        return f"{base}_val.pt" if base.endswith("_fp16") else f"{base}_val_fp16.pt"
    return train_path + "_val_fp16.pt"

def maybe_build_banks(cfg, teacher, train_loader, val_loader, device):
    kd_cfg = cfg.get("kd", {})
    use_cache_train = bool(kd_cfg.get("cache_logits", True))
    use_cache_val   = bool(kd_cfg.get("cache_val_logits", False))

    train_bank = None
    val_bank   = None

    if use_cache_train:
        train_cache_path = kd_cfg["cache_path"]  # 기존 expand에서 채워짐
        train_bank = TeacherLogitsBank(train_cache_path)
        if not train_bank.exists():
            # teacher는 eval/stop-grad 전제
            train_bank.build(teacher, train_loader, device=device)
            assert train_bank.meta["n"] == len(train_loader.dataset), "bank N mismatch"
            assert train_bank.meta["c"] == cfg["model"]["num_classes"], "bank C mismatch"

    if use_cache_val:
        # val 경로 파생(설정에 별도 키가 있으면 우선 사용)
        val_cache_path = kd_cfg.get("cache_path_val", _derive_val_cache_path(kd_cfg["cache_path"]))
        val_bank = TeacherLogitsBank(val_cache_path)
        if not val_bank.exists():
            val_bank.build(teacher, val_loader, device=device)

    return train_bank, val_bank


# ----------------------------
# One epoch train / eval (tqdm + 001 포맷용 통계)
# ----------------------------
def train_one_epoch(student, teacher, bank, loader, device, optimizer, kd_loss_fn,
                    log_interval=50, require_bank: bool = False,
                    use_at: bool = False, at_bank=None, at_crit=None, at_p: int = 2,
                    student_layers=(1,2,3,4), gamma_kd: float = 0.0, grad_clip: float = 1.0):
    student.train()
    tbar = tqdm(loader, desc="train", leave=False)
    total_loss_sum = ce_sum = kd_sum = at_sum = correct = seen = 0
    ce_fn = torch.nn.CrossEntropyLoss()

    # 훅 등록(AT 활성 시)
    if use_at:
        feats, handles = register_resnet_block_hooks(student, student_layers)
    else:
        feats, handles = {}, []

    for step, batch in enumerate(tbar):
        if len(batch) == 3: x, y, idx = batch
        else: x, y, idx = batch[0], batch[1], None
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        idx = idx.to(device) if idx is not None else None

        s_logits = student(x)

        # KD (gamma_kd==0이면 완전 스킵)
        if gamma_kd > 0:
            if bank is not None and idx is not None:
                t_logits = _teacher_logits_from_bank(bank, idx)
            else:
                if require_bank:
                    raise RuntimeError("Train logits cache required but not found (bank=None).")
                with torch.no_grad():
                    t_logits = _teacher_logits_from_teacher(teacher, x)
            loss_total, loss_ce, loss_kd = _step_kd(kd_loss_fn, s_logits, t_logits, y)
        else:
            loss_ce = ce_fn(s_logits, y).float()
            loss_kd = s_logits.new_zeros([])
            loss_total = loss_ce

        # AT (벡터 기반)  — hook으로 받은 raw feats 사용
        if use_at:
            student_feats = {f"layer{k}": feats[k] for k in student_layers}   # raw [B,C,H,W]
            loss_at_scaled, at_layer_dict = at_crit(student_feats, at_bank, idx, split="train", p=at_p)
            loss_total = loss_total + loss_at_scaled
            at_unweighted = sum(at_layer_dict.values())  # 로그용(가중치 미적용 합)
        else:
            at_unweighted = s_logits.new_zeros([])

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()

        bs = x.size(0)
        seen += bs
        total_loss_sum += float(loss_total) * bs
        ce_sum += float(loss_ce) * bs
        kd_sum += float(loss_kd) * bs
        at_sum += float(at_unweighted) * bs
        correct += (s_logits.argmax(1) == y).sum().item()

        if (step + 1) % log_interval == 0 or (step + 1) == len(loader):
            tbar.set_postfix({
                "loss": total_loss_sum / seen,
                "ce": ce_sum / seen,
                "kd": kd_sum / seen,
                "at": at_sum / seen,
                "acc": correct / max(1, seen),
            })

    # 훅 해제
    for h in handles: h.remove()

    return {
        "train_loss": total_loss_sum / max(1, seen),
        "train_loss_ce": ce_sum / max(1, seen),
        "train_loss_kd": kd_sum / max(1, seen),
        "train_loss_at": at_sum / max(1, seen),
        "train_acc": correct / max(1, seen),
    }

@torch.no_grad()
def eval_one_epoch(student, teacher, loader, device, kd_loss_fn,
                   bank=None, require_bank: bool = False,
                   use_at: bool = False, at_bank=None, at_crit=None, at_p: int = 2,
                   student_layers=(1,2,3,4), gamma_kd: float = 0.0):
    student.eval()
    tbar = tqdm(loader, desc="val", leave=False)
    total_loss_sum = ce_sum = kd_sum = at_sum = correct = seen = 0
    ce_fn = torch.nn.CrossEntropyLoss()

    # 훅(평가도 켜면 동일 경로)
    if use_at:
        feats, handles = register_resnet_block_hooks(student, student_layers)
    else:
        feats, handles = {}, []

    for batch in tbar:
        if len(batch) == 3:
            x, y, idx = batch
        else:
            x, y = batch[:2]; idx = None
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        idx = idx.to(device) if idx is not None else None

        s_logits = student(x)

        if gamma_kd > 0:
            if bank is not None and idx is not None:
                t_logits = _teacher_logits_from_bank(bank, idx)
            else:
                if require_bank:
                    raise RuntimeError("Validation logits cache required but not found (bank=None).")
                t_logits = _teacher_logits_from_teacher(teacher, x)
            loss_total, loss_ce, loss_kd = _step_kd(kd_loss_fn, s_logits, t_logits, y)
        else:
            loss_ce = ce_fn(s_logits, y).float()
            loss_kd = s_logits.new_zeros([])
            loss_total = loss_ce

        if use_at:
            student_feats = {f"layer{k}": feats[k] for k in student_layers}
            loss_at_scaled, at_layer_dict = at_crit(student_feats, at_bank, idx, split="val", p=at_p)
            loss_total = loss_total + loss_at_scaled
            at_unweighted = sum(at_layer_dict.values())
        else:
            at_unweighted = s_logits.new_zeros([])

        bs = x.size(0)
        seen += bs
        total_loss_sum += float(loss_total) * bs
        ce_sum += float(loss_ce) * bs
        kd_sum += float(loss_kd) * bs
        at_sum += float(at_unweighted) * bs
        correct += (s_logits.argmax(1) == y).sum().item()

        tbar.set_postfix({
            "loss": total_loss_sum / seen,
            "ce": ce_sum / seen,
            "kd": kd_sum / seen,
            "at": at_sum / seen,
            "acc": correct / max(1, seen),
        })

    for h in handles: h.remove()

    return {
        "val_loss": total_loss_sum / max(1, seen),
        "val_loss_ce": ce_sum / max(1, seen),
        "val_loss_kd": kd_sum / max(1, seen),
        "val_loss_at": at_sum / max(1, seen),
        "val_acc": correct / max(1, seen),
    }



# ----------------------------
# Main
# ----------------------------
def main():
    cfg = load_cfg_from_cli()
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    at_cfg = cfg.get("at", {})
    AT_ENABLED = bool(at_cfg.get("enabled", False))
    AT_APPLY_IN = set(at_cfg.get("apply_in", ["finetune"]))  # {"probe","finetune"} 중 택

    def at_active(stage: str) -> bool:
        """stage in {'probe','finetune'}"""
        return AT_ENABLED and (stage in AT_APPLY_IN)
    
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

    # --- AT bank 로드 (enabled일 때만): (N,D) 벡터 리더를 split별로 준비 ---
    if AT_ENABLED:
        at_train_path = at_cfg["bank"]["train_path"].format(dataset=ds_key)
        at_val_path   = at_cfg["bank"]["val_path"].format(dataset=ds_key)
        at_bank_train = ATBankReader(at_train_path, device=device)  # get("teacher","layerK", idx) -> (B,D)
        at_bank_val   = ATBankReader(at_val_path,   device=device)
        # 빠른 샘플 체크(옵션): 첫 배치 인덱스 몇 개로 shape 확인
        # idx_sanity = torch.arange(min(4, len(train_loader.dataset)), device=device)
        # _ = at_bank_train.get("teacher", "layer1", idx_sanity)
    else:
        at_bank_train = None
        at_bank_val = None

    # --- AT criterion (벡터 기반) ---
    at_layers = tuple(at_cfg.get("student_layers", [1,2,3,4]))
    at_layer_names = tuple(f"layer{k}" for k in at_layers)  # bank 키는 "layer1..4"
    at_crit = VectorizedATLoss(
        layer_names=at_layer_names,
        gamma_at=float(at_cfg.get("gamma_at", 10.0)),
        l2_normalize_student=True,
    )
    at_p = int(at_cfg.get("p", 2))

    # 캐시 강제 여부 플래그
    require_train_bank = bool(cfg["kd"].get("cache_logits", True))
    require_val_bank   = bool(cfg["kd"].get("cache_val_logits", False))

    # 경로에서 bank만 생성(파일이 없으면 예외)
    train_bank = None
    val_bank = None

    tr_num_samples = len(train_loader.dataset)
    va_num_samples = len(val_loader.dataset)
    num_classes=cfg["model"]["num_classes"]

    if require_train_bank:
        train_bank = TeacherLogitsBank(cfg["kd"]["cache_path"], tr_num_samples, num_classes)
        if not train_bank.exists():
            raise FileNotFoundError(f"Train logits cache not found: {cfg['kd']['cache_path']}")

    if require_val_bank:
        val_cache_path = cfg["kd"].get("cache_path_val",
                                    _derive_val_cache_path(cfg["kd"]["cache_path"]))
        val_bank = TeacherLogitsBank(val_cache_path, va_num_samples, num_classes)
        if not val_bank.exists():
            raise FileNotFoundError(f"Val logits cache not found: {val_cache_path}")

    kd_loss = build_kd_loss_fn(cfg)
    optim, sched = build_optimizer_scheduler(cfg, student)
    log_interval = int(cfg.get("logging", {}).get("log_interval", 50))
    gamma_kd = float(cfg.get("kd", {}).get("gamma_soft_kd", 0.0))
    grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0))

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
            tr = train_one_epoch(
                student, teacher, train_bank, train_loader, device, optim, kd_loss,
                log_interval, require_bank=require_train_bank,
                use_at=at_active("finetune"), at_bank=at_bank_train, at_crit=at_crit, at_p=at_p,
                student_layers=tuple(at_cfg.get("student_layers",[1,2,3,4])),
                gamma_kd=gamma_kd, grad_clip=grad_clip
            )
            va = eval_one_epoch(
                student, teacher, val_loader, device, kd_loss,
                bank=val_bank, require_bank=require_val_bank,
                use_at=at_active("finetune"), at_bank=at_bank_val, at_crit=at_crit, at_p=at_p,
                student_layers=tuple(at_cfg.get("student_layers",[1,2,3,4])),
                gamma_kd=gamma_kd
            )

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
                "train_loss_at": tr.get("train_loss_at", 0.0),
                "train_acc": tr["train_acc"],
                "val_loss": va["val_loss"],
                "val_loss_ce": va["val_loss_ce"],
                "val_loss_kd": va["val_loss_kd"],
                "val_loss_at": va.get("val_loss_at", 0.0),
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
            tr = train_one_epoch(
                student, teacher, train_bank, train_loader, device, optim, kd_loss,
                log_interval, require_bank=require_train_bank,
                use_at=at_active("finetune"), at_bank=at_bank, at_crit=at_crit, at_p=at_p,
                student_layers=tuple(at_cfg.get("student_layers",[1,2,3,4])),
                gamma_kd=gamma_kd, grad_clip=grad_clip
            )
            va = eval_one_epoch(
                student, teacher, val_loader, device, kd_loss,
                bank=val_bank, require_bank=require_val_bank,
                use_at=at_active("finetune"), at_bank=at_bank, at_crit=at_crit, at_p=at_p,
                student_layers=tuple(at_cfg.get("student_layers",[1,2,3,4])),
                gamma_kd=gamma_kd
            )

            if sched is not None: sched.step()
            time_sec = time.perf_counter() - t0
            lr = float(optim.param_groups[0]["lr"])

            line = {
                "epoch": global_epoch,
                "time_sec": time_sec,
                "train_loss": tr["train_loss"],
                "train_loss_ce": tr["train_loss_ce"],
                "train_loss_kd": tr["train_loss_kd"],
                "train_loss_at": tr.get("train_loss_at", 0.0),
                "train_acc": tr["train_acc"],
                "val_loss": va["val_loss"],
                "val_loss_ce": va["val_loss_ce"],
                "val_loss_kd": va["val_loss_kd"],
                "val_loss_at": va.get("val_loss_at", 0.0),
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
