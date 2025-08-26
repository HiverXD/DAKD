# src/losses/compute_soft_kd_loss.py
import torch
import torch.nn.functional as F
from src.losses.soft_kd import SoftTargetKDLoss

def make_compute_loss_kd(student, kd_loss_fn: SoftTargetKDLoss, teacher=None, bank=None):
    """
    Returns:
      compute_loss(model, batch, mode, step, epoch, device) -> (loss_total, metrics_dict)
    metrics_dict keys (Trainer가 epoch 평균을 jsonl로 기록):
      - loss_total, loss_ce, loss_kd, acc
    """
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    def compute_loss(model, batch, mode, step, epoch, device):
        # batch: (x, y, idx) or (x, y)
        if len(batch) == 3:
            x, y, idx = batch
        else:
            x, y, idx = batch[0], batch[1], None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        s_logits = student(x)
        if bank is not None and idx is not None:
            t_logits = bank.get(idx)
        else:
            assert teacher is not None, "teacher or bank must be provided."
            with torch.no_grad():
                t_logits = teacher(x)

        total, parts = kd_loss_fn(s_logits, t_logits, y)  # total + {"train_loss_ce","train_loss_kd","train_loss"}
        with torch.no_grad():
            acc = (s_logits.argmax(1) == y).float().mean()

        # Trainer가 접두사(train_/val_)를 붙여 집계할 수 있게 일반 키로 반환
        return total, {
            "loss_total": parts["train_loss"],
            "loss_ce":    parts["train_loss_ce"],
            "loss_kd":    parts["train_loss_kd"],
            "acc":        acc,
        }

    return compute_loss
