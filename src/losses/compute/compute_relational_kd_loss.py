# src/losses/compute_relational_kd_loss.py
from typing import Dict
import torch
from src.losses.relational_kd import cos_gram_mse

@torch.no_grad()
def _ensure_2d(x):
    assert x.dim() == 2, f"RKD expects (B,D), got {tuple(x.shape)}"

def compute_rkd_loss(student_feats: torch.Tensor,
                     teacher_feats: torch.Tensor,
                     beta_scalar: float) -> Dict[str, torch.Tensor]:
    _ensure_2d(student_feats); _ensure_2d(teacher_feats)
    with torch.cuda.amp.autocast(enabled=student_feats.is_cuda and student_feats.dtype==torch.float16):
        loss_rkd = cos_gram_mse(student_feats, teacher_feats)
    loss_rkd_w = loss_rkd * float(beta_scalar)
    return {"loss_rkd": loss_rkd, "loss_rkd_weighted": loss_rkd_w}
