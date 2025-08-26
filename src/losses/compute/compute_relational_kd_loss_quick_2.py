# src/losses/compute_relational_kd_loss_quick_2.py
from typing import Dict, Tuple
import torch
from src.losses.relational_kd_quick_2 import cos_gram_mse

def make_compute_relational_kd_loss(center: bool = True,
                                    offdiag_only: bool = True):
    """
    returns: compute_rkd(student_feats, teacher_feats) -> (loss_rkd, parts_dict)
    parts_dict: {"train_loss_rkd": float(loss_rkd)}
    """
    def compute_rkd(z_s: torch.Tensor, z_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_rkd = cos_gram_mse(z_s, z_t, center=center, offdiag_only=offdiag_only)
        return loss_rkd, {"train_loss_rkd": float(loss_rkd.detach().cpu().item())}
    return compute_rkd
