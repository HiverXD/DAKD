# src/losses/attention_transfer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable
from src.data.attention_maps import activation_to_attention, vectorize_attention_map

class VectorizedATLoss(nn.Module):
    """
    교사 AT는 bank의 (B, D) 벡터, 학생은 on-the-fly로 [B,C,H,W] -> AT map -> (B,D) → MSE.
    layer_names: ("layer1","layer2","layer3","layer4") 등
    """
    def __init__(self, layer_names: Iterable[str], gamma_at: float = 10.0, l2_normalize_student: bool = True):
        super().__init__()
        self.layer_names = tuple(layer_names)
        self.gamma_at = float(gamma_at)
        self.l2_norm_stu = bool(l2_normalize_student)

    def forward(self, student_feats: Dict[str, torch.Tensor], at_bank_reader, idx: torch.Tensor, split: str, p: int = 2):
        """
        student_feats: {"layer1": [B,C,H,W], ...}
        at_bank_reader: AT_bank reader (get("teacher", layer_name, idx) -> (B, D) float32 on device)
        idx: [B] LongTensor
        split: "train" | "val"
        """
        loss = 0.0
        per_layer = {}
        for lname in self.layer_names:
            s_map = activation_to_attention(student_feats[lname].float(), p=p)          # [B,1,H,W]
            s_vec = vectorize_attention_map(s_map, l2_normalize=self.l2_norm_stu)       # [B,D]
            t_vec = at_bank_reader.get("teacher", lname, idx)                           # [B,D]
            if s_vec.shape != t_vec.shape:
                raise SystemExit(f"[AT vec mismatch] {lname}: student{tuple(s_vec.shape)} vs teacher{tuple(t_vec.shape)}  idx[:5]={idx[:5].tolist()}")
            msel = F.mse_loss(s_vec, t_vec, reduction="mean")
            per_layer[f"teacher_{lname}"] = msel.detach()
            loss = loss + msel
        return self.gamma_at * loss, per_layer
