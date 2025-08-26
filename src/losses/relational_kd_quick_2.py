# src/losses/relational_kd_quick_2.py
import torch
import torch.nn.functional as F

__all__ = ["cos_gram_mse", "relational_kd_loss"]

@torch.no_grad()
def _offdiag_mask(b, device):
    return ~torch.eye(b, dtype=torch.bool, device=device)

def _cos_gram(z: torch.Tensor) -> torch.Tensor:
    # z: (B, D) -> Gram(B, B)
    z = F.normalize(z.float(), dim=1)  # AMP 대비 FP32 고정
    return z @ z.T

def _center(G: torch.Tensor) -> torch.Tensor:
    # double-centering (행/열 평균 제거 + 전체 평균 보정)
    return G - G.mean(dim=0, keepdim=True) - G.mean(dim=1, keepdim=True) + G.mean()

def cos_gram_mse(student_feats: torch.Tensor,
                 teacher_feats: torch.Tensor,
                 center: bool = True,
                 offdiag_only: bool = True) -> torch.Tensor:
    """
    RKD(우리 버전): 배치 임베딩 코사인 Gram 행렬 간 MSE
    - center: double-centering
    - offdiag_only: 대각(항상 1)을 제외하고 평균 MSE
    """
    Gs = _cos_gram(student_feats)
    Gt = _cos_gram(teacher_feats)

    if center:
        Gs = _center(Gs)
        Gt = _center(Gt)

    if offdiag_only:
        B = Gs.size(0)
        mask = _offdiag_mask(B, device=Gs.device)
        return F.mse_loss(Gs[mask], Gt[mask], reduction="mean")
    else:
        return F.mse_loss(Gs, Gt, reduction="mean")

# alias
relational_kd_loss = cos_gram_mse
