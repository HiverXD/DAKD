# src/losses/relational_kd.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def _row_normalize(x: torch.Tensor):
    return F.normalize(x, dim=1)

def cos_gram_mse(student_feats: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
    # (B,D) 입력 가정
    s = _row_normalize(student_feats)
    t = _row_normalize(teacher_feats)

    Gs = s @ s.T
    Gt = t @ t.T

    # 중심화 (행/열 평균 제거 + 전체평균 더하기)
    Gs = Gs - Gs.mean(dim=0, keepdim=True) - Gs.mean(dim=1, keepdim=True) + Gs.mean()
    Gt = Gt - Gt.mean(dim=0, keepdim=True) - Gt.mean(dim=1, keepdim=True) + Gt.mean()

    # off-diagonal만
    off_diag = ~torch.eye(Gs.size(0), dtype=torch.bool, device=Gs.device)
    return F.mse_loss(Gs[off_diag], Gt[off_diag], reduction='mean')
