# src/data/attention_maps.py
# student AT 계산을 위한 유틸 모듈
# p=2, 채널 제곱합 → [B,H,W] 만든 뒤 L2 정규화
import torch

def activation_to_attention(feat: torch.Tensor, p: int = 2, eps: float = 1e-12) -> torch.Tensor:
    """
    feat: [B,C,H,W] -> [B,1,H,W]  (채널 |·|^p 합 후 L2 정규화)
    """
    A = (feat.abs() ** p).sum(dim=1, keepdim=True)  # [B,1,H,W]
    flat = A.flatten(1)                              # [B, H*W]
    denom = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    A = A / denom.view(-1, 1, 1, 1)
    return A

# ★ 추가: [B,1,H,W] -> [B, H*W] (l2_normalize 여부 선택)
def vectorize_attention_map(att_1chw: torch.Tensor, l2_normalize: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """
    att_1chw: [B,1,H,W] -> vec [B, H*W]
    """
    v = att_1chw.flatten(1)  # [B, H*W]
    if l2_normalize:
        denom = v.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        v = v / denom
    return v
