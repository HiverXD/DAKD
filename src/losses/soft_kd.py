import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetKDLoss(nn.Module):
    """
    total = CE + gamma * (T^2 * KL)
    - alpha만 주어지면 하위호환: gamma = alpha / (1 - alpha)
    - 로그용 키: train_loss, train_loss_ce, train_loss_kd
    """
    def __init__(self, temperature: float = 1.0,
                 gamma: float | None = None,
                 alpha: float | None = None,    # backward-compat
                 reduction: str = "mean"):
        super().__init__()
        self.temperature = float(temperature)
        if gamma is None and alpha is not None:
            eps = 1e-8
            gamma = float(alpha) / max(1.0 - float(alpha), eps)
        self.gamma = float(0.0 if gamma is None else gamma)
        self.reduction = reduction

    def forward(self, s_logits: torch.Tensor,
                t_logits: torch.Tensor,
                target: torch.Tensor):
        T = self.temperature

        ce = F.cross_entropy(s_logits, target, reduction=self.reduction)
        kd = F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1),
            reduction="batchmean"
        ) * (T * T)

        total = ce + self.gamma * kd
        parts = {
            "train_loss": total.detach(),
            "train_loss_ce": ce.detach(),
            "train_loss_kd": kd.detach(),
        }
        return total, parts
