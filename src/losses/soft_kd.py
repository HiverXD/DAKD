# src/losses/soft_kd.py
import torch
import torch.nn.functional as F

class SoftTargetKDLoss(torch.nn.Module):
    """
    Hinton KD:
    L = (1-alpha)*CE(y, s) + alpha*T^2*KL( softmax(t/T) || log_softmax(s/T) )
    (alpha naming은 취향차. 여긴 KD 쪽 가중치를 alpha로 둠)
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        super().__init__()
        self.alpha = float(alpha)
        self.T = float(temperature)
        self.kldiv = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits_raw, targets):
        T = self.T
        ce = F.cross_entropy(student_logits, targets)
        log_p_s = F.log_softmax(student_logits / T, dim=1)
        p_t = F.softmax(teacher_logits_raw / T, dim=1).detach()
        kd = self.kldiv(log_p_s, p_t) * (T * T)
        total = (1.0 - self.alpha) * ce + self.alpha * kd
        return total, {"train_loss_ce": ce.detach(), "train_loss_kd": kd.detach(), "train_loss": total.detach()}
