# src/utils/schedulers.py
from dataclasses import dataclass

@dataclass
class BetaScheduleCfg:
    start_epoch: int = 21
    end_epoch: int = 30
    base: float = 50.0
    # step 단위 보간을 원하면 True
    stepwise: bool = True

class BetaScheduler:
    def __init__(self, cfg: BetaScheduleCfg, steps_per_epoch: int):
        self.cfg = cfg
        self.spe = max(1, steps_per_epoch)

    def value(self, epoch: int, step_in_epoch: int = 0):
        s, e, base = self.cfg.start_epoch, self.cfg.end_epoch, self.cfg.base
        if epoch < s:
            return 0.0
        if epoch > e:
            return base
        # warm-up 구간
        if self.cfg.stepwise:
            # epoch을 [s,e]로 정규화 + step 보간
            progress_epochs = (epoch - s) + (step_in_epoch / self.spe)
            total = max(1e-8, (e - s + 1))
            w = min(1.0, max(0.0, progress_epochs / total))
        else:
            # epoch 단위 선형
            w = (epoch - s + 1) / max(1, (e - s + 1))
            w = min(1.0, max(0.0, w))
        return base * w
