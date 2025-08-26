# src/utils/trainer.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Literal, Dict, Any
import torch, time, json
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json, os, torch
from collections import defaultdict

ScheduleMode = Literal["step", "epoch", "plateau"]

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[object] = None,
                 schedule_mode: ScheduleMode = "epoch",
                 device: str = "cuda",
                 train_loader=None, valid_loader=None, test_loader=None,
                 epochs: int = 1,
                 output_dir: Path | str = "runs/tmp",
                 amp: bool = True,
                 compute_loss: Optional[Callable[[nn.Module, tuple, str, int, int, str], tuple]] = None,
                 log_interval: int = 50,
                 early_stopping: Optional[Dict[str, Any]] = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_mode = schedule_mode
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.epochs = int(epochs)
        self.out_dir = Path(output_dir); (self.out_dir/"checkpoints").mkdir(parents=True, exist_ok=True)

        self.amp = amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=device, enabled=self.amp)
        self.compute_loss = compute_loss
        self.log_interval = log_interval
        self.early_stopping = early_stopping or {"enabled": False, "patience": 10, "monitor": "val_acc"}

        self.best_state_dict = None
        self.best_metric = float("-inf")
        self.history = []  # per-epoch dicts

        self.model.to(self.device)

    def _default_compute_loss(self, model, batch, mode, step, epoch, device):
        x, y = batch
        logits = model(x.to(device))
        loss = F.cross_entropy(logits, y.to(device))
        return loss, logits

    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.model.eval()
        total, correct, total_loss = 0, 0, 0.0
        for batch in loader:
            loss, logits = self._default_compute_loss(self.model, batch, "eval", 0, 0, self.device) \
                           if self.compute_loss is None else self.compute_loss(self.model, batch, "eval", 0, 0, self.device)
            y = batch[1].to(self.device)
            total_loss += float(loss.detach()) * y.size(0)
            pred = logits.argmax(1)
            correct += int((pred == y).sum())
            total += y.size(0)
        return {"loss": total_loss/total, "acc": correct/total}

    def fit(self):
        patience = int(self.early_stopping.get("patience", 10))
        monitor_key = self.early_stopping.get("monitor", "val_acc")  # 'val_acc' or 'val_loss'
        mode = self.early_stopping.get("mode", "max")                # 'max' or 'min'
        best = -float("inf") if mode == "max" else float("inf")
        cmp = (lambda a, b: a > b) if mode == "max" else (lambda a, b: a < b)
        wait = 0

        # 스케줄러 타입-모드 일치 체크
        if self.schedule_mode == "plateau":
            assert isinstance(self.scheduler, ReduceLROnPlateau), \
                "schedule_mode='plateau'이면 ReduceLROnPlateau를 사용하고 metric을 step에 전달해야 합니다."

        epoch_bar = tqdm(range(1, self.epochs + 1), desc="Epochs", unit="epoch")
        for epoch in epoch_bar:
            self.model.train()
            t0 = time.time()
            total, correct, total_loss = 0, 0, 0.0

            for step, batch in enumerate(self.train_loader, 1):
                self.optimizer.zero_grad(set_to_none=True)

                # forward + loss
                if self.amp:
                    with torch.cuda.amp.autocast():
                        loss, logits = (self._default_compute_loss(self.model, batch, "train", step, epoch, self.device)
                                        if self.compute_loss is None else
                                        self.compute_loss(self.model, batch, "train", step, epoch, self.device))
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, logits = (self._default_compute_loss(self.model, batch, "train", step, epoch, self.device)
                                    if self.compute_loss is None else
                                    self.compute_loss(self.model, batch, "train", step, epoch, self.device))
                    loss.backward()
                    self.optimizer.step()

                # metrics (미니배치 누적)
                y = batch[1].to(self.device)
                bs = y.size(0)
                total_loss += float(loss.detach()) * bs
                correct += int((logits.argmax(1) == y).sum())
                total += bs

                # per-step scheduler
                if self.scheduler is not None and self.schedule_mode == "step":
                    self.scheduler.step()

            # 평가
            val_loader = self.valid_loader or self.test_loader
            val_stats = self._eval_epoch(val_loader) if val_loader is not None else {"loss": float("nan"), "acc": float("nan")}
            train_loss = total_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            # 스케줄러 per-epoch/plateau
            if self.scheduler is not None:
                if self.schedule_mode == "epoch":
                    self.scheduler.step()
                elif self.schedule_mode == "plateau":
                    # 보통 val_loss 모니터링
                    monitor_metric = val_stats["loss"] if monitor_key == "val_loss" else val_stats["acc"]
                    self.scheduler.step(monitor_metric)

            # 기록
            epoch_stats = {
                "epoch": epoch, "time_sec": time.time() - t0,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_stats["loss"], "val_acc": val_stats["acc"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(epoch_stats)

            # tqdm 상단(에폭 바) 요약 출력
            epoch_bar.set_postfix(
                val_loss=f"{val_stats['loss']:.4f}",
                val_acc=f"{val_stats['acc']:.4f}",
                lr=f"{epoch_stats['lr']:.3e}"
            )

            # 저장: last
            torch.save(
                {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "opt": self.optimizer.state_dict(),
                    "hist": self.history,
                },
                self.out_dir / "checkpoints" / "last.pt",
            )

            # 저장: best (monitor 기준)
            current = val_stats["acc"] if monitor_key == "val_acc" else val_stats["loss"]
            if cmp(current, best):
                best = current
                # 큰 모델은 RAM에 두지 말고 파일만 저장하는 것도 고려
                best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
                torch.save(
                    {"epoch": epoch, "model": best_state_dict, "metric": best, "monitor": monitor_key, "mode": mode},
                    self.out_dir / "checkpoints" / "best.pt",
                )
                wait = 0
            else:
                wait += 1

            # early stopping
            if self.early_stopping.get("enabled", False) and wait >= patience:
                tqdm.write(f"[EarlyStop] no improvement for {patience} epochs (best {monitor_key}={best:.4f})")
                break
        # end for epoch
        epoch_bar.close()

        # 최종 테스트 (옵션)
        if self.test_loader is not None and (self.out_dir / "checkpoints" / "best.pt").exists():
            ckpt = torch.load(self.out_dir / "checkpoints" / "best.pt", map_location="cpu")
            self.model.load_state_dict(ckpt["model"])
            test_stats = self._eval_epoch(self.test_loader)
            with open(self.out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"monitor": monitor_key, "val_best": ckpt["metric"], "test_acc": test_stats["acc"], "test_loss": test_stats["loss"]},
                    f, indent=2
                )

        # history 저장
        with open(self.out_dir / "metrics_per_epoch.jsonl", "w", encoding="utf-8") as f:
            for row in self.history:
                f.write(json.dumps(row) + "\n")

    def _epoch_loop(self, loader, mode, compute_loss, device, epoch, jsonl_path):
        self.model.train(mode == "train")
        meter, seen = defaultdict(float), 0

        for step, batch in enumerate(loader):
            loss, metrics = compute_loss(self.model, batch, mode, step, epoch, device)

            if mode == "train":
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                (self.scaler.step(self.optimizer) if hasattr(self, "scaler") else self.optimizer.step())
                if self.scheduler is not None: self.scheduler.step()

            bs = batch[0].size(0) if isinstance(batch, (tuple, list)) else 1
            seen += bs
            # total도 포함해서 모든 metric 누적
            for k, v in metrics.items():
                v = v.item() if hasattr(v, "item") else float(v)
                meter[f"{mode}_{k}"] += v * bs

        # 에폭 평균
        epoch_metrics = {"epoch": int(epoch), "mode": mode}
        for k, v in list(meter.items()):
            epoch_metrics[k] = v / max(1, seen)

        # jsonl append (001 포맷 유지)
        if jsonl_path:
            os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
        return epoch_metrics
