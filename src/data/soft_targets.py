# src/data/soft_targets.py
from pathlib import Path
import os, random, torch
import torch.nn.functional as F
from math import log

class TeacherLogitsBank:
    """
    FP16 teacher logits cache on CPU, with silent diagnostics.
    - path: .pt file
    - meta: {"n": num_samples, "c": num_classes}
    - quiet=True  : 이상 없으면 아무 출력도 하지 않음
    - raise_on_error=True : 심각 이상 시 예외 발생
    - 진단 파라미터는 __init__ 인자로 조정 가능
    """
    def __init__(self, path, num_samples, num_classes, device="cuda",
                 quiet=True, raise_on_error=True,
                 diag_batches=3,        # 진단용으로 앞쪽 몇 배치만 훑음
                 skew_ratio=0.90,       # argmax 히스토그램이 한 클래스에 이 비율 이상 몰리면 이상
                 entropy_near_uniform=0.95,  # 평균 엔트로피가 ln(C)*이상 비율이면 평평 경고
                 tol_reload=2e-2,       # 저장 후 재로드 max|diff| 허용치 (float32 기준)
                 ):
        self.path, self.device = Path(path), device
        self.meta = {"n": num_samples, "c": num_classes}
        self.bank = None  # lazy load
        # 진단/로그 옵션
        self.quiet = quiet
        self.raise_on_error = raise_on_error
        self.diag_batches = int(diag_batches)
        self.skew_ratio = float(skew_ratio)
        self.entropy_near_uniform = float(entropy_near_uniform)
        self.tol_reload = float(tol_reload)
        self.warnings = []   # 빌드 과정 경고 메시지 누적

    def _log(self, msg):
        # 조용 모드라도 '이상 발견' 시에는 메시지를 쌓는다.
        self.warnings.append(msg)
        if not self.quiet:
            print(msg)

    def exists(self):
        return os.path.exists(self.path)

    @torch.no_grad()
    def build(self, teacher_model, loader, T=1.0):
        teacher_model.eval()
        n, c = self.meta["n"], self.meta["c"]

        bank_f32 = torch.empty(n, c, dtype=torch.float32)
        visited = torch.zeros(n, dtype=torch.bool)
        top1_hist = torch.zeros(c, dtype=torch.long)

        lnC = log(c)
        saw_labels = False

        for b, batch in enumerate(loader):
            # 배치 형식 해석
            if len(batch) == 3:
                x, y, idx = batch
                saw_labels = True
            elif len(batch) == 2:
                x, idx = batch
                y = None
                # 배치에 라벨이 없으면 배치 정확도 진단만 생략
            else:
                msg = "loader 배치 형태는 (x,y,idx) 또는 (x,idx) 여야 합니다."
                self._log(f"❌ {msg}")
                if self.raise_on_error: raise RuntimeError(msg)
                return

            x = x.to(self.device, non_blocking=True)
            idx = idx.to("cpu").long()

            # idx 범위 검사
            if idx.min() < 0 or idx.max() >= n:
                msg = f"❌ idx out of range: min={idx.min().item()}, max={idx.max().item()}, N={n}"
                self._log(msg)
                if self.raise_on_error: raise ValueError(msg)
                return

            # teacher live logits
            logits = teacher_model(x).detach().to("cpu", dtype=torch.float32)

            # 앞쪽 몇 배치는 조용히 진단 (이상 시에만 경고)
            if b < self.diag_batches:
                pT = F.softmax(logits / T, dim=1)
                ent = -(pT.clamp_min(1e-12).log() * pT).sum(1).mean().item()
                if ent >= self.entropy_near_uniform * lnC:
                    self._log(f"⚠️ 배치 {b}: teacher 분포가 너무 평평함 "
                              f"(avg_entropy={ent:.3f} nats, lnC={lnC:.3f}, T={T}). "
                              f"전처리/온도 확인 필요")
                if saw_labels and y is not None:
                    acc = (pT.argmax(1) == y.cpu()).float().mean().item()
                    # teacher가 안정적이라면 낮은 acc는 이상
                    if acc < 0.50:
                        self._log(f"⚠️ 배치 {b}: teacher_top1_acc={acc*100:.1f}% (비정상적으로 낮음). "
                                  f"전처리/라벨공간/모델 로딩 확인 필요")

            # bank 채우기
            bank_f32[idx] = logits
            visited[idx] = True

            # argmax 히스토그램 누적
            top1 = logits.argmax(1).to(torch.long)
            top1_hist.index_add_(0, top1, torch.ones_like(top1, dtype=torch.long))

        # 커버리지 검사
        miss = (~visited).nonzero(as_tuple=False).flatten()
        if len(miss) > 0:
            msg = f"❌ 미할당 인덱스 {len(miss)}개 존재. 예시: {miss[:10].tolist()}"
            self._log(msg)
            if self.raise_on_error: raise RuntimeError(msg)

        # 히스토그램 쏠림 검사
        max_cls = int(top1_hist.argmax().item())
        max_ratio = (top1_hist[max_cls].item() / max(1, int(top1_hist.sum().item())))
        if max_ratio >= self.skew_ratio:
            self._log(f"⚠️ argmax 히스토그램이 클래스 {max_cls}에 과도하게 쏠림 "
                      f"({max_ratio*100:.1f}%). idx/전처리/teacher 출력 확인 필요")

        # 저장(fp16)
        bank = bank_f32.to(torch.float16)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": bank, "meta": self.meta}, self.path)
        self.bank = bank

        # 재로드 일치성 검사
        probe_idx = torch.tensor(random.sample(range(n), min(64, n)), dtype=torch.long)
        live  = bank_f32[probe_idx]
        loaded = self._load_raw()["logits"][probe_idx].to(torch.float32)
        maxdiff = (live - loaded).abs().max().item()
        if maxdiff > self.tol_reload:
            self._log(f"⚠️ 저장/재로드 후 max|diff|={maxdiff:.4g} (tol={self.tol_reload}). "
                      f"파일 I/O/dtype 확인 필요")

        # 이상이 없었고 quiet=True면 아무 것도 출력하지 않음.

    def _load_raw(self):
        # 일반 로드(안전). 외부에서 신뢰 못 하는 파일이면 weights_only=True 권장.
        return torch.load(self.path, map_location="cpu")

    def _ensure_loaded(self):
        if self.bank is None:
            payload = self._load_raw()
            self.bank = payload["logits"]  # (N, C) fp16 on CPU

    def get(self, idx_tensor):
        self._ensure_loaded()
        out = self.bank[idx_tensor.cpu()].to(dtype=torch.float32, device=self.device)
        return out
