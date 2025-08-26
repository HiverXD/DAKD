# src/data/final_embeddings.py
from pathlib import Path
import os, random, torch
import torch.nn.functional as F

class TeacherEmbeddingBank:
    """
    FP16 teacher *final embeddings* cache on CPU, with silent diagnostics.
    - path: .pt file
    - meta: {"n": num_samples, "d": feat_dim}
    - quiet=True  : 이상 없으면 출력 없음 (warnings 리스트에는 누적)
    - raise_on_error=True : 심각 이상 시 예외 발생
    """
    def __init__(self, path, num_samples, feat_dim, device="cuda",
                 quiet=True, raise_on_error=True,
                 diag_batches=3,
                 tol_reload=2e-2,   # 저장/재로드 후 max|diff| 허용치 (float32 기준)
                 ):
        self.path, self.device = Path(path), device
        self.meta = {"n": int(num_samples), "d": int(feat_dim)}
        self.bank = None  # lazy
        self.quiet = quiet
        self.raise_on_error = raise_on_error
        self.diag_batches = int(diag_batches)
        self.tol_reload = float(tol_reload)
        self.warnings = []

    def _log(self, msg):
        self.warnings.append(msg)
        if not self.quiet:
            print(msg)

    def exists(self):
        return os.path.exists(self.path)

    @torch.no_grad()
    def build(self, teacher_model, loader, feature_extractor):
        """
        feature_extractor(teacher_model, x) -> (B, D) float32/float16 allowed
        """
        teacher_model.eval()
        n, d = self.meta["n"], self.meta["d"]

        bank_f32 = torch.empty(n, d, dtype=torch.float32)
        visited = torch.zeros(n, dtype=torch.bool)

        for b, batch in enumerate(loader):
            # 배치 해석: (x,y,idx) 또는 (x,idx) 또는 (idx,x,y)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                a, b_, c = batch
                if torch.is_tensor(a) and a.dim() == 4:
                    x, _, idx = a, b_, c
                elif torch.is_tensor(b_) and b_.dim() == 4:
                    x, _, idx = b_, c, a
                else:
                    x, idx = a, b_   # (x, idx)로 온 경우
                y = None
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                a, b_ = batch
                if torch.is_tensor(a) and a.dim() == 4:
                    x, idx = a, b_
                else:
                    x, idx = b_, a
                y = None
            else:
                msg = "loader 배치는 (x,y,idx)/(x,idx)/(idx,x,y) 중 하나여야 합니다."
                self._log(f"❌ {msg}")
                if self.raise_on_error: raise RuntimeError(msg)
                return

            if not (torch.is_tensor(x) and x.dim() == 4):
                msg = f"이미지 텐서는 4D(B,C,H,W) 여야 함. got={getattr(x,'shape',None)}"
                self._log(f"❌ {msg}")
                if self.raise_on_error: raise RuntimeError(msg)
                return

            x = x.to(self.device, non_blocking=True)
            idx = idx.to("cpu").long()

            if idx.min() < 0 or idx.max() >= n:
                msg = f"❌ idx out of range: min={idx.min().item()}, max={idx.max().item()}, N={n}"
                self._log(msg)
                if self.raise_on_error: raise ValueError(msg)
                return

            # features
            feats = feature_extractor(teacher_model, x)  # (B, D)
            feats = feats.detach().to("cpu", dtype=torch.float32)

            if feats.dim() != 2 or feats.size(1) != d:
                msg = f"❌ feature dim mismatch. expect (B,{d}), got {tuple(feats.shape)}"
                self._log(msg)
                if self.raise_on_error: raise RuntimeError(msg)
                return

            bank_f32[idx] = feats
            visited[idx] = True

        # 커버리지 검사
        miss = (~visited).nonzero(as_tuple=False).flatten()
        if len(miss) > 0:
            msg = f"❌ 미할당 인덱스 {len(miss)}개 존재. 예시: {miss[:10].tolist()}"
            self._log(msg)
            if self.raise_on_error: raise RuntimeError(msg)

        # 저장(fp16)
        bank = bank_f32.to(torch.float16)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": bank, "meta": self.meta}, self.path)
        self.bank = bank

        # 재로드 일치성 검사 (샘플링)
        import random as _r
        probe_idx = torch.tensor(_r.sample(range(n), min(64, n)), dtype=torch.long)
        live  = bank_f32[probe_idx]
        loaded = self._load_raw()["embeddings"][probe_idx].to(torch.float32)
        maxdiff = (live - loaded).abs().max().item()
        if maxdiff > self.tol_reload:
            self._log(f"⚠️ 저장/재로드 후 max|diff|={maxdiff:.4g} (tol={self.tol_reload}).")

    def _load_raw(self):
        return torch.load(self.path, map_location="cpu")

    def _ensure_loaded(self):
        if self.bank is None:
            payload = self._load_raw()
            self.bank = payload["embeddings"]  # (N, D) fp16 on CPU

    def get(self, idx_tensor):
        self._ensure_loaded()
        return self.bank[idx_tensor.cpu()].to(dtype=torch.float32, device=self.device)
