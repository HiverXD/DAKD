# src/data/logit_cache.py
import os, torch

def build_or_load_logit_cache(teacher, loader, path, temperature, dtype=torch.float16):
    if os.path.isfile(path):
        return torch.load(path, map_location="cpu")
    teacher.eval()
    all_logits = []
    with torch.no_grad():
        for images, _, indices in loader:   # loader가 index를 반환
            logits = teacher(images.cuda()).to("cpu")
            all_logits.append(logits.to(dtype))
    cache = torch.cat(all_logits, dim=0)  # 데이터셋 순서대로 정렬됨
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(cache, path)
    return cache
