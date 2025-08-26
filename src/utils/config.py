# src/utils/config.py
from __future__ import annotations
from pathlib import Path
import yaml
import copy
import datetime as dt
import os, json
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import subprocess

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def pretty(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)

def expand_softkd_templates(cfg: dict) -> dict:
    """
    dataset만으로 teacher ckpt / cache 경로 자동 확장 (model.* 트리 일관 유지)
    """
    if not cfg.get("kd", {}).get("enable", False):
        return cfg

    ds = cfg["data"]["dataset"]
    ckpt_root = cfg.get("paths", {}).get("ckpt_root", "src/model/ckpts")

    # 1) teacher: teachers.json > template_* > fallback
    mt = cfg.setdefault("model", {}).setdefault("teacher", {})
    teachers_json = os.path.join(ckpt_root, "teachers.json")
    t_bb, t_cl = None, None
    if os.path.exists(teachers_json):
        j = json.load(open(teachers_json, "r", encoding="utf-8"))
        if ds in j:
            t_bb = j[ds].get("backbone_path")
            t_cl = j[ds].get("classifier_path")

    t_bb = t_bb or mt.get("template_backbone", "src/model/ckpts/resnet34_{dataset}_backbone.pt").format(dataset=ds)
    t_cl = t_cl or mt.get("template_classifier", "src/model/ckpts/resnet34_{dataset}_classifier.pt").format(dataset=ds)

    mt.setdefault("arch", "resnet34")
    mt.setdefault("backbone_path", t_bb)
    mt.setdefault("classifier_path", t_cl)

    # 2) student ckpt
    ms = cfg.setdefault("model", {}).setdefault("student", {})
    ms.setdefault("arch", "resnet18")
    ms.setdefault("ckpt", os.path.join(ckpt_root, "ResNet18.pt"))

    # 3) logits cache
    kd = cfg.setdefault("kd", {})
    if kd.get("cache_logits", True):
        cache_tmpl = cfg.get("kd", {}).get("cache_template",
                     cfg.get("paths", {}).get("template_cache",
                         os.path.join(ckpt_root, "soft_targets/{dataset}_fp16.pt")))
        kd.setdefault("cache_path", cache_tmpl.format(dataset=ds))

    return cfg

def load_config(root_cfg_path: str | Path, override_paths: list[str | Path] | None = None) -> dict:
    root_cfg_path = Path(root_cfg_path)
    with root_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for p in override_paths or []:
        p = Path(p)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                _deep_update(cfg, yaml.safe_load(f))
        else:
            raise FileNotFoundError(f"Override file not found: {p}")

    # runtime 보조 필드 (참고: outdir 실제 사용은 train에서 조합)
    now = dt.datetime.now()
    namefmt = cfg["output"].get("namefmt", "%Y%m%d_%H%M%S")
    runstamp = now.strftime(namefmt)
    cfg["_runtime"] = {"runstamp": runstamp}
    return cfg

def load_config_from_cli() -> dict:
    """
    --config <path> 를 여러 번 받을 수 있게 해주는 간단 CLI 로더.
    첫 번째가 root, 이후가 override로 해석됩니다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True,
                        help="config.yaml (root) and any number of overrides")
    args = parser.parse_args()
    cfg_paths = args.config
    root, overrides = cfg_paths[0], cfg_paths[1:]
    cfg = load_config(root, overrides)
    return cfg

def save_config_used(cfg, out, filename="config_used.yaml", add_meta=True):
    """
    Write the resolved config as YAML to `out/filename` (if `out` is a directory)
    or directly to `out` (if it's a .yml/.yaml file path).

    - Creates parent dirs if needed
    - Preserves key order and unicode
    - Optionally adds minimal metadata (_meta) with timestamp and git commit
    """
    out = Path(out)
    if out.suffix.lower() in {".yml", ".yaml"}:
        file_path = out
        file_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        file_path = out / filename

    payload = deepcopy(cfg)
    if add_meta:
        meta = {
            "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
        try:
            # best effort: add short git commit if available
            githash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if githash:
                meta["git_commit"] = githash
        except Exception:
            pass
        payload = {"_meta": meta, **payload}

    with file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    return str(file_path)
