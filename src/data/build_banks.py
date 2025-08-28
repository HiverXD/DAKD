import torch
from src.utils.config import load_config, expand_softkd_templates
from src.data.data_setup import loaders
from src.data.indexed import WithIndex, collate_with_index
from src.model.load_teacher import load_backbone_and_classifier
from src.data.soft_targets import TeacherLogitsBank
from src.train_soft_kd import _derive_val_cache_path

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config[0], args.config[1:])
    cfg = expand_softkd_templates(cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds_key = cfg["data"]["dataset"]

    # dataloaders (shuffle=False, WithIndex)
    all_loaders = loaders(cfg)  # 기존 함수
    tr = all_loaders[ds_key]["train"]
    va = all_loaders[ds_key]["val"]
    # WithIndex 래핑
    tr = torch.utils.data.DataLoader(WithIndex(tr.dataset),
                                     batch_size=tr.batch_size, shuffle=False,
                                     num_workers=tr.num_workers, pin_memory=True,
                                     collate_fn=collate_with_index)
    va = torch.utils.data.DataLoader(WithIndex(va.dataset),
                                     batch_size=va.batch_size, shuffle=False,
                                     num_workers=va.num_workers, pin_memory=True,
                                     collate_fn=collate_with_index)

    # teacher
    tch_cfg = cfg["model"]["teacher"]
    teacher = load_backbone_and_classifier(
        tch_cfg["backbone_path"], tch_cfg["classifier_path"],
        tch_cfg["arch"], cfg["model"]["num_classes"]
    ).to(device).eval()

    # paths
    train_path = cfg["kd"]["cache_path"]
    val_path = cfg["kd"].get("cache_path_val", _derive_val_cache_path(train_path))

    # build
    tb = TeacherLogitsBank(train_path)
    if not tb.exists():
        tb.build(teacher, tr, device=device)
    vb = TeacherLogitsBank(val_path)
    if not vb.exists():
        vb.build(teacher, va, device=device)

if __name__ == "__main__":
    main()
