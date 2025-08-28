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
    ds_keys = cfg["dataset_lst"]["ds_keys"]
    ds_num_classes = cfg["dataset_lst"]["ds_num_classes"]

    # dataloaders (shuffle=False, WithIndex)
    train_loaders, valid_loaders = loaders()  # 기존 함수
    for i, ds_key in enumerate(ds_keys):
        tr = train_loaders[ds_key]
        va = valid_loaders[ds_key]
        # WithIndex 래핑
        tr = torch.utils.data.DataLoader(WithIndex(tr.dataset),
                                        batch_size=tr.batch_size, shuffle=False,
                                        num_workers=tr.num_workers, pin_memory=True,
                                        collate_fn=collate_with_index)
        va = torch.utils.data.DataLoader(WithIndex(va.dataset),
                                        batch_size=va.batch_size, shuffle=False,
                                        num_workers=va.num_workers, pin_memory=True,
                                        collate_fn=collate_with_index)
        tr_num_samples = len(tr.dataset)
        va_num_samples = len(va.dataset)
        num_classes=ds_num_classes[i]

        # teacher
        tch_cfg = cfg["model"]["teacher"]

        bpt = tch_cfg["backbone_path"]
        cpt = tch_cfg["classifier_path"]
        bp = bpt.format(dataset=ds_key)
        cp = cpt.format(dataset=ds_key)

        cfg["model"]["num_classes"]
        teacher = load_backbone_and_classifier(
            bp, cp,
            tch_cfg["arch"], num_classes
        ).to(device).eval()

        # paths
        template = cfg["kd"]["cache_path"]
        train_path = template.format(dataset=ds_key)
        val_path = _derive_val_cache_path(train_path)

        # build
        tb = TeacherLogitsBank(train_path, num_samples=tr_num_samples, num_classes=num_classes)
        if not tb.exists():
            tb.build(teacher, tr)
        vb = TeacherLogitsBank(val_path, num_samples=va_num_samples, num_classes=num_classes)
        if not vb.exists():
            vb.build(teacher, va)

if __name__ == "__main__":
    main()
