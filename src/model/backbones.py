# src/model/backbones.py
import os
import torch
import torch.nn as nn
import copy

# Safe import in case of torchvision API changes
try:
    from torchvision.models import resnet18, resnet34
    from torchvision.models import ResNet18_Weights, ResNet34_Weights
    _HAS_ENUM_WEIGHTS = True
except Exception:
    import torchvision
    _HAS_ENUM_WEIGHTS = False
    resnet18 = torchvision.models.resnet18
    resnet34 = torchvision.models.resnet34

ARCH2SAVE = {
    "resnet18": "ResNet18.pt",
    "resnet34": "ResNet34.pt",
}

def _load_imagenet_model(arch: str):
    """Load ImageNet pretrained model (version-compatible)"""
    if arch == "resnet18":
        if _HAS_ENUM_WEIGHTS:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model = resnet18(pretrained=True)
    elif arch == "resnet34":
        if _HAS_ENUM_WEIGHTS:
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            model = resnet34(pretrained=True)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model

def _to_backbone(model: nn.Module):
    """Convert to backbone by removing FC (replace with Identity) and return feature_dim"""
    feat_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feat_dim

def save_backbone(arch: str, save_dir: str = "src/model/ckpts") -> str:
    """Save pretrained backbone weights (.pt: {'arch','feature_dim','state_dict'})"""
    os.makedirs(save_dir, exist_ok=True)
    model = _load_imagenet_model(arch)
    model, feat_dim = _to_backbone(model)

    ckpt = {
        "arch": arch,
        "feature_dim": feat_dim,
        "state_dict": model.state_dict(),
    }
    save_path = os.path.join(save_dir, ARCH2SAVE[arch])
    torch.save(ckpt, save_path)
    return save_path

def prepare_backbones(save_dir: str = "src/model/ckpts"):
    """Save both ResNet18/34 backbones"""
    p18 = save_backbone("resnet18", save_dir)
    p34 = save_backbone("resnet34", save_dir)
    print(f"[OK] Saved backbones:\n - {p18}\n - {p34}")

def load_backbone(ckpt_path: str, device: str = "cpu") -> nn.Module:
    """Load saved backbone (FC = Identity state)"""
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt["arch"]
    model = _load_imagenet_model(arch)
    model, _ = _to_backbone(model)       # Remove FC
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    return model

def build_classifier(ckpt_path: str, num_classes: int, freeze_backbone: bool = False, device: str = "cpu") -> nn.Module:
    """Build model with backbone + new FC head"""
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt["arch"]
    feature_dim = ckpt["feature_dim"]

    model = _load_imagenet_model(arch)
    model, _ = _to_backbone(model)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    # Attach new classifier head
    model.fc = nn.Linear(feature_dim, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    model.to(device)
    return model

def export_backbone_from_classifier(model: nn.Module, arch: str, save_path: str):
    feat_dim = model.fc.in_features
    m = copy.deepcopy(model)
    m.fc = nn.Identity()
    ckpt = {"arch": arch, "feature_dim": feat_dim, "state_dict": m.state_dict()}
    torch.save(ckpt, save_path)

if __name__ == "__main__":
    # Save backbones when run as a standalone script
    prepare_backbones()
