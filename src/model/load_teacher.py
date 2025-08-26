# src/model/load_teacher.py
import torch
from torchvision import models as tv

# ---- 유틸: 키 정규화 ----
def _strip_prefix(k: str):
    for p in ("module.", "ema.", "backbone.", "encoder."):
        if k.startswith(p):
            return k[len(p):]
    return k

def _normalize(sd):
    return {_strip_prefix(k): v for k, v in sd.items()}

def make_resnet(name: str, num_classes: int):
    if name == "resnet18":
        m = tv.resnet18(weights=None)
    elif name == "resnet34":
        m = tv.resnet34(weights=None)
    else:
        raise ValueError(name)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

def _extract_state_dict(obj):
    # ckpt 포맷: dict 이고 "state_dict" 키가 있으면 그것을 사용
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj  # 이미 state_dict라고 가정

def load_backbone_and_classifier(backbone_path, classifier_path, name, num_classes, map_location="cpu"):
    m = make_resnet(name, num_classes)

    # 1) backbone 로드: backbone_only만, fc는 건드리지 않음
    bb_raw = torch.load(backbone_path, map_location=map_location, weights_only=True)
    bb_sd = _normalize(_extract_state_dict(bb_raw))
    bb_sd = {k: v for k, v in bb_sd.items() if not k.startswith("fc.")}
    m.load_state_dict(bb_sd, strict=False)

    # 2) classifier 로드: full에서 fc.*만 추출하여 덮어쓰기
    cls_raw = torch.load(classifier_path, map_location=map_location, weights_only=True)
    cls_sd_full = _normalize(_extract_state_dict(cls_raw))
    cls_fc = {k: v for k, v in cls_sd_full.items() if k.startswith("fc.")}

    # 클래스 수 확인 후 로드
    if "fc.weight" in cls_fc:
        out_dim = cls_fc["fc.weight"].shape[0]
        if out_dim != num_classes:
            # 라벨 수가 다르면 FC 재구성 후 로드 불가 → 새로 초기화
            in_feat = m.fc.in_features
            m.fc = torch.nn.Linear(in_feat, num_classes)
            # 여기서 별도 remap이 필요하면 추가 구현
        else:
            state = m.state_dict()
            state.update(cls_fc)
            m.load_state_dict(state, strict=True)
    else:
        # fc.*가 없다면 현재 FC 유지
        pass

    return m

def load_student_imagenet(resnet18_pt_path, num_classes, map_location="cpu"):
    m = make_resnet("resnet18", num_classes)
    sd_raw = torch.load(resnet18_pt_path, map_location=map_location, weights_only=True)
    sd = _normalize(_extract_state_dict(sd_raw))
    # 백본만 로드
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=False)
    # FC 새로 바인딩
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m
