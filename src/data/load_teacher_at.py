# src/data/load_teacher_at.py
# teacher at를 idx에 맞게 불러와 매칭
import torch

class TeacherATBank:
    """
    저장 포맷 가정:
    {
      "teacher_layer1": FloatTensor [N,1,H,W] or [N,H,W],
      "teacher_layer2": ...
      ...
    }
    """
    def __init__(self, path_train: str, path_val: str):
        self.train = torch.load(path_train, map_location='cpu')
        self.val   = torch.load(path_val,   map_location='cpu')

        # 3D 저장이면 4D로 보정
        for bank in (self.train, self.val):
            for k, v in list(bank.items()):
                if v.ndim == 3:
                    bank[k] = v.unsqueeze(1).contiguous()  # [N,1,H,W]

    def get(self, split: str, layer_key: str, idx: torch.Tensor, device):
        bank = self.train if split == 'train' else self.val
        maps = bank[layer_key][idx]                         # [B,1,H,W]
        return maps.to(device, non_blocking=True)
