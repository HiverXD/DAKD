# src/data/indexed.py
import torch
from torch.utils.data import Dataset

class WithIndex(Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i

def collate_with_index(batch):
    xs, ys, idxs = zip(*batch)  # each element: (x, y, idx)
    xs = torch.stack(xs, dim=0)
    ys = torch.as_tensor(ys, dtype=torch.long)
    idxs = torch.as_tensor(idxs, dtype=torch.long)
    return xs, ys, idxs
