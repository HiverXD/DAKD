# src/data/data_setup.py
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os, shutil, torch, math
from typing import Dict, Literal, Tuple

AugName = Literal["none", "light", "autoaugment", "randaugment"]

def _seed_worker(worker_id):
    # worker별 난수 시드 고정 (재현성)
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np, random
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _prepare_tiny_imagenet_val(root):
    """
    tiny-imagenet-200/val/ 에 있는 이미지들을 val_annotations.txt 기준으로
    tiny-imagenet-200/val_split/<class>/filename.jpg 로 재구성한다.
    이미 만들어져 있으면 건너뜀.
    """
    base = os.path.join(root, "tiny-imagenet-200")
    val_dir = os.path.join(base, "val")
    ann = os.path.join(val_dir, "val_annotations.txt")
    images_dir = os.path.join(val_dir, "images")
    out_dir = os.path.join(base, "val_split")

    if not (os.path.exists(val_dir) and os.path.exists(ann) and os.path.exists(images_dir)):
        return None  # 사용자가 데이터를 아직 내려받아 풀지 않았음

    if os.path.exists(out_dir):
        return out_dir  # 이미 준비됨

    os.makedirs(out_dir, exist_ok=True)
    # filename\tclass\t...
    with open(ann, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: 
                continue
            fname, cls = parts[0], parts[1]
            src = os.path.join(images_dir, fname)
            dst_dir = os.path.join(out_dir, cls)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)
            if not os.path.exists(dst):
                # Windows에서도 동작하는 안전한 방식으로 copy
                shutil.copy2(src, dst)
    return out_dir

def loaders(batch_size=128, root="dataset", num_workers=4, pin_memory=True, download=False):
    # --- stats ---
    # CIFAR-10/100 (기존)
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2470, 0.2435, 0.2616)
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std  = (0.2675, 0.2565, 0.2761)
    # STL10
    stl10_mean = (0.4467, 0.4398, 0.4066)
    stl10_std  = (0.2241, 0.2215, 0.2239)
    # Tiny-ImageNet-200 (관용적으로 쓰는 통계)
    tiny_mean = (0.4802, 0.4481, 0.3975)
    tiny_std  = (0.2302, 0.2265, 0.2262)

    # --- transforms: 리사이즈 없이 정규화만 ---
    tf_cifar10  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar10_mean,  cifar10_std)])
    tf_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar100_mean, cifar100_std)])
    tf_stl10    = transforms.Compose([transforms.ToTensor(), transforms.Normalize(stl10_mean,   stl10_std)])
    tf_tiny     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(tiny_mean,    tiny_std)])

    # --- datasets ---
    # CIFAR
    ds_cifar10_train  = datasets.CIFAR10(root=root, train=True,  download=download, transform=tf_cifar10)
    ds_cifar10_test   = datasets.CIFAR10(root=root, train=False, download=download, transform=tf_cifar10)
    ds_cifar100_train = datasets.CIFAR100(root=root, train=True,  download=download, transform=tf_cifar100)
    ds_cifar100_test  = datasets.CIFAR100(root=root, train=False, download=download, transform=tf_cifar100)

    # STL10 (split="train"/"test", "unlabeled"는 사용하지 않음)
    ds_stl10_train = datasets.STL10(root=root, split="train", download=download, transform=tf_stl10)
    ds_stl10_test  = datasets.STL10(root=root, split="test",  download=download, transform=tf_stl10)

    # Tiny-ImageNet-200 (torchvision에 전용 Dataset 없음 → ImageFolder 사용)
    # 기대 경로: <root>/tiny-imagenet-200/{train, val}
    from torchvision.datasets import ImageFolder
    tiny_base = os.path.join(root, "tiny-imagenet-200")
    tiny_train_dir = os.path.join(tiny_base, "train")
    # val은 폴더 구조를 한 번 정리해야 ImageFolder로 읽을 수 있음
    tiny_val_dir = _prepare_tiny_imagenet_val(root) or os.path.join(tiny_base, "val")  # 준비 실패 시 원본(val) 사용 시도

    ds_tiny_train = ImageFolder(root=tiny_train_dir, transform=tf_tiny)
    ds_tiny_test  = ImageFolder(root=tiny_val_dir,   transform=tf_tiny)

    train = {
        "cifar10":      ds_cifar10_train,
        "cifar100":     ds_cifar100_train,
        "stl10":        ds_stl10_train,
        "tiny_imagenet":ds_tiny_train,
    }
    test = {
        "cifar10":      ds_cifar10_test,
        "cifar100":     ds_cifar100_test,
        "stl10":        ds_stl10_test,
        "tiny_imagenet":ds_tiny_test,
    }

    train_loaders = {k: DataLoader(v, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=pin_memory,
                                   persistent_workers=(num_workers > 0))
                     for k, v in train.items()}
    test_loaders  = {k: DataLoader(v, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=pin_memory,
                                   persistent_workers=(num_workers > 0))
                     for k, v in test.items()}
    return train_loaders, test_loaders

if __name__ == "__main__":
    loaders()