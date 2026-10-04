"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species` (hoặc `Filename, Label`).
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_p = Path(labels_dir)
    train_csv = labels_p / f"train_subset{fold}.csv"
    val_csv = labels_p / f"val_subset{fold}.csv"
    test_csv = labels_p / f"test_subset{fold}.csv"

    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy đủ file split fold {fold} trong {labels_dir}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Kiểm tra:
      1. Số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. Giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. Hợp ba tập phải bằng đúng 17.509 ảnh
      4. Mọi Filename đều tồn tại trong `images_dir`
    """
    img_p = Path(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_imgs = n_train + n_val + n_test

    # 1. Tỉ lệ
    r_train = n_train / total_imgs * 100
    r_val = n_val / total_imgs * 100
    r_test = n_test / total_imgs * 100

    # 2. Giao rỗng & hợp đủ
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    ov_train_val = len(train_files & val_files)
    ov_train_test = len(train_files & test_files)
    ov_val_test = len(val_files & test_files)
    union_files = train_files | val_files | test_files
    n_union = len(union_files)

    assert ov_train_val == 0, f"Rò rỉ dữ liệu giữa Train và Val: {ov_train_val} ảnh trùng!"
    assert ov_train_test == 0, f"Rò rỉ dữ liệu giữa Train và Test: {ov_train_test} ảnh trùng!"
    assert ov_val_test == 0, f"Rò rỉ dữ liệu giữa Val và Test: {ov_val_test} ảnh trùng!"
    assert n_union == 17509, f"Hợp ba tập không bằng 17.509 ảnh: {n_union} ảnh!"

    # 3. Phân bố lớp
    train_cls = train_df["Label"].value_counts().to_dict()
    val_cls = val_df["Label"].value_counts().to_dict()
    test_cls = test_df["Label"].value_counts().to_dict()

    per_class = {
        c: {
            "name": CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class_{c}",
            "train": train_cls.get(c, 0),
            "val": val_cls.get(c, 0),
            "test": test_cls.get(c, 0),
            "total": train_cls.get(c, 0) + val_cls.get(c, 0) + test_cls.get(c, 0),
        }
        for c in range(NUM_CLASSES)
    }

    # 4. Kiểm tra tồn tại file ảnh
    all_files = list(train_df["Filename"]) + list(val_df["Filename"]) + list(test_df["Filename"])
    missing_files = [f for f in all_files if not (img_p / f).exists()]
    assert len(missing_files) == 0, f"Có {len(missing_files)} file ảnh trong CSV không tồn tại trong {images_dir}!"

    summary = {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": total_imgs},
        "ratio_pct": {"train": round(r_train, 2), "val": round(r_val, 2), "test": round(r_test, 2)},
        "overlap": {"train_val": ov_train_val, "train_test": ov_train_test, "val_test": ov_val_test},
        "union_count": n_union,
        "missing_files": len(missing_files),
        "per_class": per_class,
    }

    print("=" * 60)
    print("CHECK SPLIT SUMMARY (S1..S4)")
    print(f"Train : {n_train:5d} images ({r_train:.2f}%)")
    print(f"Val   : {n_val:5d} images ({r_val:.2f}%)")
    print(f"Test  : {n_test:5d} images ({r_test:.2f}%)")
    print(f"Total : {total_imgs:5d} images (Expected: 17509)")
    print(f"Overlap: Train&Val={ov_train_val}, Train&Test={ov_train_test}, Val&Test={ov_val_test} (Empty: PASS)")
    print(f"Missing image files: {len(missing_files)} (PASS)")
    print("=" * 60)

    return summary


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: CenterCrop(img_size) nếu khác 256, hoặc Resize((img_size, img_size)).
    """
    if train:
        if aug == "basic":
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        elif aug == "color":
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        elif aug == "trivial":
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        elif aug == "randaug":
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            raise ValueError(f"Augmentation {aug} không hợp lệ! Chọn trong [basic, color, trivial, randaug]")
    else:
        # Val / Test: không dùng augmentation ngẫu nhiên
        if img_size == 256:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (image_tensor, label:int, filename:str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = [int(lbl) for lbl in self.df["Label"].tolist()]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        fname = self.filenames[i]
        path = self.images_dir / fname
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], fname


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts().to_dict()
            class_weights_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights_dict[lbl] for lbl in df["Label"]]
            sampler_obj = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        else:
            sampler_obj = None
            shuffle = True
        drop_last = True if len(dataset) > batch_size else False
    else:
        sampler_obj = None
        shuffle = False
        drop_last = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=seed_worker if train else None,
    )
