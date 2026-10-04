"""eda.py - Thực hiện EDA phân tích dữ liệu DeepWeeds:
1. Thống kê phân bố lớp (train/val/test/tổng), tính tỉ lệ max/min, đối chiếu Table 1 bài báo.
2. Vẽ biểu đồ cột phân bố lớp (curves/eda_class_distribution.png).
3. Lấy mẫu 3 ảnh cho mỗi lớp (tổng cộng 27 ảnh) và vẽ lưới trực quan (curves/eda_sample_images.png).
4. Phân tích kích thước, số kênh và các cặp lớp dễ gây nhầm lẫn.
"""
import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import dataset

# Số liệu công bố trong Table 1 của Olsen et al. (2019)
PAPER_TABLE_1 = {
    "Chinee Apple": 1125,
    "Lantana": 1064,
    "Parkinsonia": 1031,
    "Parthenium": 1022,
    "Prickly Acacia": 1062,
    "Rubber Vine": 1009,
    "Siam Weed": 1074,
    "Snake Weed": 1016,
    "Negatives": 9106,
}


def run_eda():
    print("=" * 70)
    print("BƯỚC 0: EDA (EXPLORATORY DATA ANALYSIS) TRÊN DATASET DEEPWEEDS")
    print("=" * 70)

    # 1. Đọc dữ liệu
    labels_df = pd.read_csv("data/labels/labels.csv")
    train_df = pd.read_csv("data/labels/train_subset0.csv")
    val_df = pd.read_csv("data/labels/val_subset0.csv")
    test_df = pd.read_csv("data/labels/test_subset0.csv")

    images_dir = Path("images")
    Path("curves").mkdir(parents=True, exist_ok=True)

    # 2. Đếm số lượng theo lớp
    total_counts = labels_df["Label"].value_counts().sort_index().to_dict()
    train_counts = train_df["Label"].value_counts().sort_index().to_dict()
    val_counts = val_df["Label"].value_counts().sort_index().to_dict()
    test_counts = test_df["Label"].value_counts().sort_index().to_dict()

    print("\n--- BẢNG ĐỐI CHIẾU SỐ LƯỢNG ẢNH VỚI TABLE 1 BÀI BÁO GỐC ---")
    print(f"{'Lớp':<16} | {'Thực tế':<8} | {'Bài báo':<8} | {'Train':<6} | {'Val':<6} | {'Test':<6} | {'Sai lệch'}")
    print("-" * 70)

    match_all = True
    for c in range(dataset.NUM_CLASSES):
        name = dataset.CLASS_NAMES[c]
        actual = total_counts.get(c, 0)
        paper = PAPER_TABLE_1.get(name, 0)
        tr = train_counts.get(c, 0)
        va = val_counts.get(c, 0)
        te = test_counts.get(c, 0)
        diff = actual - paper
        if diff != 0:
            match_all = False
        print(f"{name:<16} | {actual:<8} | {paper:<8} | {tr:<6} | {va:<6} | {te:<6} | {diff}")

    print("-" * 70)
    print(f"Tổng cộng        | {len(labels_df):<8} | {sum(PAPER_TABLE_1.values()):<8} | {len(train_df):<6} | {len(val_df):<6} | {len(test_df):<6}")
    assert match_all, "Số liệu thực tế không khớp với Table 1 của bài báo!"
    print("-> ĐỐI CHIẾU THÀNH CÔNG: Dữ liệu khớp 100% với Table 1 của bài báo gốc.")

    # Tỉ lệ mất cân bằng
    max_count = max(total_counts.values())
    min_count = min(total_counts.values())
    imbalance_ratio = max_count / min_count
    neg_pct = (labels_df["Label"] == 8).mean() * 100
    print(f"\nThống kê mất cân bằng:")
    print(f"- Lớp nhiều nhất: Negatives ({max_count} ảnh, chiếm {neg_pct:.2f}%)")
    print(f"- Lớp ít nhất   : Rubber Vine ({min_count} ảnh, chiếm {min_count/len(labels_df)*100:.2f}%)")
    print(f"- Tỉ lệ chênh lệch lớp max / min: {imbalance_ratio:.2f} lần!")

    # 3. Vẽ biểu đồ phân bố lớp
    plt.figure(figsize=(12, 6))
    x = np.arange(dataset.NUM_CLASSES)
    width = 0.25

    plt.bar(x - width, [train_counts.get(c, 0) for c in range(9)], width, label="Train (~60%)", color="#2b5c8f")
    plt.bar(x, [val_counts.get(c, 0) for c in range(9)], width, label="Val (~20%)", color="#e27c38")
    plt.bar(x + width, [test_counts.get(c, 0) for c in range(9)], width, label="Test (~20%)", color="#59a14f")

    plt.xticks(x, dataset.CLASS_NAMES, rotation=35, ha="right", fontsize=10)
    plt.ylabel("Số lượng ảnh", fontsize=11)
    plt.title("Phân bố số lượng ảnh theo 9 lớp trên các tập Train / Val / Test (Fold 0)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(dataset.NUM_CLASSES):
        tot = total_counts.get(i, 0)
        plt.text(i, tot * 0.65, f"Tổng:\n{tot}", ha="center", va="bottom", fontsize=8, color="#333333")

    plt.tight_layout()
    chart_path = "curves/eda_class_distribution.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"-> Đã lưu biểu đồ phân bố lớp tại: {chart_path}")

    # 4. Kiểm tra kích thước và kênh ảnh ngẫu nhiên
    sample_files = labels_df["Filename"].sample(100, random_state=42).tolist()
    sizes = set()
    modes = set()
    for f in sample_files:
        with Image.open(images_dir / f) as img:
            sizes.add(img.size)
            modes.add(img.mode)
    print(f"\nKiểm tra ảnh mẫu ngẫu nhiên (100 ảnh):")
    print(f"- Kích thước ảnh (Width x Height): {sizes} (Chuẩn 256x256)")
    print(f"- Hệ màu: {modes} (Chuẩn RGB 3 kênh)")

    # 5. Lấy mẫu 3 ảnh mỗi lớp và vẽ lưới 9x3 ảnh
    fig, axes = plt.subplots(dataset.NUM_CLASSES, 3, figsize=(9, 22))
    for c in range(dataset.NUM_CLASSES):
        c_files = labels_df[labels_df["Label"] == c]["Filename"].sample(3, random_state=42).tolist()
        for col_idx, fname in enumerate(c_files):
            img = Image.open(images_dir / fname).convert("RGB")
            ax = axes[c, col_idx]
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 1:
                ax.set_title(f"Class {c}: {dataset.CLASS_NAMES[c]}", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.text(-10, 128, f"{fname[:15]}...", ha="right", va="center", fontsize=7, rotation=90)

    plt.tight_layout()
    sample_grid_path = "curves/eda_sample_images.png"
    plt.savefig(sample_grid_path, dpi=130)
    plt.close()
    print(f"-> Đã lưu lưới ảnh mẫu 9 lớp tại: {sample_grid_path}")

    print("\n" + "=" * 70)
    print("HOÀN TẤT EDA! TẤT CẢ SỐ LIỆU ĐÃ ĐƯỢC XÁC THỰC VÀ LƯU ẢNH TRỰC QUAN.")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
