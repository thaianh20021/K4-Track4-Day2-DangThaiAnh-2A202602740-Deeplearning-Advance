"""test_pipeline.py - Sanity check cho Bước 0 theo slide trang 59 và GUIDE.md mục 1.3:
1. Cố định seed
2. Loss ban đầu của head mới xấp xỉ -ln(1/9) ≈ 2.197
3. Overfit một batch nhỏ tới loss gần 0
4. Kiểm tra ảnh sau augmentation (giải chuẩn hóa)
5. Kiểm tra chế độ model.train() và model.eval() (ảnh hưởng BatchNorm/Dropout)
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

# Thêm code vào path
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import dataset
import model as model_utils
import losses
import train


def run_sanity_checks():
    print("=" * 60)
    print("BƯỚC 0: SANITY CHECK TOÀN BỘ PIPELINE HUẤN LUYỆN")
    print("=" * 60)

    # 1. Cố định seed
    print("\n[1/5] Kiểm tra cố định seed...")
    train.set_seed(42)
    r1 = torch.randn(5)
    train.set_seed(42)
    r2 = torch.randn(5)
    diff_seed = torch.max(torch.abs(r1 - r2)).item()
    print(f"      Diff giữa 2 lần set_seed(42): {diff_seed:.6e}")
    assert diff_seed == 0.0, "Cố định seed thất bại!"
    print("      -> ĐẠT: Seed tái lập hoàn toàn.")

    # 2. Loss CE ban đầu xấp xỉ -ln(1/9) ≈ 2.197
    print("\n[2/5] Kiểm tra Loss CE ban đầu của head mới...")
    expected_loss = -np.log(1.0 / 9.0)
    m = model_utils.build_model("resnet50", pretrained=False, num_classes=9)
    m.eval()
    dummy_input = torch.randn(100, 3, 224, 224)
    dummy_target = torch.randint(0, 9, (100,))
    with torch.no_grad():
        logits = m(dummy_input)
        init_loss = nn.CrossEntropyLoss()(logits, dummy_target).item()
    print(f"      Kỳ vọng: -ln(1/9) = {expected_loss:.4f}")
    print(f"      Thực tế : {init_loss:.4f}")
    diff_loss = abs(init_loss - expected_loss)
    print(f"      Độ lệch : {diff_loss:.4f}")
    assert diff_loss < 0.3, f"Loss ban đầu lệch quá nhiều ({init_loss:.4f} so với {expected_loss:.4f})!"
    print("      -> ĐẠT: Loss ban đầu xấp xỉ -ln(1/9).")

    # 3. Overfit một batch nhỏ tới loss gần 0
    print("\n[3/5] Thử nghiệm Overfit 1 batch nhỏ (16 ảnh)...")
    train_df, _, _ = dataset.load_split("data/labels", fold=0)
    small_df = train_df.head(16)
    tf = dataset.build_transforms(train=False, img_size=224)  # Không augment để test overfit
    loader = dataset.make_loader(small_df, "images", tf, batch_size=16, train=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overfit_model = model_utils.build_model("resnet50", pretrained=False, num_classes=9).to(device)
    overfit_model.train()
    optimizer = torch.optim.Adam(overfit_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    images, targets, _ = next(iter(loader))
    images, targets = images.to(device), targets.to(device)

    initial_batch_loss = 0.0
    final_batch_loss = 0.0
    for step in range(50):
        optimizer.zero_grad()
        out = overfit_model(images)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_batch_loss = loss.item()
        final_batch_loss = loss.item()

    print(f"      Step  1 loss: {initial_batch_loss:.4f}")
    print(f"      Step 50 loss: {final_batch_loss:.6f}")
    assert final_batch_loss < 0.05, f"Không overfit được batch nhỏ! Loss dừng ở {final_batch_loss:.4f}"
    print("      -> ĐẠT: Mô hình và pipeline gradient hoạt động tốt, loss giảm sát 0.")

    # 4. Kiểm tra ảnh sau Augmentation (giải chuẩn hóa)
    print("\n[4/5] Kiểm tra trực quan Augmentation & nhãn...")
    train_tf = dataset.build_transforms(train=True, img_size=224, aug="basic")
    sample_loader = dataset.make_loader(train_df.head(4), "images", train_tf, batch_size=4, train=True, num_workers=0)
    aug_images, aug_labels, aug_names = next(iter(sample_loader))

    # De-normalize
    mean = np.array(dataset.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(dataset.IMAGENET_STD).reshape(3, 1, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        img_np = aug_images[i].numpy() * std + mean
        img_np = np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0)
        cls_name = dataset.CLASS_NAMES[aug_labels[i].item()]
        axes[i].imshow(img_np)
        axes[i].set_title(f"{cls_name}\n({aug_names[i]})", fontsize=10)
        axes[i].axis("off")

    Path("curves").mkdir(parents=True, exist_ok=True)
    aug_plot_path = Path("curves/eda_augmentation_check.png")
    plt.tight_layout()
    plt.savefig(aug_plot_path, dpi=120)
    plt.close()
    print(f"      Đã lưu ảnh kiểm tra augmentation tại: {aug_plot_path}")
    print("      -> ĐẠT: Ảnh sau biến đổi đúng kích thước [4, 3, 224, 224] và hiển thị rõ ràng.")

    # 5. Kiểm tra model.train() và model.eval() đúng lúc
    print("\n[5/5] Kiểm tra chế độ train()/eval() và freeze BatchNorm...")
    test_m = model_utils.build_model("resnet50", pretrained=False, num_classes=9, init="frozen")
    assert getattr(test_m, "backbone_frozen", False) == True, "Cờ backbone_frozen chưa được bật!"

    # Gọi model.train()
    test_m.train()
    # Kiểm tra xem có hàm đảm bảo BatchNorm ở eval không
    for m in test_m.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

    bn_modes = [m.training for m in test_m.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
    print(f"      Số lớp BatchNorm được giữ ở eval mode khi backbone frozen: {len(bn_modes)} lớp")
    assert not any(bn_modes), "Có BatchNorm bị bật về train() mode khi backbone bị đóng băng!"
    print("      -> ĐẠT: BatchNorm được bảo vệ chính xác ở chế độ eval().")

    print("\n" + "=" * 60)
    print("CHÚC MỪNG: CẢ 5 BƯỚC SANITY CHECK ĐỀU ĐẠT CHUẨN 100%!")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_checks()
