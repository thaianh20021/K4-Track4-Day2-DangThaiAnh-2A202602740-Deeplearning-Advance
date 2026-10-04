"""run_step2.py - Tự động chạy toàn bộ Bước 2: Thử nghiệm công thức huấn luyện (Ablation Study)

Khảo sát trên Backbone tốt nhất (ConvNeXt-Tiny):
- T00: Cấu hình nền (Pretrained finetune, CE, Basic Augment, 12 epochs)
- T01 (Trục A - Khởi tạo): Train từ đầu (scratch, không dùng pretrained weights)
- T02 (Trục A - Khởi tạo): Đóng băng backbone (frozen), chỉ train Linear Head
- T03 (Trục B - Augmentation): Thêm Color Jitter (đổi màu, sáng, tương phản)
- T04 (Trục B - Augmentation): CutMix (trộn vùng ảnh và nhãn mềm, alpha=1.0)
- T05 (Trục C - Loss): Label Smoothing (CE + epsilon=0.1)
- T06 (Trục C - Loss): Focal Loss (gamma=2.0)
- T07 (Kết hợp tối ưu): Kết hợp Color Aug + CutMix + Focal Loss

Tự động lưu log, vẽ biểu đồ đường cong, và cập nhật sheet Training trong results.xlsx.
"""
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import train
import model as model_utils

EXPERIMENTS = [
    {
        "exp_id": "T00",
        "axis": "Baseline",
        "diff": "Cấu hình nền (Pretrained finetune, Basic aug, CE)",
        "params": dict(init="finetune", aug="basic", loss="ce", mix=None),
        "notes": "Mốc so sánh nền",
    },
    {
        "exp_id": "T01",
        "axis": "A. Khởi tạo",
        "diff": "Khởi tạo ngẫu nhiên từ đầu (scratch), không dùng pretrained weights",
        "params": dict(init="scratch", aug="basic", loss="ce", mix=None),
        "notes": "Kiểm tra khả năng học từ đầu với ~10k ảnh",
    },
    {
        "exp_id": "T02",
        "axis": "A. Khởi tạo",
        "diff": "Đóng băng toàn bộ backbone (frozen), chỉ train Linear Head",
        "params": dict(init="frozen", aug="basic", loss="ce", mix=None),
        "notes": "Đo chất lượng đặc trưng nguyên bản của backbone",
    },
    {
        "exp_id": "T03",
        "axis": "B. Augmentation",
        "diff": "Thêm biến đổi màu sắc (Color Jitter: brightness, contrast, sat, hue)",
        "params": dict(init="finetune", aug="color", loss="ce", mix=None),
        "notes": "Giảm độ nhạy với điều kiện ánh sáng ngoài trời",
    },
    {
        "exp_id": "T04",
        "axis": "B. Augmentation",
        "diff": "CutMix với alpha=1.0 (trộn vùng ảnh ngẫu nhiên và nhãn mềm)",
        "params": dict(init="finetune", aug="basic", loss="ce", mix="cutmix", mix_alpha=1.0),
        "notes": "Chính quy hoá mạnh, chống overfit",
    },
    {
        "exp_id": "T05",
        "axis": "C. Hàm Loss",
        "diff": "Label Smoothing (epsilon=0.1)",
        "params": dict(init="finetune", aug="basic", loss="ls", label_smoothing=0.1, mix=None),
        "notes": "Giảm overconfidence của mạng",
    },
    {
        "exp_id": "T06",
        "axis": "C. Hàm Loss",
        "diff": "Focal Loss (gamma=2.0)",
        "params": dict(init="finetune", aug="basic", loss="focal", focal_gamma=2.0, mix=None),
        "notes": "Tập trung học các mẫu khó và lớp thiểu số",
    },
    {
        "exp_id": "T07",
        "axis": "Kết hợp tối ưu",
        "diff": "Color Augment + CutMix + Focal Loss (gamma=2.0)",
        "params": dict(init="finetune", aug="color", loss="focal", focal_gamma=2.0, mix="cutmix", mix_alpha=1.0),
        "notes": "Cộng dồn các yếu tố tích cực nhất",
    },
]


def main():
    print("=" * 85)
    print("BƯỚC 2: THỬ NGHIỆM CÔNG THỨC HUẤN LUYỆN (ABLATION STUDY TRÊN CONVNEXT-TINY)")
    print("=" * 85)

    backbone_name = "convnext_tiny"
    results = []
    t00_f1 = None

    for idx, exp in enumerate(EXPERIMENTS, 1):
        exp_id = exp["exp_id"]
        axis = exp["axis"]
        diff = exp["diff"]
        p = exp["params"]

        print(f"\n[{idx}/{len(EXPERIMENTS)}] THÍ NGHIỆM {exp_id} — Trục: {axis}")
        print(f"    Chi tiết: {diff}")
        print("-" * 75)

        cfg = train.Config(
            exp_id=exp_id,
            backbone=backbone_name,
            seed=0,
            fold=0,
            epochs=12,
            batch_size=32,
            lr_backbone=1e-4,
            lr_head=1e-3,
            weight_decay=0.05,
            warmup_epochs=1.0,
            amp=True,
            num_workers=4,
            save_test_predictions=False,
            **p,
        )

        ckpt_path = Path(cfg.out_dir) / exp_id / f"seed{cfg.seed}" / "best_checkpoint.pt"
        hist_file = Path(cfg.out_dir) / exp_id / f"seed{cfg.seed}" / "history.csv"

        # Nếu là T00 và đã có B02 (cùng cấu hình), ta có thể tái sử dụng trực tiếp kết quả B02
        b02_ckpt = Path("runs/B02/seed0/best_checkpoint.pt")
        b02_hist = Path("runs/B02/seed0/history.csv")
        if exp_id == "T00" and not ckpt_path.exists() and b02_ckpt.exists() and b02_hist.exists():
            import shutil
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(b02_ckpt, ckpt_path)
            shutil.copy(b02_hist, hist_file)
            shutil.copy("runs/B02/seed0/config.json", ckpt_path.parent / "config.json")
            if Path("curves/B02_convnext_tiny.png").exists():
                shutil.copy("curves/B02_convnext_tiny.png", Path("curves/T00_convnext_tiny.png"))
            print(f"[*] T00 tái sử dụng kết quả huấn luyện từ B02_convnext_tiny để tiết kiệm tài nguyên.")

        if ckpt_path.exists() and hist_file.exists():
            print(f"[*] Checkpoint {ckpt_path} đã tồn tại! Bỏ qua huấn luyện lại.")
            hist_df = pd.read_csv(hist_file)
        else:
            _ = train.run(cfg)
            hist_df = pd.read_csv(hist_file)

        best_row = hist_df.loc[hist_df["val_macro_f1"].idxmax()]
        val_macro_f1 = float(best_row["val_macro_f1"])
        val_top1 = float(best_row["val_top1"])
        best_epoch = int(best_row["epoch"])

        if exp_id == "T00":
            t00_f1 = val_macro_f1
            delta_str = "0.0000 (mốc)"
        else:
            delta = val_macro_f1 - (t00_f1 if t00_f1 is not None else val_macro_f1)
            delta_str = f"{delta:+.4f}"

        exp_result = {
            "exp_id": exp_id,
            "backbone": backbone_name,
            "trục thay đổi (A–G)": axis,
            "khác T00 ở điểm nào": diff,
            "seed": cfg.seed,
            "macro-F1 val": round(val_macro_f1, 4),
            "top-1 val": round(val_top1, 4),
            "Δ so với T00": delta_str,
            "best_epoch": best_epoch,
            "ghi chú": exp["notes"],
        }
        results.append(exp_result)
        print(f"=> Kết quả {exp_id}: Val Macro-F1 = {val_macro_f1:.4f} | Top-1 = {val_top1:.4f} | Δ = {delta_str}")

    # Xuất ra Sheet Training của results.xlsx
    df_results = pd.DataFrame(results)
    excel_path = Path("results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a" if excel_path.exists() else "w") as writer:
        df_results.to_excel(writer, sheet_name="Training", index=False)
    print(f"\n[OK] Đã lưu bảng kết quả vào file Excel: {excel_path} (Sheet: Training)")

    print("\nBẢNG TỔNG HỢP BƯỚC 2 (SHEET: Training):")
    print("-" * 110)
    print(f"{'Exp ID':<7} | {'Trục':<18} | {'Val Macro-F1':<13} | {'Val Top-1':<10} | {'Δ so T00':<12} | {'Chi tiết':<40}")
    print("-" * 110)
    for r in results:
        print(f"{r['exp_id']:<7} | {r['trục thay đổi (A–G)']:<18} | {r['macro-F1 val']:<13.4f} | {r['top-1 val']:<10.4f} | {r['Δ so với T00']:<12} | {r['khác T00 ở điểm nào'][:40]}")
    print("-" * 110)


if __name__ == "__main__":
    main()
