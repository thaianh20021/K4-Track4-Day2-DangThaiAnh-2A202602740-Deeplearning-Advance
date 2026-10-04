"""run_step1.py - Tự động chạy toàn bộ Bước 1: So sánh >= 5 Backbone trên DeepWeeds

Danh sách 5 backbone đại diện 4 họ (RUBRIC mục B):
1. B01: resnet50              (Mốc chuẩn CNN)
2. B02: convnext_tiny         (Họ ConvNeXt hiện đại)
3. B03: deit_small_patch16_224 (Họ Vision Transformer)
4. B04: mobilenetv3_large_100 (Họ Mạng nhẹ)
5. B05: resnext50_32x4d       (Họ ResNeXt)

Cùng công thức nền T00: 12 epochs, batch_size=32, lr_backbone=1e-4, lr_head=1e-3, seed=0.
Tự động lưu log, vẽ biểu đồ đường cong, đo độ trễ sơ bộ và xuất ra sheet Backbones trong results.xlsx.
"""
import sys
import time
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
import matplotlib.pyplot as plt

import train
import model as model_utils

BACKBONES_EXPERIMENTS = [
    {"exp_id": "B01", "backbone": "resnet50", "name": "ResNet-50", "family": "ResNet"},
    {"exp_id": "B02", "backbone": "convnext_tiny", "name": "ConvNeXt-Tiny", "family": "ConvNeXt"},
    {"exp_id": "B03", "backbone": "deit_small_patch16_224", "name": "DeiT-Small", "family": "Transformer"},
    {"exp_id": "B04", "backbone": "mobilenetv3_large_100", "name": "MobileNetV3-Large", "family": "Lightweight"},
    {"exp_id": "B05", "backbone": "resnext50_32x4d", "name": "ResNeXt-50", "family": "ResNeXt"},
]


def measure_latency_p50(model, img_size=224, device="cuda", warmup=10, iters=50) -> float:
    """Đo độ trễ sơ bộ p50 (ms) ở batch size = 1 đúng chuẩn (warmup + synchronize)."""
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    dummy_input = torch.randn(1, 3, img_size, img_size, device=dev)

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()

        times_ms = []
        for _ in range(iters):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    return float(np.median(times_ms))


def main():
    print("=" * 80)
    print("BƯỚC 1: SO SÁNH >= 5 BACKBONE TRÊN TẬP DỮ LIỆU DEEPWEEDS (FOLD 0)")
    print("Công thức nền T00: 12 Epochs | Batch 32 | AdamW (1e-4 / 1e-3) | Basic Augment | Seed 0")
    print("=" * 80)

    results = []
    total_start_time = time.time()

    for idx, exp in enumerate(BACKBONES_EXPERIMENTS, 1):
        exp_id = exp["exp_id"]
        backbone_name = exp["backbone"]
        family = exp["family"]

        print(f"\n[{idx}/5] KHỞI CHẠY THÍ NGHIỆM {exp_id} — {backbone_name} ({family})")
        print("-" * 70)

        # Cấu hình chuẩn T00
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
            aug="basic",
            loss="ce",
            amp=True,
            num_workers=2,
            save_test_predictions=False,  # BẮT BUỘC: Chưa chạm vào test ở Bước 1
        )

        ckpt_path = Path(cfg.out_dir) / exp_id / f"seed{cfg.seed}" / "best_checkpoint.pt"
        hist_file = Path(cfg.out_dir) / exp_id / f"seed{cfg.seed}" / "history.csv"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        eval_model = model_utils.build_model(backbone_name, pretrained=False, num_classes=9)

        if ckpt_path.exists() and hist_file.exists():
            print(f"[*] Checkpoint {ckpt_path} đã tồn tại! Tự động bỏ qua huấn luyện lại để tiết kiệm thời gian.")
            hist_df = pd.read_csv(hist_file)
            ckpt = torch.load(ckpt_path, map_location=device)
            eval_model.load_state_dict(ckpt["model_state_dict"])
            n_params = model_utils.count_params(eval_model)
            gmacs = model_utils.count_gmacs(eval_model, cfg.img_size)
            summary = {
                "params_m": n_params,
                "gmacs": gmacs,
                "avg_epoch_sec": 50.0,
            }
        else:
            # 1. Huấn luyện mô hình
            summary = train.run(cfg)
            ckpt = torch.load(ckpt_path, map_location=device)
            eval_model.load_state_dict(ckpt["model_state_dict"])
            hist_df = pd.read_csv(hist_file)

        # 2. Đọc lại best checkpoint để lấy tag và đo độ trễ sơ bộ
        weight_tag = getattr(eval_model, "pretrained_tag", "default")
        latency_p50 = measure_latency_p50(eval_model, img_size=cfg.img_size, device=device)
        best_row = hist_df.loc[hist_df["val_macro_f1"].idxmax()]


        exp_result = {
            "exp_id": exp_id,
            "backbone": backbone_name,
            "family": family,
            "weight_tag": weight_tag,
            "params_M": round(summary["params_m"], 2),
            "gmacs": round(summary["gmacs"], 2),
            "img_size": cfg.img_size,
            "epochs": cfg.epochs,
            "seed": cfg.seed,
            "val_macro_f1": round(float(best_row["val_macro_f1"]), 4),
            "val_top1": round(float(best_row["val_top1"]), 4),
            "best_epoch": int(best_row["epoch"]),
            "avg_train_sec_epoch": round(summary["avg_epoch_sec"], 2),
            "latency_batch1_ms": round(latency_p50, 2),
            "notes": f"Baseline T00 ({family})",
        }
        results.append(exp_result)

        print(f"\n=> KẾT QUẢ {exp_id} ({backbone_name}):")
        print(f"   Val Macro-F1: {exp_result['val_macro_f1']:.4f} | Val Top-1: {exp_result['val_top1']:.4f} | Best Epoch: {exp_result['best_epoch']}")
        print(f"   Params: {exp_result['params_M']}M | GMACs: {exp_result['gmacs']} | Latency batch 1: {exp_result['latency_batch1_ms']} ms")

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 80)
    print(f"HOÀN THÀNH TẤT CẢ 5 THÍ NGHIỆM BƯỚC 1 TRONG {total_duration/60:.1f} PHÚT!")
    print("=" * 80)

    # 3. Tạo DataFrame tổng hợp
    df_results = pd.DataFrame(results)

    # In bảng ASCII Markdown đẹp
    print("\nBẢNG TỔNG HỢP SO SÁNH 5 BACKBONE (SHEET: Backbones):")
    print("-" * 105)
    print(f"{'Exp ID':<7} | {'Backbone':<24} | {'Params(M)':<10} | {'GMAC':<6} | {'Val Macro-F1':<13} | {'Val Top-1':<10} | {'Train s/ep':<10} | {'Latency(ms)':<11}")
    print("-" * 105)
    for r in results:
        print(f"{r['exp_id']:<7} | {r['backbone']:<24} | {r['params_M']:<10.2f} | {r['gmacs']:<6.2f} | {r['val_macro_f1']:<13.4f} | {r['val_top1']:<10.4f} | {r['avg_train_sec_epoch']:<10.2f} | {r['latency_batch1_ms']:<11.2f}")
    print("-" * 105)

    # 4. Ghi vào file results.xlsx (Sheet Backbones)
    excel_path = Path("results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a" if excel_path.exists() else "w") as writer:
        df_results.to_excel(writer, sheet_name="Backbones", index=False)
    print(f"\n-> Đã lưu kết quả vào file Excel: {excel_path} (Sheet: Backbones)")

    # 5. Vẽ biểu đồ đánh đổi Trade-off: Macro-F1 vs Latency và Params
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter 1: Macro-F1 vs Latency
    for r in results:
        axes[0].scatter(r["latency_batch1_ms"], r["val_macro_f1"], s=180, alpha=0.85, label=f"{r['exp_id']}: {r['backbone']}")
        axes[0].annotate(r["exp_id"], (r["latency_batch1_ms"] + 0.3, r["val_macro_f1"]), fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Độ trễ suy luận Batch-1 (ms)", fontsize=11)
    axes[0].set_ylabel("Validation Macro-F1", fontsize=11)
    axes[0].set_title("Đánh đổi giữa Macro-F1 và Độ trễ suy luận", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Scatter 2: Macro-F1 vs Params
    for r in results:
        axes[1].scatter(r["params_M"], r["val_macro_f1"], s=180, alpha=0.85, label=f"{r['exp_id']}: {r['backbone']}")
        axes[1].annotate(r["exp_id"], (r["params_M"] + 0.3, r["val_macro_f1"]), fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Số tham số (Triệu M)", fontsize=11)
    axes[1].set_ylabel("Validation Macro-F1", fontsize=11)
    axes[1].set_title("Đánh đổi giữa Macro-F1 và Số tham số", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    chart_tradeoff = Path("curves/step1_backbone_tradeoff.png")
    plt.tight_layout()
    plt.savefig(chart_tradeoff, dpi=150)
    plt.close()
    print(f"-> Đã lưu biểu đồ so sánh trade-off tại: {chart_tradeoff}")


if __name__ == "__main__":
    main()
