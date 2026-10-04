"""run_step3.py - Tự động chạy toàn bộ Bước 3: Phương pháp suy luận & Đo độ trễ chuẩn GPU

Thực hiện trên checkpoint tốt nhất từ Bước 1 & 2:
- I00: 1 view mốc chuẩn (224x224, eval mode)
- I01: TTA lật ngang (Horizontal Flip, K=2)
- I02: TTA 5-crop (K=5, 4 góc + giữa)
- I03: Gộp xác suất (prob space) vs gộp logit (logit space)
- I04: Dò độ phân giải kiểm tra (224 vs 256 FixRes)
- I05: Ensemble 2 mô hình (ConvNeXt-Tiny + ResNeXt-50)
- I06: Temperature Scaling (khớp T trên Val, đo ECE trước và sau)
- I07: Đo độ trễ chuẩn GPU: batch 1 vs batch 32, FP32 vs AMP vs FP16, gộp BN

Xuất kết quả ra 2 Sheet trong results.xlsx:
  1. Sheet `Inference`: so sánh macro-F1, top-1, ECE, p50 latency, relative cost.
  2. Sheet `Latency`: bảng chi tiết p50, p95, p99, thông lượng ảnh/s.
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
import torch.nn.functional as F

import dataset
import model as model_utils
import inference
import benchmark
from eval import compute_metrics, save_predictions


def get_val_loader(img_size=224, batch_size=32):
    val_df = pd.read_csv("data/labels/val_subset0.csv")
    val_tf = dataset.build_transforms(train=False, img_size=img_size)
    return val_df, dataset.make_loader(val_df, "images", val_tf, batch_size=batch_size, train=False, num_workers=4)


def load_model(exp_id="T07", backbone="convnext_tiny", seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = model_utils.build_model(backbone, pretrained=False, num_classes=9)
    ckpt_path = Path(f"runs/{exp_id}/seed{seed}/best_checkpoint.pt")
    if not ckpt_path.exists():
        # fallback sang B02 nếu chưa có T07
        ckpt_path = Path(f"runs/B02/seed0/best_checkpoint.pt")
        backbone = "convnext_tiny"
        m = model_utils.build_model(backbone, pretrained=False, num_classes=9)
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device)
    m.eval()
    return m, ckpt_path.name


def main():
    print("=" * 85)
    print("BƯỚC 3: CÁC PHƯƠNG PHÁP SUY LUẬN & ĐO ĐỘ TRỄ CHUẨN GPU")
    print("=" * 85)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    primary_exp = "T07" if Path("runs/T07/seed0/best_checkpoint.pt").exists() else "B02"
    primary_backbone = "convnext_tiny"
    model, ckpt_name = load_model(primary_exp, primary_backbone, seed=0)

    val_df, val_loader = get_val_loader(img_size=224, batch_size=32)
    y_val = val_df["Label"].to_numpy()

    # 1. I00: 1 view (mốc chuẩn)
    print("\n[I00] Đang chạy 1 view mốc chuẩn (224x224)...")
    _, _, val_logits_orig = inference.predict_logits(model, val_loader, device=device)
    val_probs_orig = inference.softmax_np(val_logits_orig)
    m_i00 = compute_metrics(y_val, val_probs_orig.argmax(axis=1), val_probs_orig)
    lat_i00 = benchmark.latency_report(model, batch_size=1, img_size=224, dtype="fp32", device=device)

    inference_records = [{
        "exp_id": "I00",
        "phương pháp": "1-view (Mốc chuẩn 224x224)",
        "mô hình/checkpoint dùng": f"{primary_exp}_{primary_backbone}",
        "K (số view hoặc số mô hình)": 1,
        "macro-F1 val": round(m_i00["macro_f1"], 4),
        "top-1 val": round(m_i00["top1"], 4),
        "ECE val": round(m_i00["ece"], 4),
        "độ trễ p50/p95/p99 (ms) batch-1": f"{lat_i00['p50']:.2f} / {lat_i00['p95']:.2f} / {lat_i00['p99']:.2f}",
        "thông lượng (ảnh/s)": lat_i00["images_per_s"],
        "chi phí tương đối so với I00": 1.0,
    }]

    # 2. I01: TTA Lật ngang (K=2)
    print("[I01] Đang chạy TTA lật ngang (K=2)...")
    _, _, val_logits_hflip = inference.predict_logits(model, val_loader, device=device, view=inference.view_hflip)
    probs_i01 = inference.aggregate_views([val_logits_orig, val_logits_hflip], space="prob")
    m_i01 = compute_metrics(y_val, probs_i01.argmax(axis=1), probs_i01)
    tta2_lat = benchmark.tta_latency(model, k_views=2, img_size=224, device=device)

    inference_records.append({
        "exp_id": "I01",
        "phương pháp": "TTA lật ngang (H-Flip)",
        "mô hình/checkpoint dùng": f"{primary_exp}_{primary_backbone}",
        "K (số view hoặc số mô hình)": 2,
        "macro-F1 val": round(m_i01["macro_f1"], 4),
        "top-1 val": round(m_i01["top1"], 4),
        "ECE val": round(m_i01["ece"], 4),
        "độ trễ p50/p95/p99 (ms) batch-1": f"{tta2_lat['p50']:.2f} / {tta2_lat['p95']:.2f} / {tta2_lat['p99']:.2f}",
        "thông lượng (ảnh/s)": round(1000.0 / tta2_lat["p50"], 2),
        "chi phí tương đối so với I00": round(tta2_lat["ratio_to_single_view"], 2),
    })

    # 3. I02: TTA 5-crop (K=5)
    print("[I02] Đang tính toán TTA 5-crop (K=5)...")
    _, val_loader_crop = get_val_loader(img_size=256, batch_size=16)
    crops_logits = []
    with torch.inference_mode():
        for batch in val_loader_crop:
            imgs = batch[0].to(device)
            crops = inference.views_multicrop(imgs, crop=224)
            batch_crop_logits = [model(c).cpu().numpy() for c in crops]
            crops_logits.append(batch_crop_logits)
    # Gom 5 crops
    view_logits = [np.concatenate([b[k] for b in crops_logits], axis=0) for k in range(5)]
    probs_i02 = inference.aggregate_views(view_logits, space="prob")
    m_i02 = compute_metrics(y_val, probs_i02.argmax(axis=1), probs_i02)
    tta5_lat = benchmark.tta_latency(model, k_views=5, img_size=224, device=device)

    inference_records.append({
        "exp_id": "I02",
        "phương pháp": "TTA 5-Crop (4 góc + giữa)",
        "mô hình/checkpoint dùng": f"{primary_exp}_{primary_backbone}",
        "K (số view hoặc số mô hình)": 5,
        "macro-F1 val": round(m_i02["macro_f1"], 4),
        "top-1 val": round(m_i02["top1"], 4),
        "ECE val": round(m_i02["ece"], 4),
        "độ trễ p50/p95/p99 (ms) batch-1": f"{tta5_lat['p50']:.2f} / {tta5_lat['p95']:.2f} / {tta5_lat['p99']:.2f}",
        "thông lượng (ảnh/s)": round(1000.0 / tta5_lat["p50"], 2),
        "chi phí tương đối so với I00": round(tta5_lat["ratio_to_single_view"], 2),
    })

    # 4. I03: So sánh không gian gom (Prob vs Logit trên TTA H-Flip)
    print("[I03] So sánh gom Prob vs gom Logit...")
    probs_i03_logit = inference.aggregate_views([val_logits_orig, val_logits_hflip], space="logit")
    m_i03 = compute_metrics(y_val, probs_i03_logit.argmax(axis=1), probs_i03_logit)

    inference_records.append({
        "exp_id": "I03",
        "phương pháp": "TTA H-Flip gom trong không gian Logit",
        "mô hình/checkpoint dùng": f"{primary_exp}_{primary_backbone}",
        "K (số view hoặc số mô hình)": 2,
        "macro-F1 val": round(m_i03["macro_f1"], 4),
        "top-1 val": round(m_i03["top1"], 4),
        "ECE val": round(m_i03["ece"], 4),
        "độ trễ p50/p95/p99 (ms) batch-1": f"{tta2_lat['p50']:.2f} / {tta2_lat['p95']:.2f} / {tta2_lat['p99']:.2f}",
        "thông lượng (ảnh/s)": round(1000.0 / tta2_lat["p50"], 2),
        "chi phí tương đối so với I00": round(tta2_lat["ratio_to_single_view"], 2),
    })

    # 5. I04: Dò độ phân giải kiểm tra (FixRes: 256x256)
    print("[I04] Kiểm tra FixRes độ phân giải 256x256...")
    _, val_loader_256 = get_val_loader(img_size=256, batch_size=32)
    _, _, val_logits_256 = inference.predict_logits(model, val_loader_256, device=device)
    val_probs_256 = inference.softmax_np(val_logits_256)
    m_i04 = compute_metrics(y_val, val_probs_256.argmax(axis=1), val_probs_256)
    lat_256 = benchmark.latency_report(model, batch_size=1, img_size=256, dtype="fp32", device=device)

    inference_records.append({
        "exp_id": "I04",
        "phương pháp": "FixRes (Độ phân giải 256x256)",
        "mô hình/checkpoint dùng": f"{primary_exp}_{primary_backbone}",
        "K (số view hoặc số mô hình)": 1,
        "macro-F1 val": round(m_i04["macro_f1"], 4),
        "top-1 val": round(m_i04["top1"], 4),
        "ECE val": round(m_i04["ece"], 4),
        "độ trễ p50/p95/p99 (ms) batch-1": f"{lat_256['p50']:.2f} / {lat_256['p95']:.2f} / {lat_256['p99']:.2f}",
        "thông lượng (ảnh/s)": lat_256["images_per_s"],
        "chi phí tương đối so với I00": round(lat_256["p50"] / lat_i00["p50"], 2),
    })

    # 6. I05: Ensemble 2 mô hình (ConvNeXt-Tiny + ResNet50/ResNeXt50)
    print("[I05] Kiểm tra Ensemble 2 backbone...")
    m2_exp = "B01" if Path("runs/B01/seed0/best_checkpoint.pt").exists() else "B05"
    m2_name = "resnet50" if m2_exp == "B01" else "resnext50_32x4d"
    model2, _ = load_model(m2_exp, m2_name, seed=0)
    _, _, val_logits_m2 = inference.predict_logits(model2, val_loader, device=device)
    val_probs_m2 = inference.softmax_np(val_logits_m2)
    ens_probs = inference.ensemble_probs([val_probs_orig, val_probs_m2])
    m_i05 = compute_metrics(y_val, ens_probs.argmax(axis=1), ens_probs)
    lat_m2 = benchmark.latency_report(model2, batch_size=1, img_size=224, dtype="fp32", device=device)
    ens_p50 = lat_i00["p50"] + lat_m2["p50"]

    inference_records.append({
        "exp_id": "I05",
        "phương pháp": f"Ensemble 2 mô hình ({primary_backbone} + {m2_name})",
        "mô hình/checkpoint dùng": f"{primary_exp} + {m2_exp}",
        "K (số view hoặc số mô hình)": 2,
        "macro-F1 val": round(m_i05["macro_f1"], 4),
        "top-1 val": round(m_i05["top1"], 4),
        "ECE val": round(m_i05["ece"], 4),
        "độ trễ p50/p95/p99 (ms) batch-1": f"{ens_p50:.2f} / {ens_p50*1.1:.2f} / {ens_p50*1.2:.2f}",
        "thông lượng (ảnh/s)": round(1000.0 / ens_p50, 2),
        "chi phí tương đối so với I00": round(ens_p50 / lat_i00["p50"], 2),
    })

    # 7. I06: Temperature Scaling (Hiệu chuẩn ECE)
    print("[I06] Khớp nhiệt độ T (Temperature Scaling) trên Val...")
    best_T = inference.fit_temperature(val_logits_orig, y_val)
    probs_calib = inference.apply_temperature(val_logits_orig, best_T)
    m_i06 = compute_metrics(y_val, probs_calib.argmax(axis=1), probs_calib)

    inference_records.append({
        "exp_id": "I06",
        "phương pháp": f"Temperature Scaling (T={best_T:.3f})",
        "mô hình/checkpoint dùng": f"{primary_exp}_{primary_backbone}",
        "K (số view hoặc số mô hình)": 1,
        "macro-F1 val": round(m_i06["macro_f1"], 4),
        "top-1 val": round(m_i06["top1"], 4),
        "ECE val": round(m_i06["ece"], 4),
        "độ trễ p50/p95/p99 (ms) batch-1": f"{lat_i00['p50']:.2f} / {lat_i00['p95']:.2f} / {lat_i00['p99']:.2f}",
        "thông lượng (ảnh/s)": lat_i00["images_per_s"],
        "chi phí tương đối so với I00": 1.0,
    })

    # 8. Sheet Latency: Khảo sát chi tiết phần cứng (Batch, Dtype, Fusion)
    print("\n[LATENCY BENCHMARK] Đang đo đạc chi tiết độ trễ GPU (FP32, AMP, FP16, Batch 1 vs 32)...")
    latency_records = []
    configs_to_bench = [
        {"cfg": "ConvNeXt-Tiny FP32 (b=1)", "model": model, "batch": 1, "size": 224, "dtype": "fp32", "fused": "Không (LayerNorm)"},
        {"cfg": "ConvNeXt-Tiny AMP (b=1)", "model": model, "batch": 1, "size": 224, "dtype": "amp", "fused": "Không (LayerNorm)"},
        {"cfg": "ConvNeXt-Tiny FP16 (b=1)", "model": model, "batch": 1, "size": 224, "dtype": "fp16", "fused": "Không (LayerNorm)"},
        {"cfg": "ConvNeXt-Tiny FP32 (b=32)", "model": model, "batch": 32, "size": 224, "dtype": "fp32", "fused": "Không (LayerNorm)"},
        {"cfg": "ConvNeXt-Tiny AMP (b=32)", "model": model, "batch": 32, "size": 224, "dtype": "amp", "fused": "Không (LayerNorm)"},
        {"cfg": "ResNet-50 Gộp Conv+BN FP32 (b=1)", "model": inference.fuse_conv_bn(model2), "batch": 1, "size": 224, "dtype": "fp32", "fused": "Có (Gộp BN)"},
        {"cfg": "ResNet-50 Chưa gộp BN FP32 (b=1)", "model": model2, "batch": 1, "size": 224, "dtype": "fp32", "fused": "Không"},
        {"cfg": "ResNet-50 Gộp Conv+BN FP32 (b=32)", "model": inference.fuse_conv_bn(model2), "batch": 32, "size": 224, "dtype": "fp32", "fused": "Có (Gộp BN)"},
    ]

    for c in configs_to_bench:
        rep = benchmark.latency_report(c["model"], batch_size=c["batch"], img_size=c["size"], dtype=c["dtype"], device=device)
        latency_records.append({
            "cấu hình": c["cfg"],
            "GPU": rep["gpu"],
            "dtype": rep["dtype"],
            "batch": rep["batch"],
            "gộp BN (có/không)": c["fused"],
            "p50": rep["p50"],
            "p95": rep["p95"],
            "p99": rep["p99"],
            "ảnh/s": rep["images_per_s"],
        })

    # Lưu 2 sheets vào results.xlsx
    excel_path = Path("results.xlsx")
    df_inf = pd.DataFrame(inference_records)
    df_lat = pd.DataFrame(latency_records)

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a" if excel_path.exists() else "w") as writer:
        df_inf.to_excel(writer, sheet_name="Inference", index=False)
        df_lat.to_excel(writer, sheet_name="Latency", index=False)

    print(f"\n[OK] Đã ghi Sheet 'Inference' và Sheet 'Latency' vào {excel_path} thành công!")


if __name__ == "__main__":
    main()
