"""run_step4.py - Tự động chạy toàn bộ Bước 4: Chạy đa seed (3 seeds) & Đánh giá TEST

Theo quy định nghiêm ngặt S1 - S4 (README.md mục 2.2):
- Tập Test CHỈ ĐƯỢC CHẠM ĐÚNG 1 LẦN cho mỗi seed ở cấu hình chốt cuối cùng!
- Chạy 3 seeds (0, 1, 2) cho:
    1. T00: Mốc chuẩn Baseline (ResNet-50)
    2. F01: Cấu hình tối ưu chốt từ Bước 1 & Bước 2 (ConvNeXt-Tiny + Color Aug + CutMix + Focal Loss)
- Tự động gọi eval.py score và eval.py grade để chấm điểm theo barem chuẩn.
- Ghi kết quả vào Sheet Final và Sheet PerClass trong results.xlsx.
"""
import os
import sys
import subprocess
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

import dataset
import losses
import model as model_utils
import train
from eval import read_pred, compute_metrics, CLASS_NAMES

SEEDS = [0, 1, 2]


def run_experiment_seeds(exp_id, backbone, p_dict, desc, seeds=SEEDS):
    records = []
    print(f"\n========================================================")
    print(f"CHẠY SEEDS CHO CẤU HÌNH: {exp_id} ({desc})")
    print(f"========================================================")

    for seed in seeds:
        print(f"\n---> [{exp_id}] Đang xử lý Seed {seed}...")

        cfg = train.Config(
            exp_id=exp_id,
            backbone=backbone,
            seed=seed,
            fold=0,
            epochs=12,
            batch_size=32,
            lr_backbone=1e-4,
            lr_head=1e-3,
            weight_decay=0.05,
            warmup_epochs=1.0,
            amp=True,
            num_workers=4,
            save_test_predictions=True,  # BẬT ĐÁNH GIÁ TEST
            **p_dict,
        )

        out_p = Path(cfg.out_dir) / exp_id / f"seed{seed}"
        ckpt_path = out_p / "best_checkpoint.pt"
        hist_file = out_p / "history.csv"
        test_pred_file = Path(cfg.pred_dir) / f"{exp_id}_seed{seed}_test.csv"

        # Tự động tái sử dụng checkpoint có sẵn cho Seed 0 (T00 từ B01, F01 từ T07)
        if seed == 0:
            candidate = None
            if exp_id == "T00":
                candidate = "runs/B01/seed0"
            elif exp_id == "F01":
                if Path("runs/T07/seed0/best_checkpoint.pt").exists():
                    candidate = "runs/T07/seed0"
                elif Path("runs/B02/seed0/best_checkpoint.pt").exists():
                    candidate = "runs/B02/seed0"

            needs_copy = False
            if candidate and Path(candidate).exists():
                if not ckpt_path.exists():
                    needs_copy = True
                else:
                    try:
                        cur_ckpt = torch.load(ckpt_path, map_location="cpu")
                        if cur_ckpt.get("cfg", {}).get("backbone") != backbone:
                            needs_copy = True
                    except Exception:
                        needs_copy = True

            if needs_copy and candidate:
                import shutil
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(Path(candidate) / "best_checkpoint.pt", ckpt_path)
                if (Path(candidate) / "history.csv").exists():
                    shutil.copy(Path(candidate) / "history.csv", hist_file)
                if (Path(candidate) / "config.json").exists():
                    shutil.copy(Path(candidate) / "config.json", ckpt_path.parent / "config.json")
                print(f"[*] {exp_id} Seed 0: Tái sử dụng checkpoint từ {candidate} ({backbone}), KHÔNG CẦN train lại!")

        if test_pred_file.exists():
            print(f"[*] Đã tồn tại kết quả Test: {test_pred_file}. Đọc lại trực tiếp...")
            hist_df = pd.read_csv(hist_file) if hist_file.exists() else None
            val_f1 = float(hist_df.loc[hist_df["val_macro_f1"].idxmax()]["val_macro_f1"]) if hist_df is not None else 0.0
        elif ckpt_path.exists():
            # Checkpoint đã có sẵn -> CHỈ SUY LUẬN TEST, KHÔNG TRAIN LẠI!
            print(f"[*] Checkpoint {ckpt_path} đã có sẵn! Chỉ nạp và đánh giá trên tập Test (mất ~15s, KHÔNG train lại)...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m_eval = model_utils.build_model(backbone, pretrained=False, num_classes=9)
            ckpt = torch.load(ckpt_path, map_location=device)
            m_eval.load_state_dict(ckpt["model_state_dict"])
            m_eval.to(device)

            test_df = pd.read_csv("data/labels/test_subset0.csv")
            test_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)
            test_loader = dataset.make_loader(test_df, "images", test_tf, batch_size=cfg.batch_size, train=False, num_workers=4)

            criterion = losses.build_criterion(cfg.loss)
            test_fnames, test_targets, test_logits, _ = train.evaluate(m_eval, test_loader, criterion, device)
            test_probs = train.softmax_np(test_logits)
            test_pred_file.parent.mkdir(parents=True, exist_ok=True)
            train.save_predictions(test_pred_file, test_fnames, test_targets, test_probs)

            hist_df = pd.read_csv(hist_file) if hist_file.exists() else None
            val_f1 = float(hist_df.loc[hist_df["val_macro_f1"].idxmax()]["val_macro_f1"]) if hist_df is not None else float(ckpt.get("best_macro_f1", 0.0))
        else:
            # Chỉ train nếu chưa có checkpoint (ví dụ Seed 1 và Seed 2 mới)
            summary = train.run(cfg)
            val_f1 = summary["best_val_macro_f1"]

        # Đọc file test prediction bằng chuẩn eval.py
        pred = read_pred(str(test_pred_file))
        m = compute_metrics(pred.y_true, pred.y_pred, pred.probs)


        records.append({
            "exp_id": exp_id,
            "cấu hình (backbone + công thức + suy luận)": desc,
            "seed": seed,
            "macro-F1 val": round(val_f1, 4),
            "macro-F1 test": round(m["macro_f1"], 4),
            "top-1 test": round(m["top1"], 4),
            "ECE test": round(m["ece"], 4),
            "metrics": m,
        })
        print(f"      [Seed {seed}] Val F1: {val_f1:.4f} | Test F1: {m['macro_f1']:.4f} | Test Top-1: {m['top1']:.4f} | ECE: {m['ece']:.4f}")

    return records


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chạy Bước 4: Đánh giá Test & Đa Seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Danh sách seed cần chạy (ví dụ: --seeds 0 hoặc --seeds 0 1 2)")
    args = parser.parse_args()

    active_seeds = args.seeds
    print("=" * 85)
    print(f"BƯỚC 4: ĐÁNH GIÁ CHUNG CUỘC TRÊN TẬP TEST (SEEDS: {active_seeds})")
    print("=" * 85)

    # 1. Cấu hình Mốc T00 (ResNet-50 Baseline)
    t00_records = run_experiment_seeds(
        exp_id="T00",
        backbone="resnet50",
        p_dict=dict(init="finetune", aug="basic", loss="ce", mix=None),
        desc="T00: ResNet-50 Baseline (CE, Basic Aug, 1-view)",
        seeds=active_seeds,
    )

    # 2. Cấu hình Tối ưu F01 (ConvNeXt-Tiny + Color Aug + CutMix + Focal Loss)
    f01_records = run_experiment_seeds(
        exp_id="F01",
        backbone="convnext_tiny",
        p_dict=dict(init="finetune", aug="color", loss="focal", focal_gamma=2.0, mix="cutmix", mix_alpha=1.0),
        desc="F01: ConvNeXt-Tiny + Color Aug + CutMix + Focal Loss",
        seeds=active_seeds,
    )


    # 3. Tạo Sheet Final với dòng trung bình mean ± std
    final_rows = []
    for grp_name, recs in [("T00", t00_records), ("F01", f01_records)]:
        for r in recs:
            final_rows.append({
                "exp_id": r["exp_id"],
                "cấu hình (backbone + công thức + suy luận)": r["cấu hình (backbone + công thức + suy luận)"],
                "seed": r["seed"],
                "macro-F1 val": r["macro-F1 val"],
                "macro-F1 test": r["macro-F1 test"],
                "top-1 test": r["top-1 test"],
                "ECE test": r["ECE test"],
            })
        # Dòng tổng hợp mean ± std
        val_f1s = [r["macro-F1 val"] for r in recs]
        test_f1s = [r["macro-F1 test"] for r in recs]
        test_top1s = [r["top-1 test"] for r in recs]
        test_eces = [r["ECE test"] for r in recs]

        final_rows.append({
            "exp_id": f"{grp_name}_mean±std",
            "cấu hình (backbone + công thức + suy luận)": f"{grp_name} (Tổng hợp 3 seeds)",
            "seed": "All",
            "macro-F1 val": f"{np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}",
            "macro-F1 test": f"{np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}",
            "top-1 test": f"{np.mean(test_top1s):.4f} ± {np.std(test_top1s):.4f}",
            "ECE test": f"{np.mean(test_eces):.4f} ± {np.std(test_eces):.4f}",
        })

    df_final = pd.DataFrame(final_rows)

    # 4. Tạo Sheet PerClass
    # Lấy phân tích từng lớp của seed 0 đại diện (hoặc trung bình các seed) cho T00 và F01
    per_class_rows = []

    for grp_name, recs in [("T00 (Baseline)", t00_records), ("F01 (Tối ưu)", f01_records)]:
        # Lấy seed 0
        rep_m = recs[0]["metrics"]
        y_true = recs[0]["metrics"]
        pred = read_pred(f"predictions/{recs[0]['exp_id']}_seed0_test.csv")

        # Tính precision, recall, f1 per class
        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, f1, supp = precision_recall_fscore_support(pred.y_true, pred.y_pred, labels=list(range(9)), zero_division=0)

        for i, cname in enumerate(CLASS_NAMES):
            per_class_rows.append({
                "mô hình": grp_name,
                "lớp": cname,
                "số ảnh test": int(supp[i]),
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "F1": round(float(f1[i]), 4),
            })

    df_per_class = pd.DataFrame(per_class_rows)

    # Ghi vào results.xlsx
    excel_path = Path("results.xlsx")
    writer_kwargs = {"engine": "openpyxl"}
    if excel_path.exists():
        writer_kwargs["mode"] = "a"
        writer_kwargs["if_sheet_exists"] = "replace"
    else:
        writer_kwargs["mode"] = "w"

    with pd.ExcelWriter(excel_path, **writer_kwargs) as writer:
        df_final.to_excel(writer, sheet_name="Final", index=False)
        df_per_class.to_excel(writer, sheet_name="PerClass", index=False)

    print(f"\n[OK] Đã lưu Sheet 'Final' và Sheet 'PerClass' vào {excel_path} thành công!")

    # 5. Gọi eval.py score và grade để in điểm
    print("\n" + "=" * 80)
    print("CHẤM ĐIỂM CHÍNH THỨC VỚI EVAL.PY")
    print("=" * 80)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    subprocess.run([sys.executable, "eval.py", "score", "--pred", "predictions/T00_seed0_test.csv"], env=env)
    subprocess.run([sys.executable, "eval.py", "score", "--pred", "predictions/F01_seed0_test.csv"], env=env)
    subprocess.run([
        sys.executable, "eval.py", "grade",
        "--final", "predictions/F01_seed0_test.csv",
        "--baseline", "predictions/T00_seed0_test.csv",
        "--test-csv", "data/labels/test_subset0.csv",
        "--labels", "data/labels/labels.csv",
    ], env=env)



if __name__ == "__main__":
    main()
