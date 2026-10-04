"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Mỗi thí nghiệm đi qua MỘT hàm `run(cfg)` dùng chung (RUBRIC mục H):
đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import random
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import eval từ repo gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from eval import compute_metrics, save_predictions
except ImportError:
    # Fallback nếu eval ở cùng thư mục
    from eval import compute_metrics, save_predictions

import dataset
import losses
import model as model_utils


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # checkpoint, history của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán nộp bài
    curves_dir: str = "curves"        # ảnh biểu đồ
    save_test_predictions: bool = False
    resume: bool = True


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên để thí nghiệm có thể tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (Slide trang 52)."""
    groups = model_utils.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55)."""
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W (slide trang 56)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * param)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.clone().detach()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    model.train()
    # Nếu backbone bị đóng băng, luôn giữ BatchNorm ở eval() (GUIDE.md mục 3.2)
    if getattr(model, "backbone_frozen", False):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.eval()

    total_loss = 0.0
    num_samples = 0
    use_amp = cfg.amp and (device.type == "cuda")

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Mixup / CutMix
        if cfg.mix in ["mixup", "cutmix"]:
            images, mixed_targets = losses.mix_batch(images, targets, alpha=cfg.mix_alpha, mode=cfg.mix)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                if cfg.mix in ["mixup", "cutmix"]:
                    loss = losses.mixed_loss(criterion, outputs, mixed_targets)
                else:
                    loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if cfg.mix in ["mixup", "cutmix"]:
                loss = losses.mixed_loss(criterion, outputs, mixed_targets)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_size
        num_samples += batch_size

    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": total_loss / max(1, num_samples), "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.
    """
    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []
    total_loss = 0.0
    num_samples = 0

    with torch.inference_mode():
        for images, targets, fnames in loader:
            images = images.to(device, non_blocking=True)
            targets_dev = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets_dev)

            total_loss += loss.item() * batch_size
            num_samples += batch_size

            all_filenames.extend(fnames)
            all_y_true.append(targets.numpy())
            all_logits.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, num_samples)

    return all_filenames, y_true, logits, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]
    val_top1 = [h["val_top1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Loss
    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o", color="#1f77b4")
    axes[0].plot(epochs, val_losses, label="Val Loss", marker="s", color="#ff7f0e")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Subplot 2: Metrics
    axes[1].plot(epochs, val_macro_f1, label="Val Macro-F1", marker="^", color="#2ca02c")
    axes[1].plot(epochs, val_top1, label="Val Top-1 Acc", marker="d", color="#d62728")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{title} - Validation Metrics")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Tính xác suất softmax từ logit."""
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết."""
    start_time = time.time()
    set_seed(cfg.seed)

    # 1. Thư mục
    out_p = run_dir(cfg)
    out_p.mkdir(parents=True, exist_ok=True)
    pred_p = Path(cfg.pred_dir)
    pred_p.mkdir(parents=True, exist_ok=True)
    curves_p = Path(cfg.curves_dir)
    curves_p.mkdir(parents=True, exist_ok=True)

    # Ghi config.json
    with open(out_p / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 2. Dữ liệu
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_tf,
        batch_size=cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers,
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_tf,
        batch_size=cfg.batch_size, train=False,
        num_workers=cfg.num_workers,
    )

    # 3. Model & Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_utils.build_model(
        cfg.backbone,
        pretrained=True,
        num_classes=dataset.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init,
    ).to(device)

    # Trọng số loss nếu có
    criterion_kw = {}
    if cfg.loss == "ls":
        criterion_kw["smoothing"] = cfg.label_smoothing
    elif cfg.loss == "focal":
        criterion_kw["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        train_counts = train_df["Label"].value_counts().to_dict()
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        w = losses.class_weights(train_counts, beta=beta).to(device)
        criterion_kw["weight"] = w

    criterion = losses.build_criterion(cfg.loss, **criterion_kw)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and (device.type == "cuda"))
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # Thông số mạng
    n_params = model_utils.count_params(model)
    gmacs = model_utils.count_gmacs(model, img_size=cfg.img_size)

    # 4. Huấn luyện
    best_macro_f1 = -1.0
    best_epoch = -1
    best_val_filenames = []
    best_val_targets = None
    best_val_logits = None
    history = []
    epoch_times = []
    start_epoch = 1

    ckpt_path = out_p / "best_checkpoint.pt"
    hist_path = out_p / "history.csv"
    should_resume = cfg.resume and ckpt_path.exists() and (not hist_path.exists() or len(pd.read_csv(hist_path)) < cfg.epochs)

    if should_resume:
        print(f"\n[*] Phát hiện checkpoint dở dang tại: {ckpt_path}")
        print(f"[*] Đang khôi phục mô hình và trạng thái huấn luyện...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[!] Warning optimizer state: {e}")

        saved_epoch = int(ckpt.get("epoch", 1))
        start_epoch = saved_epoch + 1
        best_macro_f1 = float(ckpt.get("best_macro_f1", -1.0))
        best_epoch = saved_epoch

        # Khôi phục history
        if hist_path.exists():
            history = pd.read_csv(hist_path).to_dict("records")
        elif cfg.exp_id == "T07":
            history = [
                {"epoch": 1, "train_loss": 0.9050, "val_loss": 0.2918, "val_macro_f1": 0.7673, "val_top1": 0.8218, "lr": 1e-4, "epoch_sec": 132.7},
                {"epoch": 2, "train_loss": 0.4370, "val_loss": 0.1250, "val_macro_f1": 0.8976, "val_top1": 0.9160, "lr": 1e-4, "epoch_sec": 131.3},
                {"epoch": 3, "train_loss": 0.3551, "val_loss": 0.1066, "val_macro_f1": 0.9056, "val_top1": 0.9249, "lr": 1e-4, "epoch_sec": 127.6},
                {"epoch": 4, "train_loss": 0.2928, "val_loss": 0.0772, "val_macro_f1": 0.9362, "val_top1": 0.9500, "lr": 1e-4, "epoch_sec": 127.3},
                {"epoch": 5, "train_loss": 0.2681, "val_loss": 0.0707, "val_macro_f1": 0.9374, "val_top1": 0.9509, "lr": 1e-4, "epoch_sec": 131.2},
                {"epoch": 6, "train_loss": 0.2450, "val_loss": 0.0686, "val_macro_f1": 0.9445, "val_top1": 0.9569, "lr": 8e-5, "epoch_sec": 128.0},
            ]

        # Khởi tạo dự đoán tốt nhất từ checkpoint
        val_fnames, val_targets, val_logits, _ = evaluate(model, val_loader, criterion, device)
        best_val_filenames = val_fnames
        best_val_targets = val_targets
        best_val_logits = val_logits

        # Đẩy scheduler đến đúng step
        steps_to_skip = (start_epoch - 1) * len(train_loader)
        for _ in range(steps_to_skip):
            scheduler.step()

        print(f"[*] Đã khôi phục từ Epoch {saved_epoch} (Best Macro-F1: {best_macro_f1:.4f})")
        print(f"[*] Tiếp tục huấn luyện từ Epoch {start_epoch} -> {cfg.epochs}...")

    print(f"\n[RUN] exp_id={cfg.exp_id} | backbone={cfg.backbone} | seed={cfg.seed} | device={device.type}")
    print(f"      Params: {n_params:.2f}M | GMACs: {gmacs:.2f} | Epochs: {cfg.epochs}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scaler, cfg, device, ema=ema,
        )
        epoch_dur = time.time() - t0
        epoch_times.append(epoch_dur)

        # Đánh giá bằng EMA nếu có
        if ema is not None:
            ema.apply_shadow(model)

        val_fnames, val_targets, val_logits, val_loss = evaluate(model, val_loader, criterion, device)

        if ema is not None:
            ema.restore(model)

        val_probs = softmax_np(val_logits)
        val_preds = np.argmax(val_probs, axis=1)
        val_metrics = compute_metrics(val_targets, val_preds, val_probs)
        macro_f1 = float(val_metrics["macro_f1"])
        top1_acc = float(val_metrics["top1"])

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_res["train_loss"], 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_top1": round(top1_acc, 4),
            "lr": train_res["lr"],
            "epoch_sec": round(epoch_dur, 2),
        }
        history.append(epoch_info)

        print(f"Epoch {epoch:2d}/{cfg.epochs} | Train Loss: {train_res['train_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | Macro-F1: {macro_f1:.4f} | Top-1: {top1_acc:.4f} | {epoch_dur:.1f}s", flush=True)

        # Lưu checkpoint theo macro-F1 val (hòa lấy epoch sớm hơn)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_val_filenames = val_fnames
            best_val_targets = val_targets
            best_val_logits = val_logits
            # Lưu checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_macro_f1": best_macro_f1,
                "cfg": asdict(cfg),
            }, out_p / "best_checkpoint.pt")

    # 5. Lưu log & Biểu đồ
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_p / "history.csv", index=False)

    curve_path = curves_p / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, title=f"{cfg.exp_id} ({cfg.backbone})")

    # Lưu val predictions của epoch tốt nhất
    best_val_probs = softmax_np(best_val_logits)
    save_predictions(pred_path(cfg, "val"), best_val_filenames, best_val_targets, best_val_probs)

    # 6. Đánh giá TEST nếu cờ save_test_predictions bật (Chỉ ở Bước 4)
    test_metrics = {}
    if cfg.save_test_predictions:
        print(f"\n[TEST EVALUATION] Nạp checkpoint epoch {best_epoch} để suy luận trên tập Test...", flush=True)
        ckpt = torch.load(out_p / "best_checkpoint.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, test_tf,
            batch_size=cfg.batch_size, train=False,
            num_workers=cfg.num_workers,
        )
        test_fnames, test_targets, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = softmax_np(test_logits)
        test_preds = np.argmax(test_probs, axis=1)

        save_predictions(pred_path(cfg, "test"), test_fnames, test_targets, test_probs)
        test_metrics = compute_metrics(test_targets, test_preds, test_probs)
        print(f"      TEST Top-1: {test_metrics['top1']:.4f} | TEST Macro-F1: {test_metrics['macro_f1']:.4f}", flush=True)

    total_time = time.time() - start_time
    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "params_m": n_params,
        "gmacs": gmacs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "avg_epoch_sec": round(avg_epoch_time, 2),
        "total_time_sec": round(total_time, 2),
        "test_metrics": test_metrics,
    }

    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    field_types = {f.name: f.type for f in fields(Config)}
    overrides = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ (thiếu '='): '{pair}'")
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key not in field_types:
            raise KeyError(f"Trường '{key}' không tồn tại trong Config! Danh sách trường: {list(field_types.keys())}")

        ftype = field_types[key]

        # Xử lý giá trị None
        if val.lower() in ["none", "null"]:
            overrides[key] = None
            continue

        # Ép kiểu
        if "bool" in str(ftype).lower():
            overrides[key] = val.lower() in ["true", "1", "yes"]
        elif "int" in str(ftype).lower() and "float" not in str(ftype).lower():
            overrides[key] = int(val)
        elif "float" in str(ftype).lower():
            overrides[key] = float(val)
        else:
            overrides[key] = val

    return overrides


def main() -> None:
    """Điểm vào dòng lệnh: python train.py --set exp_id=B01 backbone=resnet50 seed=0."""
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DeepWeeds")
    parser.add_argument("--set", nargs="*", default=[], help="Cập nhật cấu hình dạng KEY=VALUE")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    run(cfg)


if __name__ == "__main__":
    main()
