"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Softmax ổn định trên numpy."""
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def predict_logits(model, loader, device="cuda", view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.eval()
    model.to(dev)

    all_fnames = []
    all_targets = []
    all_logits = []

    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:
                images, targets, filenames = batch
            else:
                images, targets = batch[:2]
                filenames = [f"img_{i}.jpg" for i in range(len(targets))]

            images = images.to(dev, non_blocking=True)
            if view is not None:
                images = view(images)

            logits = model(images)
            all_fnames.extend(filenames)
            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    return (
        np.array(all_fnames),
        np.concatenate(all_targets, axis=0) if all_targets else np.array([]),
        np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 9)),
    )


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`."""
    _, _, h, w = x.shape
    if h < crop or w < crop:
        x = F.interpolate(x, size=(max(h, crop), max(w, crop)), mode="bilinear", align_corners=False)
        _, _, h, w = x.shape

    top_left = x[:, :, :crop, :crop]
    top_right = x[:, :, :crop, w - crop:]
    bot_left = x[:, :, h - crop:, :crop]
    bot_right = x[:, :, h - crop:, w - crop:]
    center = x[:, :, (h - crop) // 2:(h - crop) // 2 + crop, (w - crop) // 2:(w - crop) // 2 + crop]

    return [top_left, top_right, bot_left, bot_right, center]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán xác suất."""
    if space == "prob":
        probs_list = [softmax_np(l) for l in logits_per_view]
        return np.mean(probs_list, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        return softmax_np(avg_logits)
    else:
        raise ValueError(f"Không hỗ trợ space='{space}', chỉ nhận 'prob' hoặc 'logit'.")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình."""
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)."""
    logits = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)

    def nll_obj(T):
        scaled = logits / max(T, 1e-4)
        probs = softmax_np(scaled)
        correct_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
        return -np.mean(np.log(correct_probs))

    res = minimize_scalar(nll_obj, bounds=(0.05, 5.0), method="bounded")
    return float(res.x)


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    return softmax_np(logits / max(T, 1e-4))


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước (chính xác lúc suy luận)."""
    model_eval = copy.deepcopy(model).eval()

    def _fuse_recursive(m):
        for child_name, child in m.named_children():
            _fuse_recursive(child)
        try:
            torch.nn.utils.fusion.fuse_conv_bn_eval(m)
        except Exception:
            pass

    try:
        _fuse_recursive(model_eval)
    except Exception:
        pass

    return model_eval
