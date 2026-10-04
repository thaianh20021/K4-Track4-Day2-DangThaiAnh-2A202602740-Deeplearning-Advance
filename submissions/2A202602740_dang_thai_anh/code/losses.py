"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).
Giao diện giữ nguyên:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted"."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Loại loss '{kind}' không được hỗ trợ! Chọn trong [ce, ls, focal, ce_weighted]")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        n_classes = logits.size(-1)
        # NLL loss cho true class
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        # Uniform smooth loss
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) (slide trang 57).

    Khi gamma = 0 và alpha = None, bằng chính xác cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Lấy p_t và log(p_t) của lớp thực tế
        log_pt = log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha.gather(dim=0, index=target)
            loss = at * loss

        return loss.mean()


def class_weights(counts: dict | list | np.ndarray, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo số mẫu hiệu dụng: w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp
    """
    if isinstance(counts, dict):
        num_classes = max(counts.keys()) + 1 if counts else 9
        counts_list = [counts.get(i, 1) for i in range(num_classes)]
    else:
        counts_list = list(counts)

    counts_arr = np.array(counts_list, dtype=np.float32)
    # Tránh chia cho 0
    counts_arr = np.maximum(counts_arr, 1.0)
    c = len(counts_arr)

    if beta <= 0.0:
        inv_counts = 1.0 / counts_arr
        weights = inv_counts / np.mean(inv_counts)
    else:
        # Effective number of samples: En = (1 - beta^n) / (1 - beta)
        effective_num = (1.0 - np.power(beta, counts_arr)) / (1.0 - beta)
        weights = 1.0 / effective_num
        weights = weights * (c / np.sum(weights))

    return torch.tensor(weights, dtype=torch.float32)


def rand_bbox(size: tuple[int, int], lam: float) -> tuple[int, int, int, int]:
    """Tạo toạ độ hộp cắt chữ nhật cho CutMix."""
    w, h = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # Tâm ngẫu nhiên
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0,
              mode: str = "cutmix") -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, float]]:
    """Trộn một batch ảnh và nhãn theo Mixup hoặc CutMix.

    Trả về (x_mixed, (y_a, y_b, lam)) với lam được điều chỉnh theo diện tích thật.
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        x_mixed = x.clone()
        _, _, h, w = x.shape
        bbx1, bby1, bbx2, bby2 = rand_bbox((w, h), lam)
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]
        # Điều chỉnh lam theo diện tích thực của hộp cắt
        lam = 1.0 - float((bbx2 - bbx1) * (bby2 - bby1)) / float(w * h)
    else:
        raise ValueError(f"Mode {mode} không được hỗ trợ! Chọn 'mixup' hoặc 'cutmix'")

    return x_mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
