"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations


import time
import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    """
    # 1. Warmup: chạy bỏ qua
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    # 2. Đo thời gian thực
    times_ms = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times = np.array(times_ms)
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    p99 = float(np.percentile(times, 99))
    mean_val = float(np.mean(times))

    return {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "mean": mean_val,
        "n": iters,
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}
    """
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.eval()
    model.to(dev)

    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    if dtype.lower() == "fp16":
        model_eval = model.half()
        dummy_input = dummy_input.half()

        def forward_fn():
            with torch.inference_mode():
                _ = model_eval(dummy_input)
    elif dtype.lower() == "amp":
        def forward_fn():
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    _ = model(dummy_input)
    else:  # fp32
        model_eval = model.float()
        dummy_input = dummy_input.float()

        def forward_fn():
            with torch.inference_mode():
                _ = model_eval(dummy_input)

    res = bench(forward_fn, warmup=warmup, iters=iters, sync=sync)
    p50 = res["p50"]
    throughput = (batch_size / (p50 / 1000.0)) if p50 > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype.upper(),
        "batch": batch_size,
        "img_size": img_size,
        "p50": round(p50, 3),
        "p95": round(res["p95"], 3),
        "p99": round(res["p99"], 3),
        "images_per_s": round(throughput, 2),
        "torch": torch.__version__,
    }


def tta_latency(model, k_views: int, img_size: int = 224, device: str = "cuda", **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63)."""
    base_rep = latency_report(model, batch_size=1, img_size=img_size, dtype="fp32", device=device, **kw)
    p50_1view = base_rep["p50"]

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    dummy_input = torch.randn(1, 3, img_size, img_size, device=dev)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    def tta_fn():
        with torch.inference_mode():
            for _ in range(k_views):
                _ = model(dummy_input)

    res = bench(tta_fn, warmup=5, iters=30, sync=sync)
    res["k_views"] = k_views
    res["expected_k_x_p50"] = round(k_views * p50_1view, 3)
    res["ratio_to_single_view"] = round(res["p50"] / p50_1view, 2) if p50_1view > 0 else 0.0
    return res

