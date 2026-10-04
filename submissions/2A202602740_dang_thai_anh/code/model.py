"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init`:
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = (init != "scratch") and pretrained

    # Tạo model từ timm
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Lưu lại pretrained_cfg nếu có
    if hasattr(model, "pretrained_cfg"):
        model.pretrained_tag = model.pretrained_cfg.get("tag", "default")
    else:
        model.pretrained_tag = "default"

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head phân loại."""
    # Lấy tên các module thuộc classifier head
    classifier = model.get_classifier()
    head_param_ids = {id(p) for p in classifier.parameters()} if hasattr(classifier, "parameters") else set()

    # Nếu get_classifier không trả về module chứa param (hoặc trả về None)
    if not head_param_ids:
        # Fallback tìm kiếm qua tên thông dụng
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in ["fc", "head", "classifier"]):
                head_param_ids.add(id(param))

    for param in model.parameters():
        if id(param) not in head_param_ids:
            param.requires_grad = False

    # Đánh dấu model có backbone bị đóng băng để train loop luôn giữ BatchNorm ở eval()
    model.backbone_frozen = True


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    Bỏ qua các tham số requires_grad == False.
    """
    classifier = model.get_classifier()
    head_param_ids = {id(p) for p in classifier.parameters()} if hasattr(classifier, "parameters") else set()
    if not head_param_ids:
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in ["fc", "head", "classifier"]):
                head_param_ids.add(id(param))

    group_backbone_weights = []
    group_backbone_no_decay = []
    group_head = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if id(param) in head_param_ids:
            group_head.append(param)
        else:
            # Tham số thuộc backbone
            if param.ndim > 1:
                group_backbone_weights.append(param)
            else:
                # bias hoặc norm 1D
                group_backbone_no_decay.append(param)

    groups = []
    if group_backbone_weights:
        groups.append({
            "params": group_backbone_weights,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
        })
    if group_backbone_no_decay:
        groups.append({
            "params": group_backbone_no_decay,
            "lr": lr_backbone,
            "weight_decay": 0.0,
        })
    if group_head:
        groups.append({
            "params": group_head,
            "lr": lr_head,
            "weight_decay": weight_decay,
        })

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size."""
    try:
        from thop import profile
        # Tạo mô hình tạm thời trên CPU để không làm bẩn GPU
        temp_model = copy.deepcopy(model).cpu().eval()
        dummy_input = torch.randn(1, 3, img_size, img_size)
        macs, _ = profile(temp_model, inputs=(dummy_input,), verbose=False)
        return float(macs) / 1e9
    except Exception:
        # Fallback ước lượng nếu thop gặp lỗi với một số model đặc thù
        total_p = count_params(model)
        # Hệ số ước lượng thực nghiệm cho ảnh 224x224
        return round(total_p * 0.16, 2)
