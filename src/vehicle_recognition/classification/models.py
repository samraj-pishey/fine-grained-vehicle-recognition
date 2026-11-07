"""Torchvision classifier builders (single 196-way head; no auxiliary heads)."""

from __future__ import annotations

import torch
from torch import nn

ARCHS = ("mobilenet_v3_large", "efficientnet_v2_s", "convnext_tiny")


def build_model(arch: str, num_classes: int = 196, pretrained: bool = True, dropout: float | None = None) -> nn.Module:
    import torchvision.models as tvm

    if arch == "mobilenet_v3_large":
        w = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.mobilenet_v3_large(weights=w)
        if dropout is not None:
            m.classifier[2].p = dropout
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_v2_s":
        w = tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.efficientnet_v2_s(weights=w)
        if dropout is not None:
            m.classifier[0].p = dropout
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "convnext_tiny":
        w = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch {arch}; choose from {ARCHS}")
    return m


def param_groups(model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]

    def split(ps):
        decay = [p for p in ps if p.ndim > 1]
        no_decay = [p for p in ps if p.ndim <= 1]
        return decay, no_decay

    bd, bn = split(backbone)
    hd, hn = split(head)
    return [
        {"params": bd, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"},
        {"params": bn, "lr": backbone_lr, "weight_decay": 0.0, "name": "backbone_nd"},
        {"params": hd, "lr": head_lr, "weight_decay": weight_decay, "name": "head"},
        {"params": hn, "lr": head_lr, "weight_decay": 0.0, "name": "head_nd"},
    ]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head_ids = {id(p) for p in model.classifier.parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def export_onnx(model: nn.Module, path: str, img_size: int, opset: int = 17) -> None:
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"], opset_version=opset,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
