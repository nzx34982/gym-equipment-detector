from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .dataset_voc import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """Build Faster R-CNN and replace its classifier for this dataset."""
    weights = (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        if pretrained
        else None
    )
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
        weights_backbone=None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location=device)


def load_model_for_inference(checkpoint_path: Path, device: torch.device):
    """Load either the project's metadata checkpoint or a legacy state dictionary."""
    checkpoint = _load_checkpoint(checkpoint_path, device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model = build_model(pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
