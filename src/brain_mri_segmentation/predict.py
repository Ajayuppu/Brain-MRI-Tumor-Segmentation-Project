"""Inference: TIFF image(s) -> binary 256x256 PNG mask(s) using a trained run and its validation-selected threshold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from .data import normalise, read_image
from .select import load_trained_model

MASK_FOREGROUND_VALUE = 255


def predict_masks(
    run_dir: Path, selection_file: Path, image_paths: list[str], output_dir: Path, device: str
) -> list[Path]:
    """Write one <image name>_predicted_mask.png per input image and return their paths."""
    config = yaml.safe_load(open(run_dir / "config.yaml"))
    statistics = json.load(open(run_dir / "channel_statistics.json"))
    threshold = json.load(open(selection_file))["selected_threshold"]
    model = load_trained_model(run_dir, config["base_channels"], device)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for image_path in image_paths:
        image = normalise(read_image(image_path), statistics).transpose(2, 0, 1)
        with torch.no_grad():
            probability = torch.sigmoid(model(torch.from_numpy(image)[None].to(device)))[0, 0].cpu().numpy()
        mask = ((probability >= threshold) * MASK_FOREGROUND_VALUE).astype(np.uint8)
        target = output_dir / f"{Path(image_path).stem}_predicted_mask.png"
        Image.fromarray(mask).save(target)
        written.append(target)
    return written
