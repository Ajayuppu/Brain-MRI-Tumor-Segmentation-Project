"""Loss conditions.

bce                     baseline: BCEWithLogits
bce_dice                proposed: BCE + dice_weight * soft Dice pooled over the whole batch
                        (not a mean of per-slice Dice)
bce_positive_only_dice  diagnostic: Dice term computed only on slices whose mask has a lesion (an all-empty batch
                        contributes a differentiable zero)
weighted_bce            diagnostic: BCE with pos_weight = min((1 - f) / f, cap), f = training foreground pixel fraction
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import torch
import torch.nn.functional as F

from .data import IMAGE_SIZE


def soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, smoothing: float = 1.0, positive_slices_only: bool = False
) -> torch.Tensor:
    """1 - (2*sum(p*y) + s) / (sum(p) + sum(y) + s), pooled over every pixel in the batch."""
    probabilities = torch.sigmoid(logits)
    if positive_slices_only:
        keep = target.flatten(1).sum(1) > 0
        if keep.sum() == 0:
            return logits.sum() * 0.0
        probabilities, target = probabilities[keep], target[keep]
    intersection = (probabilities * target).sum()
    return 1 - (2 * intersection + smoothing) / (probabilities.sum() + target.sum() + smoothing)


def positive_class_weight_from_training(train_manifest: pd.DataFrame, cap: float) -> float:
    """min((1 - f) / f, cap) with f the foreground pixel fraction of the training masks (from the manifest)."""
    foreground_fraction = train_manifest.foreground_pixels.sum() / (len(train_manifest) * IMAGE_SIZE * IMAGE_SIZE)
    return float(min((1 - foreground_fraction) / foreground_fraction, cap))


def build_loss(
    name: str,
    dice_weight: float | None = None,
    positive_class_weight: float | None = None,
    soft_dice_smoothing: float = 1.0,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return loss(logits, target) for one of the four loss conditions."""
    if name == "weighted_bce" and positive_class_weight is None:
        raise ValueError("weighted_bce needs positive_class_weight")

    def loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if name == "weighted_bce":
            weight = torch.tensor([positive_class_weight], device=logits.device)
            return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        if name == "bce":
            return bce
        if name == "bce_dice":
            return bce + dice_weight * soft_dice_loss(logits, target, soft_dice_smoothing)
        if name == "bce_positive_only_dice":
            return bce + dice_weight * soft_dice_loss(logits, target, soft_dice_smoothing, positive_slices_only=True)
        raise ValueError(name)

    return loss
