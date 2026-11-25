"""Loss conditions: finite values, batch-pooled Dice, positive-only Dice ignores empty slices, weighted BCE."""

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from brain_mri_segmentation.losses import build_loss, positive_class_weight_from_training, soft_dice_loss


def make_batch():
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 32, 32)
    target = torch.zeros(4, 1, 32, 32)
    target[0, :, :5, :5] = 1
    return logits, target


@pytest.mark.parametrize("name", ["bce", "bce_dice", "bce_positive_only_dice", "weighted_bce"])
def test_every_loss_is_finite_and_differentiable(name):
    logits, target = make_batch()
    logits.requires_grad_(True)
    loss = build_loss(name, dice_weight=1.0, positive_class_weight=10.0)(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None


def test_soft_dice_is_pooled_over_the_batch():
    logits, target = make_batch()
    probabilities = torch.sigmoid(logits)
    expected = 1 - (2 * (probabilities * target).sum() + 1) / (probabilities.sum() + target.sum() + 1)
    assert torch.allclose(soft_dice_loss(logits, target), expected)


def test_bce_dice_equals_bce_plus_weighted_dice():
    logits, target = make_batch()
    expected = F.binary_cross_entropy_with_logits(logits, target) + 2.0 * soft_dice_loss(logits, target)
    assert torch.allclose(build_loss("bce_dice", dice_weight=2.0)(logits, target), expected)


def test_positive_only_dice_ignores_empty_slices():
    logits, target = make_batch()
    assert soft_dice_loss(logits, torch.zeros_like(target), positive_slices_only=True).item() == 0.0
    assert torch.allclose(
        soft_dice_loss(logits, target, positive_slices_only=True), soft_dice_loss(logits[:1], target[:1])
    )


def test_weighted_bce_weight_from_training_foreground_fraction_with_cap():
    manifest = pd.DataFrame({"foreground_pixels": [655, 0, 0, 0]})  # f = 655 / (4 * 65536)
    fraction = 655 / (4 * 65536)
    assert positive_class_weight_from_training(manifest, cap=1000) == pytest.approx((1 - fraction) / fraction)
    assert positive_class_weight_from_training(manifest, cap=100) == 100


def test_weighted_bce_requires_a_weight():
    with pytest.raises(ValueError):
        build_loss("weighted_bce")
