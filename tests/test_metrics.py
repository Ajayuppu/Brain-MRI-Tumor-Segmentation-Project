"""Metric conventions from the project description (section 5)."""

import numpy as np
import pytest

from brain_mri_segmentation.metrics import (
    best_threshold,
    confusion_counts,
    dice_score,
    iou_score,
    per_slice_table,
    summarise,
)

EMPTY = np.zeros((256, 256), np.uint8)


def square(row_start, row_end, column_start, column_end):
    mask = EMPTY.copy()
    mask[row_start:row_end, column_start:column_end] = 1
    return mask


def test_both_empty_scores_one():
    assert dice_score(*confusion_counts(EMPTY, EMPTY)) == 1.0
    assert iou_score(*confusion_counts(EMPTY, EMPTY)) == 1.0


def test_exactly_one_empty_scores_zero():
    lesion = square(0, 2, 0, 2)
    assert dice_score(*confusion_counts(lesion, EMPTY)) == 0.0
    assert dice_score(*confusion_counts(EMPTY, lesion)) == 0.0
    assert iou_score(*confusion_counts(EMPTY, lesion)) == 0.0


def test_partial_overlap_dice_and_iou():
    prediction, truth = square(0, 10, 0, 10), square(0, 10, 5, 15)  # TP 50, FP 50, FN 50
    assert dice_score(*confusion_counts(prediction, truth)) == pytest.approx(0.5)
    assert iou_score(*confusion_counts(prediction, truth)) == pytest.approx(1 / 3)


def test_summary_separates_positive_and_empty_slices():
    truths = [EMPTY, square(0, 10, 0, 10)]
    predictions = [square(0, 1, 0, 3), EMPTY]
    summary = summarise(per_slice_table(predictions, truths, ["a", "b"], ["1", "2"]))
    assert summary["positive_slice_dice"] == 0.0
    assert summary["empty_slices_with_false_positives"] == 1
    assert summary["empty_slice_false_positive_pixels_total"] == 3
    assert summary["pooled_foreground_dice"] == 0.0


def test_always_empty_reference():
    truths = [EMPTY, EMPTY, square(0, 4, 0, 4)]
    summary = summarise(per_slice_table([EMPTY] * 3, truths, ["a", "a", "b"], ["1", "2", "3"]))
    assert summary["mean_slice_dice"] == pytest.approx(2 / 3)
    assert summary["positive_slice_dice"] == 0.0
    assert summary["empty_slices_with_false_positives"] == 0
    assert summary["pixel_accuracy"] == pytest.approx(1 - 16 / (3 * 256 * 256))


def test_patient_dice_pools_pixels_within_patient():
    truths = [square(0, 10, 0, 10), square(0, 10, 0, 10)]
    predictions = [square(0, 10, 0, 10), EMPTY]  # patient a: TP 100, FN 100 -> 2/3
    summary = summarise(per_slice_table(predictions, truths, ["a", "a"], ["1", "2"]))
    assert summary["patient_dice_mean"] == pytest.approx(2 / 3)


def test_rejects_resized_masks():
    with pytest.raises(AssertionError):
        per_slice_table([np.zeros((128, 128))], [np.zeros((128, 128))], ["a"], ["1"])


def test_best_threshold_ties_go_to_lowest_threshold():
    sweep = {
        0.3: {"pooled_foreground_dice": 0.8},
        0.5: {"pooled_foreground_dice": 0.8},
        0.7: {"pooled_foreground_dice": 0.7},
    }
    assert best_threshold(sweep, "pooled_foreground_dice") == 0.3
