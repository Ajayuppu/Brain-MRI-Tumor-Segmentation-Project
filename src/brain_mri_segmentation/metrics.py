"""Evaluation metrics at native 256x256.

Per-slice Dice/IoU convention: prediction and mask both empty -> 1; exactly one of them empty -> 0.
Pooled foreground Dice/IoU sum TP/FP/FN over all slices. Patient Dice pools TP/FP/FN within each patient.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .data import IMAGE_SIZE


def confusion_counts(prediction: np.ndarray, ground_truth: np.ndarray) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) in pixels."""
    p, g = prediction.astype(bool), ground_truth.astype(bool)
    return int(np.logical_and(p, g).sum()), int(np.logical_and(p, ~g).sum()), int(np.logical_and(~p, g).sum())


def dice_score(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2 * tp / denominator


def iou_score(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return 1.0 if denominator == 0 else tp / denominator


def per_slice_table(
    predictions: Iterable[np.ndarray],
    ground_truths: Iterable[np.ndarray],
    patient_ids: Iterable[str],
    slice_ids: Iterable[str],
    image_size: int = IMAGE_SIZE,
) -> pd.DataFrame:
    """One row per slice with TP/FP/FN, pixel counts and slice Dice/IoU (evaluated at native resolution)."""
    rows = []
    for prediction, truth, patient_id, slice_id in zip(predictions, ground_truths, patient_ids, slice_ids, strict=True):
        assert prediction.shape == truth.shape == (image_size, image_size), (prediction.shape, truth.shape)
        tp, fp, fn = confusion_counts(prediction, truth)
        rows.append(
            dict(
                patient_id=patient_id,
                slice_id=slice_id,
                tp=tp,
                fp=fp,
                fn=fn,
                ground_truth_pixels=int(truth.sum()),
                predicted_pixels=int(prediction.sum()),
                total_pixels=int(truth.size),
                ground_truth_empty=int(truth.sum() == 0),
                dice=dice_score(tp, fp, fn),
                iou=iou_score(tp, fp, fn),
            )
        )
    return pd.DataFrame(rows)


def per_patient_dice(slices: pd.DataFrame) -> pd.Series:
    """Dice per patient with TP/FP/FN pooled over that patient's slices."""
    return (
        slices.groupby("patient_id")[["tp", "fp", "fn"]]
        .sum()
        .apply(lambda r: dice_score(r.tp, r.fp, r.fn), axis=1)
        .rename("patient_dice")
    )


def summarise(slices: pd.DataFrame) -> dict:
    """All reported metrics for one set of predictions."""
    TP, FP, FN = slices.tp.sum(), slices.fp.sum(), slices.fn.sum()
    positive = slices[slices.ground_truth_empty == 0]
    empty = slices[slices.ground_truth_empty == 1]
    patients = per_patient_dice(slices)
    nan = float("nan")
    return {
        "n_slices": int(len(slices)),
        "pooled_foreground_dice": float(dice_score(TP, FP, FN)),
        "pooled_foreground_iou": float(iou_score(TP, FP, FN)),
        "mean_slice_dice": float(slices.dice.mean()),
        "mean_slice_iou": float(slices.iou.mean()),
        "positive_slice_dice": float(positive.dice.mean()) if len(positive) else nan,
        "positive_slice_iou": float(positive.iou.mean()) if len(positive) else nan,
        "n_positive_slices": int(len(positive)),
        "n_empty_slices": int(len(empty)),
        "empty_slices_with_false_positives": int((empty.fp > 0).sum()),
        "empty_slice_false_positive_rate": float((empty.fp > 0).mean()) if len(empty) else nan,
        "empty_slice_false_positive_pixels_total": int(empty.fp.sum()),
        "empty_slice_false_positive_pixels_mean": float(empty.fp.mean()) if len(empty) else nan,
        "patient_dice_mean": float(patients.mean()),
        "patient_dice_median": float(patients.median()),
        "patient_dice_std": float(patients.std(ddof=1)) if len(patients) > 1 else 0.0,
        "patient_dice_min": float(patients.min()),
        "n_patients": int(len(patients)),
        # Optional context only (background dominates): fraction of all pixels classified correctly.
        "pixel_accuracy": float(1 - (FP + FN) / slices.total_pixels.sum()),
    }


def threshold_sweep(
    probabilities: np.ndarray, ground_truths: np.ndarray, manifest: pd.DataFrame, thresholds: list[float]
) -> dict:
    """{threshold: summary} for every threshold in the grid."""
    return {
        float(t): summarise(per_slice_table(probabilities >= t, ground_truths, manifest.patient_id, manifest.slice_id))
        for t in thresholds
    }


def best_threshold(sweep: dict, metric: str) -> float:
    """Threshold with the highest metric; ties go to the first (lowest) threshold in grid order, as in the study."""
    return max(sweep, key=lambda t: sweep[t][metric])
