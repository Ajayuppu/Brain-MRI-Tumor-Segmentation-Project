"""Data audit for the course split (project description sections 3 and 4B).

Training/validation: positive and empty slices, lesion sizes, foreground fraction, per-channel intensities.
Test: patient IDs, pair counts and stored image sizes only (no label statistics, no images read).
Checks: split sizes match the course, patients are disjoint, the split matches the course patient_splits.csv,
no duplicate image hashes across splits, every train/val file exists.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import SPLITS, check_patient_disjoint, load_manifest, load_patient_ids, read_image

TEST_METADATA_COLUMNS = ["patient_id", "split", "image_path", "height", "width", "channels", "image_sha256"]


def label_statistics(manifest: pd.DataFrame, settings: dict) -> dict:
    """Positive/empty counts and lesion-size statistics from manifest metadata (train/val only)."""
    positive = manifest[manifest.has_foreground == 1]
    pixels_per_slice = settings["image_height"] * settings["image_width"]
    per_patient_positive_fraction = manifest.groupby("patient_id").has_foreground.mean()
    return {
        "positive_slices": int(len(positive)),
        "empty_slices": int((manifest.has_foreground == 0).sum()),
        "positive_slice_fraction": float(manifest.has_foreground.mean()),
        "foreground_pixel_fraction": float(manifest.foreground_pixels.sum() / (len(manifest) * pixels_per_slice)),
        "lesion_area_pixels_quantiles": {
            str(q): float(positive.foreground_pixels.quantile(q)) for q in settings["lesion_area_quantiles"]
        },
        "small_lesion_slices": int((positive.foreground_pixels < settings["small_lesion_threshold_pixels"]).sum()),
        "patients_without_positive_slices": int((per_patient_positive_fraction == 0).sum()),
        "per_patient_positive_fraction_min_median_max": [
            float(per_patient_positive_fraction.min()),
            float(per_patient_positive_fraction.median()),
            float(per_patient_positive_fraction.max()),
        ],
    }


def channel_intensity_statistics(data_root: str, manifest: pd.DataFrame, channel_names: list[str]) -> dict:
    """Per-channel mean, standard deviation and fraction of zero pixels over a split's images (streamed)."""
    total = np.zeros(len(channel_names))
    total_squared = np.zeros(len(channel_names))
    zeros = np.zeros(len(channel_names))
    count = 0
    for relative_path in manifest.image_path:
        pixels = read_image(os.path.join(data_root, relative_path)).reshape(-1, len(channel_names)).astype(np.float64)
        total += pixels.sum(0)
        total_squared += (pixels**2).sum(0)
        zeros += (pixels == 0).sum(0)
        count += len(pixels)
    mean = total / count
    std = np.sqrt(np.maximum(total_squared / count - mean**2, 0.0))
    return {
        name: {"mean": float(mean[i]), "std": float(std[i]), "zero_pixel_fraction": float(zeros[i] / count)}
        for i, name in enumerate(channel_names)
    }


def audit_dataset(
    data_root: str, settings: dict, course_patient_split: pd.DataFrame, check_course_sizes: bool = True
) -> dict:
    labelled = {split: load_manifest(data_root, split) for split in ("train", "val")}
    test_metadata = pd.read_csv(os.path.join(data_root, "test_manifest.csv"), usecols=TEST_METADATA_COLUMNS)
    metadata = {**labelled, "test": test_metadata}
    expected_shape = (settings["image_height"], settings["image_width"], settings["image_channels"])
    problems: list[str] = []
    audit: dict = {"splits": {}, "checks": {}}

    for split, manifest in metadata.items():
        sizes = (int(manifest.patient_id.nunique()), int(len(manifest)))
        stored_shapes_ok = bool((manifest[["height", "width", "channels"]] == expected_shape).all(axis=None))
        audit["splits"][split] = {
            "patients": sizes[0],
            "image_mask_pairs": sizes[1],
            "all_rows_belong_to_split": bool((manifest.split == split).all()),
            "all_images_256x256x3": stored_shapes_ok,
        }
        if check_course_sizes and list(sizes) != list(settings["expected_split_sizes"][split]):
            problems.append(f"{split}: {sizes} differs from the course split {settings['expected_split_sizes'][split]}")
        if not stored_shapes_ok:
            problems.append(f"{split}: some stored image sizes are not {expected_shape}")

    patients = {split: load_patient_ids(data_root, split) for split in SPLITS}
    overlap = check_patient_disjoint(patients)
    audit["checks"]["patients_shared_between_splits"] = overlap
    if any(overlap.values()):
        problems.append(f"patients shared between splits: {overlap}")

    course_assignment = dict(zip(course_patient_split.patient_id, course_patient_split.split, strict=True))
    mismatched = [p for split in SPLITS for p in patients[split] if course_assignment.get(p) != split]
    audit["checks"]["patients_not_matching_course_split_file"] = len(mismatched)
    if mismatched:
        problems.append(f"{len(mismatched)} patients are not in the split the course file assigns them")

    all_hashes = pd.concat([m.image_sha256 for m in metadata.values()])
    audit["checks"]["duplicate_image_hashes_across_all_splits"] = int(all_hashes.duplicated().sum())
    missing = [
        p
        for m in labelled.values()
        for p in list(m.image_path) + list(m.mask_path)
        if not os.path.exists(os.path.join(data_root, p))
    ]
    audit["checks"]["missing_train_or_validation_files"] = len(missing)
    if missing:
        problems.append(f"{len(missing)} train/validation files are missing, e.g. {missing[:3]}")

    audit["label_statistics"] = {split: label_statistics(m, settings) for split, m in labelled.items()}
    audit["channel_intensity_statistics"] = {
        split: channel_intensity_statistics(data_root, m, settings["channel_names"]) for split, m in labelled.items()
    }
    audit["problems"] = problems
    return audit
