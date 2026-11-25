"""Course split, patient disjointness, aligned augmentation and per-sample augmentation randomness."""

from pathlib import Path

import numpy as np
import pandas as pd

from brain_mri_segmentation.config import DATA_PREPARATION_CONFIG, PROJECT_ROOT, load_yaml
from brain_mri_segmentation.data import COURSE_SPLIT_SIZES, augment, augmentation_generator, check_patient_disjoint

SETTINGS = load_yaml(DATA_PREPARATION_CONFIG)
COURSE_SPLIT = pd.read_csv(PROJECT_ROOT / SETTINGS["course_patient_split_file"])


def test_course_split_file_has_77_17_16_patients_and_expected_pairs():
    for split, (patients, pairs) in SETTINGS["expected_split_sizes"].items():
        rows = COURSE_SPLIT[COURSE_SPLIT.split == split]
        assert len(rows) == patients and rows.pair_count.sum() == pairs
        assert (patients, pairs) == COURSE_SPLIT_SIZES[split]


def test_course_split_is_patient_disjoint():
    assert COURSE_SPLIT.patient_id.is_unique
    patients = {s: set(COURSE_SPLIT.patient_id[COURSE_SPLIT.split == s]) for s in ("train", "val", "test")}
    assert all(count == 0 for count in check_patient_disjoint(patients).values())


def test_disjointness_check_detects_a_shared_patient():
    assert check_patient_disjoint({"train": {"A", "B"}, "val": {"B"}, "test": {"C"}})["train-val"] == 1


def test_augmentation_keeps_image_and_mask_aligned_and_channel_order():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = (np.random.rand(256, 256) > 0.9).astype(np.uint8)
        image = np.repeat(mask[..., None], 3, 2).astype(np.float32) * np.array([1, 2, 3], np.float32)
        augmented_image, augmented_mask = augment(image, mask, rng)
        assert np.array_equal(augmented_image[..., 0], augmented_mask)
        assert np.array_equal(augmented_image[..., 2], 3 * augmented_mask)


def test_augmentation_randomness_differs_per_slice_and_epoch_but_is_reproducible():
    draws = lambda epoch: [tuple(augmentation_generator(3, epoch, i).random(2)) for i in range(8)]  # noqa: E731
    assert len(set(draws(0))) == 8  # every slice gets its own stream (the fixed worker-copy bug)
    assert draws(0) != draws(1)  # streams change between epochs
    assert draws(0) == draws(0)  # and do not depend on which worker process draws them


def test_test_manifest_is_not_shipped_with_the_project():
    # The local data/ folder is git-ignored, so only shipped folders are checked.
    shipped = [path for path in Path(PROJECT_ROOT).rglob("test_manifest.csv")
               if "data" not in path.relative_to(PROJECT_ROOT).parts[:1]]
    assert not shipped
