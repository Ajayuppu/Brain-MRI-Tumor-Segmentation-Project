"""Manifest-driven LGG dataset.

* Images are read with tifffile (keeps the stored channel order: pre-contrast, FLAIR, post-contrast), native 256x256.
* Masks are stored as 0/255 and become binary with (mask > 0).
* Per-channel mean/std are fitted on TRAINING images only and applied to every split.
* Training augmentation (horizontal flip p=0.5, rot90 k in {0,1,2,3}) is applied identically to image and mask.
* Augmentation randomness is drawn from a generator keyed by (seed, epoch, slice index), so it is independent of
  DataLoader workers. (The original pilot kept one generator in the dataset object; every worker received a copy of
  it each epoch and repeated the same augmentation stream. That bug was fixed in the study and is kept fixed here.)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
IMAGE_SIZE = 256
COURSE_SPLIT_SIZES = {"train": (77, 2763), "val": (17, 598), "test": (16, 568)}  # (patients, image-mask pairs)


def load_manifest(data_root: str, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(split)
    manifest = pd.read_csv(os.path.join(data_root, f"{split}_manifest.csv"))
    if not (manifest["split"] == split).all():
        raise ValueError(f"{split}_manifest.csv contains rows from another split")
    return manifest.reset_index(drop=True)


def load_patient_ids(data_root: str, split: str) -> set:
    """Patient IDs only (safe for the test split: no labels are read)."""
    return set(pd.read_csv(os.path.join(data_root, f"{split}_manifest.csv"), usecols=["patient_id"]).patient_id)


def check_patient_disjoint(patients_by_split: dict) -> dict:
    """Number of shared patients for every pair of splits; all must be 0."""
    names = list(patients_by_split)
    return {
        f"{a}-{b}": len(patients_by_split[a] & patients_by_split[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }


def read_image(path: str) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"unexpected image shape {image.shape} for {path}")
    return image.astype(np.float32)  # height x width x channel, stored channel order


def read_mask(path: str) -> np.ndarray:
    import tifffile

    mask = tifffile.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def fit_channel_statistics(data_root: str, train_manifest: pd.DataFrame) -> dict:
    """Per-channel mean and standard deviation over all TRAINING pixels."""
    total = np.zeros(3)
    total_squared = np.zeros(3)
    count = 0
    for relative_path in train_manifest["image_path"]:
        pixels = read_image(os.path.join(data_root, relative_path)).reshape(-1, 3).astype(np.float64)
        total += pixels.sum(0)
        total_squared += (pixels**2).sum(0)
        count += pixels.shape[0]
    mean = total / count
    std = np.sqrt(np.maximum(total_squared / count - mean**2, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist(), "fitted_on": "train", "n_images": int(len(train_manifest))}


def save_json(obj: dict, path) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Identical horizontal flip / rot90 for image and mask."""
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    k = int(rng.integers(0, 4))
    image, mask = np.rot90(image, k, axes=(0, 1)), np.rot90(mask, k, axes=(0, 1))
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def augmentation_generator(seed: int, epoch: int, index: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(epoch), int(index)])


def normalise(image: np.ndarray, channel_statistics: dict) -> np.ndarray:
    mean = np.asarray(channel_statistics["mean"], np.float32)
    std = np.asarray(channel_statistics["std"], np.float32)
    return ((image - mean) / std).astype(np.float32)


class SliceDataset:
    """torch-compatible dataset returning (image CHW float32, mask 1HW float32, row index)."""

    def __init__(self, data_root, manifest, channel_statistics, train=False, seed=0):
        self.data_root, self.manifest, self.train = data_root, manifest, train
        self.channel_statistics = channel_statistics
        self.seed, self.epoch = int(seed), 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        image = read_image(os.path.join(self.data_root, row["image_path"]))
        mask = read_mask(os.path.join(self.data_root, row["mask_path"]))
        if self.train:
            image, mask = augment(image, mask, augmentation_generator(self.seed, self.epoch, index))
        image = normalise(image, self.channel_statistics)
        return image.transpose(2, 0, 1).copy(), mask[None].astype(np.float32), index


class ImageOnlyDataset:
    """Images only (no mask file is ever opened). Used for validation/test prediction."""

    def __init__(self, data_root, manifest, channel_statistics):
        self.data_root, self.manifest, self.channel_statistics = data_root, manifest, channel_statistics

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        image = read_image(os.path.join(self.data_root, self.manifest["image_path"].iloc[index]))
        return normalise(image, self.channel_statistics).transpose(2, 0, 1).copy()
