"""Create a tiny synthetic dataset with the same folder layout and manifest columns as the course dataset.

Used only by the smoke test (no real MRI data). Each slice is a noisy 256x256x3 image; about half of the slices
contain a bright disc whose mask is stored as 0/255, like the course masks.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

IMAGE_SIZE = 256
PATIENTS_PER_SPLIT = {"train": 4, "val": 2, "test": 2}
SLICES_PER_PATIENT = 6
DISC_RADIUS_RANGE = (12, 40)
BACKGROUND_INTENSITY = 60
NOISE_STANDARD_DEVIATION = 15
DISC_INTENSITY_BOOST = (90, 60, 40)


def write_slice(folder: Path, patient_id: str, index: int, rng: np.random.Generator) -> dict:
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    if index % 2 == 0:
        radius = rng.integers(*DISC_RADIUS_RANGE)
        centre = rng.integers(radius + 1, IMAGE_SIZE - radius - 1, size=2)
        mask[(yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius**2] = 255
    image = rng.normal(BACKGROUND_INTENSITY, NOISE_STANDARD_DEVIATION, (IMAGE_SIZE, IMAGE_SIZE, 3))
    image += (mask[..., None] > 0) * np.array(DISC_INTENSITY_BOOST)
    image = np.clip(image, 0, 255).astype(np.uint8)
    slice_id = f"20000101_{index + 1}"
    image_rel = f"{folder.name}/{patient_id}/images/{patient_id}_{slice_id}.tif"
    mask_rel = f"{folder.name}/{patient_id}/masks/{patient_id}_{slice_id}_mask.tif"
    for relative, array in ((image_rel, image), (mask_rel, mask)):
        (folder.parent / relative).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(folder.parent / relative, array)
    return {
        "patient_id": patient_id,
        "slice_id": slice_id,
        "split": folder.name,
        "image_path": image_rel,
        "mask_path": mask_rel,
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE,
        "channels": 3,
        "has_foreground": int(mask.any()),
        "foreground_pixels": int((mask > 0).sum()),
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    root = Path(arguments.output_dir)
    rng = np.random.default_rng(arguments.seed)
    split_rows = []
    for split, n_patients in PATIENTS_PER_SPLIT.items():
        rows = []
        for p in range(n_patients):
            patient_id = f"SYNTHETIC_{split.upper()}_{p:02d}"
            rows += [write_slice(root / split, patient_id, i, rng) for i in range(SLICES_PER_PATIENT)]
            split_rows.append({"patient_id": patient_id, "split": split, "pair_count": SLICES_PER_PATIENT})
        pd.DataFrame(rows).to_csv(root / f"{split}_manifest.csv", index=False)
    pd.DataFrame(split_rows).to_csv(root / "patient_splits.csv", index=False)
    print(f"synthetic dataset written to {root}")


if __name__ == "__main__":
    main()
