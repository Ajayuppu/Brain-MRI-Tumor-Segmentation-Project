"""Inference: predict binary 256x256 PNG masks for TIFF images with one trained run and its validation threshold.

Example: python scripts/07_predict_masks_for_images.py --run bce_dice_lambda1_seed0 path/to/slice.tif
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_segmentation.command_line import add_path_options, default_device, resolve_paths  # noqa: E402
from brain_mri_segmentation.config import run_directory  # noqa: E402
from brain_mri_segmentation.logging_setup import configure_logging  # noqa: E402
from brain_mri_segmentation.predict import predict_masks  # noqa: E402

LOGGER = logging.getLogger("predict_masks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_options(parser)
    parser.add_argument("--run", required=True, help="trained run name, e.g. bce_dice_lambda1_seed0")
    parser.add_argument("--output-dir", help="default: results/predicted_masks/<run>")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("images", nargs="+", help="TIFF image files (256x256x3)")
    arguments = parser.parse_args()
    paths = resolve_paths(arguments)
    configure_logging()
    results_dir = Path(paths["results_dir"])
    output_dir = Path(arguments.output_dir or results_dir / "predicted_masks" / arguments.run)
    written = predict_masks(
        run_directory(results_dir, arguments.run),
        results_dir / "validation" / arguments.run / "selection.json",
        arguments.images,
        output_dir,
        arguments.device or default_device(),
    )
    for path in written:
        LOGGER.info("wrote %s", path)


if __name__ == "__main__":
    main()
