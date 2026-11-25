"""Step 3: choose each run's decision threshold on VALIDATION only and score the always-empty reference.

Outputs per run in results/validation/<run>/: selection.json, threshold_sweep.csv, validation_slices.csv,
validation_patient_dice.csv; plus results/validation/always_empty_reference/ and validation_summary_by_run.csv.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from brain_mri_segmentation.command_line import add_path_options, default_device, resolve_paths  # noqa: E402
from brain_mri_segmentation.config import ALWAYS_EMPTY_REFERENCE, resolve_run_names, run_directory  # noqa: E402
from brain_mri_segmentation.logging_setup import configure_logging  # noqa: E402
from brain_mri_segmentation.select import (  # noqa: E402
    load_validation,
    score_always_empty_reference,
    select_run_on_validation,
)

LOGGER = logging.getLogger("select_on_validation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_options(parser)
    parser.add_argument("--runs", nargs="+", help="subset of run names (default: all 14)")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--smoke-test-max-validation-slices", type=int, default=0)
    arguments = parser.parse_args()
    paths = resolve_paths(arguments)
    validation_dir = Path(paths["results_dir"]) / "validation"
    configure_logging(validation_dir / "select_on_validation.log")
    device = arguments.device or default_device()

    validation = load_validation(paths["data_root"], arguments.smoke_test_max_validation_slices)
    rows = [
        {
            "run_name": ALWAYS_EMPTY_REFERENCE,
            "selected_threshold": None,
            **score_always_empty_reference(validation, validation_dir / ALWAYS_EMPTY_REFERENCE),
        }
    ]
    for run_name in resolve_run_names(arguments.runs):
        selection = select_run_on_validation(
            run_directory(paths["results_dir"], run_name),
            paths["data_root"],
            validation_dir / run_name,
            validation,
            device,
        )
        rows.append(
            {
                "run_name": run_name,
                "best_epoch": selection["best_epoch"],
                "selected_threshold": selection["selected_threshold"],
                "selected_threshold_is_grid_edge": selection["selected_threshold_is_grid_edge"],
                **selection["validation_summary_at_selected_threshold"],
            }
        )
    pd.DataFrame(rows).to_csv(validation_dir / "validation_summary_by_run.csv", index=False)
    LOGGER.info("Wrote %s", validation_dir / "validation_summary_by_run.csv")


if __name__ == "__main__":
    main()
