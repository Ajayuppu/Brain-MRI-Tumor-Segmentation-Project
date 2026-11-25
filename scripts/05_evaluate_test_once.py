"""Step 5: evaluate the FROZEN method on the test split exactly once.

Refuses to run if anything recorded in the frozen-method manifest changed, and refuses to run again after a completed
evaluation. Outputs in results/test/: test_results.json, test_summary.csv, test_patient_dice.csv,
<run>_test_slices.csv, test_evaluation_ledger.json.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_segmentation.command_line import add_path_options, default_device, resolve_paths  # noqa: E402
from brain_mri_segmentation.evaluate import evaluate_frozen_runs_on_test, run_test_evaluation_once  # noqa: E402
from brain_mri_segmentation.logging_setup import configure_logging  # noqa: E402

LOGGER = logging.getLogger("evaluate_test_once")
FROZEN_MANIFEST_NAME = "frozen_method_manifest.json"
REPORTED_METRICS = [
    "pooled_foreground_dice",
    "positive_slice_dice",
    "empty_slices_with_false_positives",
    "n_empty_slices",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_options(parser)
    parser.add_argument("--confirm-method-is-frozen", action="store_true", required=True)
    parser.add_argument("--retry-reason", help="only after a logged failure that happened before test masks were read")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    arguments = parser.parse_args()
    paths = resolve_paths(arguments)
    results_dir = Path(paths["results_dir"])
    output_dir = results_dir / "test"
    frozen_path = results_dir / "frozen_method" / FROZEN_MANIFEST_NAME
    configure_logging()
    device = arguments.device or default_device()
    results = run_test_evaluation_once(
        output_dir,
        frozen_path,
        lambda frozen, mark_labels_loaded: evaluate_frozen_runs_on_test(frozen, mark_labels_loaded, output_dir, device),
        arguments.retry_reason,
    )
    for name, summary in results.items():
        LOGGER.info(
            "%-30s %s",
            name,
            "  ".join(
                f"{m}={summary[m]:.4f}" if isinstance(summary[m], float) else f"{m}={summary[m]}"
                for m in REPORTED_METRICS
            ),
        )
    LOGGER.info("Test results written to %s", output_dir)


if __name__ == "__main__":
    main()
