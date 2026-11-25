"""Step 4: freeze the method (checkpoints, thresholds, normalisation, configs, evaluation code, data manifests).

Run this only when every method decision is final. Freezing is one-shot.
Output: results/frozen_method/frozen_method_manifest.json (+ .sha256)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_segmentation.command_line import add_path_options, resolve_paths  # noqa: E402
from brain_mri_segmentation.config import resolve_run_names  # noqa: E402
from brain_mri_segmentation.freeze import freeze_method  # noqa: E402
from brain_mri_segmentation.logging_setup import configure_logging  # noqa: E402

LOGGER = logging.getLogger("freeze_method")
FROZEN_MANIFEST_NAME = "frozen_method_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_options(parser)
    parser.add_argument("--runs", nargs="+", help="runs to freeze (default: all 14)")
    parser.add_argument("--confirm-method-is-final", action="store_true", required=True)
    parser.add_argument("--allow-smoke-test-runs", action="store_true", help="smoke test only")
    arguments = parser.parse_args()
    paths = resolve_paths(arguments)
    output = Path(paths["results_dir"]) / "frozen_method" / FROZEN_MANIFEST_NAME
    configure_logging(output.parent / "freeze_method.log")
    frozen = freeze_method(
        resolve_run_names(arguments.runs),
        paths["results_dir"],
        paths["data_root"],
        output,
        allow_smoke_test_runs=arguments.allow_smoke_test_runs,
    )
    for run_name, spec in frozen["runs"].items():
        LOGGER.info("frozen %s (threshold %.2f)", run_name, spec["selected_threshold"])
    LOGGER.info("Wrote %s and its .sha256", output)


if __name__ == "__main__":
    main()
