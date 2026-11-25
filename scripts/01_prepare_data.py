"""Step 1: audit the course dataset split and fit normalisation statistics on the training split only.

Outputs: results/data_preparation/data_audit.json and results/data_preparation/train_channel_statistics.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from brain_mri_segmentation.command_line import add_path_options, resolve_paths  # noqa: E402
from brain_mri_segmentation.config import DATA_PREPARATION_CONFIG, PROJECT_ROOT, load_yaml  # noqa: E402
from brain_mri_segmentation.data import fit_channel_statistics, load_manifest, save_json  # noqa: E402
from brain_mri_segmentation.data_audit import audit_dataset  # noqa: E402
from brain_mri_segmentation.logging_setup import configure_logging  # noqa: E402

LOGGER = logging.getLogger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_options(parser)
    parser.add_argument(
        "--skip-course-size-check",
        action="store_true",
        help="only for the synthetic smoke-test dataset (sizes differ from 77/17/16)",
    )
    parser.add_argument("--course-patient-split-file", help="override the course patient_splits.csv (smoke test)")
    arguments = parser.parse_args()
    paths = resolve_paths(arguments)
    output_dir = Path(paths["results_dir"]) / "data_preparation"
    configure_logging(output_dir / "prepare_data.log")
    settings = load_yaml(DATA_PREPARATION_CONFIG)

    split_file = arguments.course_patient_split_file or PROJECT_ROOT / settings["course_patient_split_file"]
    LOGGER.info("Auditing %s against the course split file %s", paths["data_root"], split_file)
    audit = audit_dataset(
        paths["data_root"], settings, pd.read_csv(split_file), check_course_sizes=not arguments.skip_course_size_check
    )
    save_json(audit, output_dir / "data_audit.json")
    LOGGER.info("Splits: %s", json.dumps(audit["splits"]))
    LOGGER.info("Checks: %s", json.dumps(audit["checks"]))
    if audit["problems"]:
        for problem in audit["problems"]:
            LOGGER.error(problem)
        raise SystemExit("Data audit failed; see results/data_preparation/data_audit.json")

    statistics_path = output_dir / "train_channel_statistics.json"
    if statistics_path.exists():
        LOGGER.info("Keeping existing %s (delete it to refit)", statistics_path)
    else:
        LOGGER.info("Fitting per-channel normalisation on the training split only")
        save_json(
            fit_channel_statistics(paths["data_root"], load_manifest(paths["data_root"], "train")), statistics_path
        )
    LOGGER.info("Training channel statistics: %s", json.dumps(json.load(open(statistics_path))))


if __name__ == "__main__":
    main()
