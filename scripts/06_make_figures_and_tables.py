"""Step 6: tables (results/tables/) and figures (results/figures/) from the saved validation and test results."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_segmentation.command_line import add_path_options, resolve_paths  # noqa: E402
from brain_mri_segmentation.config import resolve_run_names  # noqa: E402
from brain_mri_segmentation.figures import make_all_figures_and_tables  # noqa: E402
from brain_mri_segmentation.logging_setup import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_options(parser)
    parser.add_argument("--runs", nargs="+", help="subset of run names (default: all 14)")
    arguments = parser.parse_args()
    paths = resolve_paths(arguments)
    configure_logging()
    make_all_figures_and_tables(paths["results_dir"], paths["data_root"], resolve_run_names(arguments.runs))


if __name__ == "__main__":
    main()
