"""Step 2: train the 14 study runs (about 12 minutes each on one NVIDIA L40S).

By default runs are trained one after another on one device. With --parallel-gpus, runs are spread over the listed
GPUs (one run per GPU at a time, or more with --runs-per-gpu); each run is a separate process, so results do not depend
on how the runs are scheduled. Finished runs (results/runs/<run>/run_summary.json present) are skipped; an interrupted
run resumes from its last completed epoch.
Smoke-test options shrink the run and mark it as a smoke test so it can never be frozen for the study.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_segmentation.command_line import add_path_options, default_device, resolve_paths  # noqa: E402
from brain_mri_segmentation.config import load_run_config, resolve_run_names, run_directory  # noqa: E402
from brain_mri_segmentation.logging_setup import configure_logging  # noqa: E402
from brain_mri_segmentation.parallel import run_in_parallel, visible_gpu_indices  # noqa: E402
from brain_mri_segmentation.train import RUN_FINISHED_MARKER, train_run  # noqa: E402

LOGGER = logging.getLogger("train_all_runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_options(parser)
    parser.add_argument("--runs", nargs="+", help="subset of run names (default: all 14, in study order)")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parallel = parser.add_argument_group("parallel training across GPUs")
    parallel.add_argument(
        "--parallel-gpus",
        nargs="*",
        type=int,
        metavar="GPU_INDEX",
        help="train runs in parallel on these GPU indices; with no indices, use every visible GPU",
    )
    parallel.add_argument("--runs-per-gpu", type=int, default=1, help="concurrent runs per GPU (default 1)")
    smoke = parser.add_argument_group("smoke test only (marks runs as smoke tests)")
    smoke.add_argument("--smoke-test-epochs", type=int)
    smoke.add_argument("--smoke-test-max-train-slices", type=int, default=0)
    smoke.add_argument("--smoke-test-max-validation-slices", type=int, default=0)
    smoke.add_argument("--smoke-test-batch-size", type=int)
    smoke.add_argument("--smoke-test-data-loader-workers", type=int)
    arguments = parser.parse_args()
    paths = resolve_paths(arguments)
    results_dir = Path(paths["results_dir"])
    configure_logging(results_dir / "runs" / "train_all_runs.log")
    statistics_path = results_dir / "data_preparation" / "train_channel_statistics.json"
    if not statistics_path.exists():
        raise SystemExit("Run scripts/01_prepare_data.py first (training channel statistics are missing).")
    channel_statistics = json.load(open(statistics_path))
    device = arguments.device or default_device()

    overrides = {
        "epochs": arguments.smoke_test_epochs,
        "batch_size": arguments.smoke_test_batch_size,
        "data_loader_workers": arguments.smoke_test_data_loader_workers,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    is_smoke_test = bool(
        overrides or arguments.smoke_test_max_train_slices or arguments.smoke_test_max_validation_slices
    )

    run_names = resolve_run_names(arguments.runs)
    if arguments.parallel_gpus is not None:
        train_in_parallel(arguments, run_names, results_dir)
        return
    for position, run_name in enumerate(run_names, start=1):
        run_dir = run_directory(results_dir, run_name)
        if (run_dir / RUN_FINISHED_MARKER).exists():
            LOGGER.info("[%d/%d] %s already finished; skipping", position, len(run_names), run_name)
            continue
        config = {**load_run_config(run_name), **overrides}
        LOGGER.info(
            "[%d/%d] training %s on %s%s",
            position,
            len(run_names),
            run_name,
            device,
            " (SMOKE TEST)" if is_smoke_test else "",
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        run_log = logging.FileHandler(run_dir / "training.log")
        run_log.setFormatter(logging.getLogger().handlers[0].formatter)
        logging.getLogger().addHandler(run_log)
        try:
            summary = train_run(
                config,
                paths["data_root"],
                run_dir,
                channel_statistics,
                device=device,
                max_train_slices=arguments.smoke_test_max_train_slices,
                max_validation_slices=arguments.smoke_test_max_validation_slices,
                is_smoke_test=is_smoke_test,
            )
        finally:
            logging.getLogger().removeHandler(run_log)
            run_log.close()
        LOGGER.info(
            "%s finished in %.0f s; best epoch %d", run_name, summary["training_seconds"], summary["best_epoch"]
        )


def train_in_parallel(arguments: argparse.Namespace, run_names: list[str], results_dir: Path) -> None:
    """Re-invoke this script once per unfinished run, each pinned to one GPU slot."""
    gpus = arguments.parallel_gpus or visible_gpu_indices()
    if not gpus:
        raise SystemExit("--parallel-gpus given but no GPU is visible")
    pending = [r for r in run_names if not (run_directory(results_dir, r) / RUN_FINISHED_MARKER).exists()]
    LOGGER.info(
        "%d of %d runs to train on GPUs %s (%d per GPU)", len(pending), len(run_names), gpus, arguments.runs_per_gpu
    )
    forwarded = _strip_options(sys.argv[1:], {"--parallel-gpus", "--runs-per-gpu", "--runs", "--device"})

    def command_for_run(run_name: str) -> list[str]:
        return [sys.executable, __file__, *forwarded, "--runs", run_name, "--device", "cuda"]

    def console_log(run_name: str) -> Path:
        run_dir = run_directory(results_dir, run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "console.log"

    failed = run_in_parallel(pending, gpus, arguments.runs_per_gpu, command_for_run, console_log)
    if failed:
        raise SystemExit(f"Failed runs: {failed}. Re-run the same command to resume them.")
    LOGGER.info("All runs finished.")


def _strip_options(arguments: list[str], options_with_values: set[str]) -> list[str]:
    """Remove the given options and their values from an argument list."""
    kept, skipping = [], False
    for argument in arguments:
        if argument.startswith("--"):
            skipping = argument in options_with_values
        if not skipping:
            kept.append(argument)
    return kept


if __name__ == "__main__":
    main()
