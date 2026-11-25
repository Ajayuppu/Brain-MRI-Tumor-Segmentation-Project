"""Configuration loading: shared paths (configs/paths.yaml) and one YAML file per training run (configs/runs/)."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG_DIR = PROJECT_ROOT / "configs" / "runs"
PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"
DATA_PREPARATION_CONFIG = PROJECT_ROOT / "configs" / "data_preparation.yaml"
ANALYSIS_CONFIG = PROJECT_ROOT / "configs" / "analysis.yaml"

# The 14 runs of the study, in the order they are trained.
STUDY_RUN_ORDER = [
    "bce_seed0",
    "bce_dice_lambda1_seed0",
    "bce_seed1",
    "bce_dice_lambda1_seed1",
    "bce_seed2",
    "bce_dice_lambda1_seed2",
    "weighted_bce_seed0",
    "bce_positive_only_dice_seed0",
    "weighted_bce_seed1",
    "bce_positive_only_dice_seed1",
    "weighted_bce_seed2",
    "bce_positive_only_dice_seed2",
    "bce_dice_lambda0.5_seed0",
    "bce_dice_lambda2_seed0",
]

# Name used for the trivial always-empty-mask reference in every results table.
ALWAYS_EMPTY_REFERENCE = "always_empty_reference"

REQUIRED_RUN_KEYS = [
    "run_name",
    "study_group",
    "loss",
    "dice_weight",
    "soft_dice_smoothing",
    "positive_class_weight",
    "positive_class_weight_cap",
    "seed",
    "base_channels",
    "epochs",
    "batch_size",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "learning_rate_schedule",
    "data_loader_workers",
    "drop_last_incomplete_batch",
    "selection_metric",
    "threshold_grid",
]
KNOWN_LOSSES = ("bce", "bce_dice", "bce_positive_only_dice", "weighted_bce")


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def condition_name(run_name: str) -> str:
    """'bce_dice_lambda1_seed2' -> 'bce_dice_lambda1' (the loss condition without its seed)."""
    return run_name.rsplit("_seed", 1)[0]


def load_paths(
    paths_file: str | Path | None = None, data_root: str | None = None, results_dir: str | None = None
) -> dict:
    """Read configs/paths.yaml; command-line values override it. Relative results_dir resolves from the project root."""
    paths = load_yaml(paths_file or PATHS_CONFIG)
    if data_root:
        paths["data_root"] = data_root
    if results_dir:
        paths["results_dir"] = results_dir
    results = Path(paths["results_dir"]).expanduser()
    paths["results_dir"] = str(results if results.is_absolute() else (PROJECT_ROOT / results).resolve())
    paths["data_root"] = str(Path(paths["data_root"]).expanduser())
    return paths


def load_run_config(run_name_or_path: str | Path) -> dict:
    """Load and check configs/runs/<run_name>.yaml (or an explicit YAML path)."""
    path = Path(run_name_or_path)
    if path.suffix != ".yaml":
        path = RUN_CONFIG_DIR / f"{run_name_or_path}.yaml"
    config = load_yaml(path)
    missing = [k for k in REQUIRED_RUN_KEYS if k not in config]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if config["run_name"] != path.stem:
        raise ValueError(f"{path}: run_name '{config['run_name']}' must equal the file name")
    if config["loss"] not in KNOWN_LOSSES:
        raise ValueError(f"{path}: unknown loss '{config['loss']}'")
    return config


def resolve_run_names(requested: list[str] | None) -> list[str]:
    """The requested run names, or all 14 study runs in training order."""
    names = requested or STUDY_RUN_ORDER
    for n in names:
        if not (RUN_CONFIG_DIR / f"{n}.yaml").exists():
            raise SystemExit(f"No config file configs/runs/{n}.yaml")
    return names


def run_directory(results_dir: str | Path, run_name: str) -> Path:
    return Path(results_dir) / "runs" / run_name
