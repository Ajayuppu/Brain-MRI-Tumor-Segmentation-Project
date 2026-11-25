"""The 14 run configs follow the study protocol; the parallel scheduler uses every GPU slot."""

import sys

import pytest

from brain_mri_segmentation.config import STUDY_RUN_ORDER, condition_name, load_run_config
from brain_mri_segmentation.parallel import run_in_parallel

SHARED_BUDGET = {
    "base_channels": 32,
    "epochs": 50,
    "batch_size": 16,
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "learning_rate_schedule": "cosine",
    "data_loader_workers": 4,
    "drop_last_incomplete_batch": True,
    "selection_metric": "pooled_foreground_dice",
    "soft_dice_smoothing": 1.0,
}
EXPECTED_THRESHOLDS = [round(0.05 * k, 2) for k in range(1, 20)]


def test_fourteen_runs_with_expected_names():
    assert len(STUDY_RUN_ORDER) == len(set(STUDY_RUN_ORDER)) == 14
    conditions = {condition_name(r) for r in STUDY_RUN_ORDER}
    assert conditions == {
        "bce",
        "bce_dice_lambda1",
        "weighted_bce",
        "bce_positive_only_dice",
        "bce_dice_lambda0.5",
        "bce_dice_lambda2",
    }


@pytest.mark.parametrize("run_name", STUDY_RUN_ORDER)
def test_every_run_shares_the_same_budget_and_selection_rule(run_name):
    config = load_run_config(run_name)
    assert {k: config[k] for k in SHARED_BUDGET} == SHARED_BUDGET
    assert config["threshold_grid"] == EXPECTED_THRESHOLDS
    assert config["seed"] == int(run_name.rsplit("_seed", 1)[1])


def test_loss_settings_match_run_names():
    assert load_run_config("bce_dice_lambda0.5_seed0")["dice_weight"] == 0.5
    assert load_run_config("bce_dice_lambda2_seed0")["dice_weight"] == 2.0
    weighted = load_run_config("weighted_bce_seed1")
    assert weighted["positive_class_weight"] == "auto" and weighted["positive_class_weight_cap"] == 100
    assert load_run_config("bce_positive_only_dice_seed2")["loss"] == "bce_positive_only_dice"


def test_parallel_scheduler_runs_everything_and_reports_failures(tmp_path):
    def command(run_name):
        code = "import os,sys; print(os.environ['CUDA_VISIBLE_DEVICES']); sys.exit(1 if 'bad' in sys.argv[1] else 0)"
        return [sys.executable, "-c", code, run_name]

    failed = run_in_parallel(["a", "b", "bad", "c"], [0, 1], 1, command, lambda r: tmp_path / f"{r}.log")
    assert failed == ["bad"]
    assert {(tmp_path / f"{r}.log").read_text().strip() for r in "abc"} <= {"0", "1"}
