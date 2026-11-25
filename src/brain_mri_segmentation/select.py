"""Validation-only selection of the decision threshold for a finished run.

The checkpoint (best epoch) was already chosen during training by the same validation rule. Here the saved best.pt is
re-scored on validation over the threshold grid and the threshold with the highest pooled foreground Dice is chosen
(ties -> lowest threshold). Nothing in this module reads the test split.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .data import ImageOnlyDataset, load_manifest, read_mask, save_json
from .hashing import sha256_file
from .metrics import best_threshold, per_patient_dice, per_slice_table, summarise, threshold_sweep
from .model import UNet
from .train import RUN_FINISHED_MARKER, predict_probabilities

LOGGER = logging.getLogger(__name__)
RESCORE_TOLERANCE = 1e-4  # allowed difference between training-time and re-scored validation Dice


def load_validation(data_root: str, max_validation_slices: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Validation manifest and its binary masks."""
    manifest = load_manifest(data_root, "val")
    if max_validation_slices:
        manifest = manifest.iloc[:max_validation_slices].reset_index(drop=True)
    masks = np.stack([read_mask(os.path.join(data_root, p)) for p in manifest.mask_path])
    return manifest, masks


def load_trained_model(run_dir: Path, base_channels: int, device: str) -> UNet:
    """U-Net with the run's best.pt weights, in evaluation mode."""
    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    return model.eval()


def select_run_on_validation(run_dir: Path, data_root: str, output_dir: Path, validation: tuple, device: str) -> dict:
    """Choose the threshold for one run on validation and write its validation outputs to output_dir."""
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    if not (run_dir / RUN_FINISHED_MARKER).exists():
        raise SystemExit(f"{run_dir.name} has not finished training")
    config = yaml.safe_load(open(run_dir / "config.yaml"))
    run_summary = json.load(open(run_dir / RUN_FINISHED_MARKER))
    channel_statistics = json.load(open(run_dir / "channel_statistics.json"))
    manifest, masks = validation
    metric = config["selection_metric"]

    epoch_history = pd.read_csv(run_dir / "metrics.csv")
    best_epoch_from_history = int(epoch_history[f"validation_{metric}_at_best_threshold"].idxmax())
    if best_epoch_from_history != run_summary["best_epoch"]:
        raise SystemExit(f"{run_dir.name}: metrics.csv and run_summary.json disagree on the best epoch")

    loader = DataLoader(
        ImageOnlyDataset(data_root, manifest, channel_statistics),
        batch_size=config["batch_size"],
        num_workers=config["data_loader_workers"],
    )
    probabilities = predict_probabilities(load_trained_model(run_dir, config["base_channels"], device), loader, device)
    sweep = threshold_sweep(probabilities, masks, manifest, config["threshold_grid"])
    threshold = best_threshold(sweep, metric)
    recorded_score = run_summary[f"best_validation_{metric}_over_threshold_grid"]
    if abs(sweep[threshold][metric] - recorded_score) > RESCORE_TOLERANCE:
        LOGGER.warning(
            "%s: re-scored validation %s %.5f differs from training-time %.5f (GPU non-determinism?)",
            run_dir.name,
            metric,
            sweep[threshold][metric],
            recorded_score,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"threshold": t, **s} for t, s in sweep.items()]).to_csv(
        output_dir / "threshold_sweep.csv", index=False
    )
    slices = per_slice_table(probabilities >= threshold, masks, manifest.patient_id, manifest.slice_id)
    slices.to_csv(output_dir / "validation_slices.csv", index=False)
    per_patient_dice(slices).to_csv(output_dir / "validation_patient_dice.csv")
    selection = {
        "run_name": config["run_name"],
        "best_epoch": run_summary["best_epoch"],
        "selected_threshold": threshold,
        "selection_metric": metric,
        "selected_threshold_is_grid_edge": threshold in (min(config["threshold_grid"]), max(config["threshold_grid"])),
        "validation_summary_at_selected_threshold": sweep[threshold],
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
    }
    save_json(selection, output_dir / "selection.json")
    LOGGER.info(
        "%s: best epoch %d, threshold %.2f, validation %s %.4f, positive-slice Dice %.4f, empty FP %d/%d",
        config["run_name"],
        selection["best_epoch"],
        threshold,
        metric,
        sweep[threshold][metric],
        sweep[threshold]["positive_slice_dice"],
        sweep[threshold]["empty_slices_with_false_positives"],
        sweep[threshold]["n_empty_slices"],
    )
    return selection


def score_always_empty_reference(validation: tuple, output_dir: Path) -> dict:
    """Trivial reference that predicts an empty mask for every slice."""
    manifest, masks = validation
    output_dir.mkdir(parents=True, exist_ok=True)
    slices = per_slice_table(np.zeros_like(masks), masks, manifest.patient_id, manifest.slice_id)
    slices.to_csv(output_dir / "validation_slices.csv", index=False)
    summary = summarise(slices)
    save_json(summary, output_dir / "summary.json")
    return summary
