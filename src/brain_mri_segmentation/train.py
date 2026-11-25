"""Train one run. Validation is used for checkpoint (epoch) selection only; the test split is never read.

Per epoch: train on all training slices (shuffled, drop_last), step the cosine schedule, predict validation
probabilities, compute pooled foreground Dice for every threshold in the grid, and keep the checkpoint (best.pt)
of the epoch whose best-over-grid validation pooled Dice is highest (strictly greater -> earliest epoch wins ties).

The run is resumable: after every epoch the full state (model, optimiser, scheduler, shuffling generator, RNG states,
best score, history) is saved to last_epoch_state.pt; an interrupted run continues from the next epoch.
On completion, run_summary.json is written (this marks the run as finished) and last_epoch_state.pt is deleted.
"""

from __future__ import annotations

import logging
import os
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .data import ImageOnlyDataset, SliceDataset, load_manifest, read_mask, save_json
from .hashing import sha256_file, timestamp
from .losses import build_loss, positive_class_weight_from_training
from .metrics import best_threshold, per_slice_table, summarise, threshold_sweep
from .model import UNet, count_trainable_parameters

LOGGER = logging.getLogger(__name__)
RUN_FINISHED_MARKER = "run_summary.json"
RESUME_STATE = "last_epoch_state.pt"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_probabilities(model: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Sigmoid probabilities (N x H x W) for an image-only loader."""
    model.eval()
    batches = []
    for images in loader:
        batches.append(torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0])
    return np.concatenate(batches)


def environment_record() -> dict:
    """Software and GPU versions recorded with every run."""
    gpu = None
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "numpy": np.__version__,
        "nvidia_smi": gpu,
    }


def _atomic_torch_save(obj: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _capture_rng_states(shuffle_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "shuffle_generator": shuffle_generator.get_state(),
    }


def _restore_rng_states(states: dict, shuffle_generator: torch.Generator) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if states["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])
    shuffle_generator.set_state(states["shuffle_generator"])


def train_run(
    config: dict,
    data_root: str,
    run_dir: Path,
    channel_statistics: dict,
    *,
    device: str | None = None,
    max_train_slices: int = 0,
    max_validation_slices: int = 0,
    is_smoke_test: bool = False,
) -> dict:
    """Train one configured run into run_dir and return its run summary."""
    run_dir = Path(run_dir)
    if (run_dir / RUN_FINISHED_MARKER).exists():
        raise SystemExit(f"{run_dir} is already finished; refusing to overwrite it.")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config["seed"])
    seed_everything(seed)

    train_manifest, validation_manifest = load_manifest(data_root, "train"), load_manifest(data_root, "val")
    if max_train_slices:
        train_manifest = train_manifest.iloc[:max_train_slices].reset_index(drop=True)
    if max_validation_slices:
        validation_manifest = validation_manifest.iloc[:max_validation_slices].reset_index(drop=True)
    if channel_statistics.get("fitted_on") != "train":
        raise SystemExit("channel statistics must be fitted on the training split")

    positive_class_weight = None
    if config["loss"] == "weighted_bce":
        positive_class_weight = config["positive_class_weight"]
        if positive_class_weight in (None, "auto"):
            positive_class_weight = positive_class_weight_from_training(
                train_manifest, config["positive_class_weight_cap"]
            )

    workers = int(config["data_loader_workers"])
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)
    train_loader = DataLoader(
        SliceDataset(data_root, train_manifest, channel_statistics, train=True, seed=seed),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=workers,
        generator=shuffle_generator,
        drop_last=bool(config["drop_last_incomplete_batch"]),
    )
    validation_loader = DataLoader(
        ImageOnlyDataset(data_root, validation_manifest, channel_statistics),
        batch_size=config["batch_size"],
        num_workers=workers,
    )
    if len(train_loader) == 0:
        raise SystemExit("no training batches (batch_size larger than the number of training slices)")
    validation_masks = np.stack([read_mask(os.path.join(data_root, p)) for p in validation_manifest.mask_path])

    model = UNet(base_channels=config["base_channels"]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config["epochs"])
    loss_function = build_loss(
        config["loss"], config["dice_weight"], positive_class_weight, config["soft_dice_smoothing"]
    )
    metric = config["selection_metric"]
    grid = config["threshold_grid"]

    start_epoch, best_score, best_epoch, history, seconds_before_resume, resumed_from = 0, -1.0, None, [], 0.0, []
    if (run_dir / RESUME_STATE).exists():
        state = torch.load(run_dir / RESUME_STATE, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng_states(state["rng_states"], shuffle_generator)
        start_epoch, best_score, best_epoch = state["next_epoch"], state["best_score"], state["best_epoch"]
        history, seconds_before_resume = state["history"], state["elapsed_seconds"]
        resumed_from = state.get("resumed_from_epochs", []) + [start_epoch]
        LOGGER.info(
            "[%s] resuming at epoch %d (best so far %.4f at epoch %s)",
            config["run_name"],
            start_epoch,
            best_score,
            best_epoch,
        )

    started = time.time()
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_loader.dataset.set_epoch(epoch)
        loss_sum, seen = 0.0, 0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimiser.zero_grad()
            loss = loss_function(model(images), masks)
            loss.backward()
            optimiser.step()
            loss_sum += loss.item() * len(images)
            seen += len(images)
        learning_rate = optimiser.param_groups[0]["lr"]
        scheduler.step()

        probabilities = predict_probabilities(model, validation_loader, device)
        sweep = threshold_sweep(probabilities, validation_masks, validation_manifest, grid)
        epoch_threshold = best_threshold(sweep, metric)
        at_half = (
            sweep[0.5]
            if 0.5 in sweep
            else summarise(
                per_slice_table(
                    probabilities >= 0.5, validation_masks, validation_manifest.patient_id, validation_manifest.slice_id
                )
            )
        )
        score = sweep[epoch_threshold][metric]
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            _atomic_torch_save(model.state_dict(), run_dir / "best.pt")
        elapsed = seconds_before_resume + time.time() - started
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": loss_sum / seen,
                f"validation_{metric}_at_best_threshold": score,
                "validation_best_threshold_this_epoch": epoch_threshold,
                f"validation_{metric}_at_0.5": at_half[metric],
                "validation_positive_slice_dice_at_0.5": at_half["positive_slice_dice"],
                "validation_empty_slices_with_false_positives_at_0.5": at_half["empty_slices_with_false_positives"],
                "is_new_best_epoch": int(improved),
                "elapsed_seconds": round(elapsed, 1),
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_states": _capture_rng_states(shuffle_generator),
                "next_epoch": epoch + 1,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
                "elapsed_seconds": elapsed,
                "resumed_from_epochs": resumed_from,
            },
            run_dir / RESUME_STATE,
        )
        LOGGER.info(
            f"[{config['run_name']}] epoch {epoch + 1}/{config['epochs']} loss {loss_sum / seen:.4f} "
            f"val {metric} {score:.4f} @thr {epoch_threshold} (0.5: {at_half[metric]:.4f}, "
            f"empty FP {at_half['empty_slices_with_false_positives']}){'  *best*' if improved else ''}"
        )

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    save_json(channel_statistics, run_dir / "channel_statistics.json")
    summary = {
        "run_name": config["run_name"],
        "finished_at": timestamp(),
        "is_smoke_test": bool(is_smoke_test),
        "best_epoch": best_epoch,
        f"best_validation_{metric}_over_threshold_grid": best_score,
        "epochs_run": len(history),
        "positive_class_weight_used": positive_class_weight,
        "trainable_parameters": count_trainable_parameters(model),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "training_seconds": round(history[-1]["elapsed_seconds"], 1),
        "resumed_at_epochs": resumed_from,
        "n_train_slices": len(train_manifest),
        "n_validation_slices": len(validation_manifest),
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "selection_rule": "checkpoint = epoch with the highest best-over-grid validation pooled foreground Dice; "
        "threshold is re-selected on that checkpoint by scripts/03_select_on_validation.py",
        "environment": environment_record(),
    }
    save_json(summary, run_dir / RUN_FINISHED_MARKER)
    (run_dir / RESUME_STATE).unlink(missing_ok=True)
    return summary
