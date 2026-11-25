"""One-time test evaluation of the frozen method.

Order of operations:
1. verify_frozen_method() re-checks every hash; any mismatch -> refuse before anything is claimed.
2. The test set is claimed atomically: an O_EXCL lock file and an append-only ledger are written BEFORE any test file
   is read. A completed evaluation can never run again. A failed attempt may be retried only with a written reason and
   only if it failed before any test mask was read; every attempt stays in the ledger.
3. Pass 1 predicts from test IMAGES only (no mask file is opened). The ledger then records test_labels_loaded=true.
4. Pass 2 reads the test masks and scores every frozen run plus the always-empty reference.

Limits: this cannot stop someone editing the code, deleting results/test/, or scoring test predictions another way.
"""

from __future__ import annotations

import json
import os
import socket
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .config import ALWAYS_EMPTY_REFERENCE
from .data import ImageOnlyDataset, load_manifest, read_mask, save_json
from .freeze import ProtocolError, verify_frozen_method
from .hashing import sha256_file, timestamp
from .metrics import per_patient_dice, per_slice_table, summarise
from .model import UNet
from .train import predict_probabilities

LEDGER_FILE = "test_evaluation_ledger.json"
LOCK_FILE = "test_evaluation.lock"
RESULTS_FILE = "test_results.json"
TRACEBACK_CHARACTERS_KEPT = 2000


def _write_ledger(path: Path, ledger: dict) -> None:
    save_json(ledger, path)


def claim_test_set(output_dir: Path, frozen_path: Path, retry_reason: str | None = None) -> dict:
    """Atomically claim the single test evaluation, or raise ProtocolError explaining why not."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path, ledger_path = output_dir / LOCK_FILE, output_dir / LEDGER_FILE
    frozen_sha256 = sha256_file(frozen_path)
    if ledger_path.exists():
        ledger = json.load(open(ledger_path))
        last = ledger["attempts"][-1]
        if last["status"] == "completed":
            raise ProtocolError("The test evaluation already completed; refusing to run it again.")
        if last["status"] == "started":
            raise ProtocolError(
                "A test evaluation is marked in progress (or crashed hard). Inspect the ledger; "
                "do not re-run silently."
            )
        if not retry_reason:
            raise ProtocolError("The previous attempt failed; a retry needs --retry-reason and is logged.")
        if last.get("test_labels_loaded"):
            raise ProtocolError("The failed attempt had already read test masks; retry refused. Report it.")
        if ledger["frozen_manifest_sha256"] != frozen_sha256:
            raise ProtocolError("The frozen-method manifest differs from the one used in the failed attempt.")
        lock_path.unlink(missing_ok=True)
    else:
        if retry_reason:
            raise ProtocolError("--retry-reason given but there is no failed attempt.")
        ledger = {"frozen_manifest_sha256": frozen_sha256, "attempts": []}
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ProtocolError(f"{LOCK_FILE} exists; the test set was already claimed.") from error
    os.write(descriptor, f"{timestamp()} pid={os.getpid()} host={socket.gethostname()}\n".encode())
    os.close(descriptor)
    ledger["attempts"].append(
        {"started_at": timestamp(), "status": "started", "test_labels_loaded": False, "retry_reason": retry_reason}
    )
    _write_ledger(ledger_path, ledger)
    return ledger


def update_last_attempt(output_dir: Path, **fields) -> None:
    ledger_path = output_dir / LEDGER_FILE
    ledger = json.load(open(ledger_path))
    ledger["attempts"][-1].update(fields)
    _write_ledger(ledger_path, ledger)


def run_test_evaluation_once(
    output_dir: str | Path,
    frozen_path: str | Path,
    evaluation: Callable[[dict, Callable[[], None]], dict],
    retry_reason: str | None = None,
) -> dict:
    """verify -> claim -> evaluation(frozen, mark_labels_loaded) -> record the outcome in the ledger."""
    output_dir, frozen_path = Path(output_dir), Path(frozen_path)
    frozen = verify_frozen_method(frozen_path)
    claim_test_set(output_dir, frozen_path, retry_reason)
    try:
        results = evaluation(frozen, lambda: update_last_attempt(output_dir, test_labels_loaded=True))
        save_json(results, output_dir / RESULTS_FILE)
        update_last_attempt(
            output_dir,
            status="completed",
            finished_at=timestamp(),
            results_sha256=sha256_file(output_dir / RESULTS_FILE),
        )
        return results
    except BaseException as error:
        update_last_attempt(
            output_dir,
            status="failed",
            finished_at=timestamp(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc()[-TRACEBACK_CHARACTERS_KEPT:],
        )
        raise


def evaluate_frozen_runs_on_test(
    frozen: dict, mark_labels_loaded: Callable[[], None], output_dir: Path, device: str
) -> dict:
    """Two-pass scoring of every frozen run and the always-empty reference on the test split."""
    data_root = frozen["data_root"]
    test_manifest_path = str((Path(data_root) / "test_manifest.csv").resolve())
    if test_manifest_path not in frozen["data_manifest_sha256"]:
        raise ProtocolError("The test manifest was not hashed at freeze time; refusing.")
    manifest = load_manifest(data_root, "test")

    probabilities_by_run = {}
    for run_name, spec in frozen["runs"].items():  # pass 1: images only
        config = yaml.safe_load(open(Path(spec["run_dir"]) / "config.yaml"))
        model = UNet(base_channels=config["base_channels"]).to(device)
        model.load_state_dict(torch.load(Path(spec["run_dir"]) / "best.pt", map_location=device))
        loader = DataLoader(
            ImageOnlyDataset(data_root, manifest, spec["normalisation"]),
            batch_size=config["batch_size"],
            num_workers=config["data_loader_workers"],
        )
        probabilities_by_run[run_name] = predict_probabilities(model, loader, device)

    mark_labels_loaded()  # recorded before the first mask is read
    masks = np.stack([read_mask(os.path.join(data_root, p)) for p in manifest.mask_path])  # pass 2
    results = {}
    patient_dice = {}
    predictions = {ALWAYS_EMPTY_REFERENCE: (np.zeros_like(masks), None)}
    predictions.update(
        {
            run: (probabilities_by_run[run] >= spec["selected_threshold"], spec["selected_threshold"])
            for run, spec in frozen["runs"].items()
        }
    )
    for name, (binary_predictions, threshold) in predictions.items():
        slices = per_slice_table(binary_predictions, masks, manifest.patient_id, manifest.slice_id)
        slices.to_csv(output_dir / f"{name}_test_slices.csv", index=False)
        results[name] = {"threshold": threshold, **summarise(slices)}
        patient_dice[name] = per_patient_dice(slices)
    pd.DataFrame(patient_dice).to_csv(output_dir / "test_patient_dice.csv")
    pd.DataFrame([{"run_name": name, **summary} for name, summary in results.items()]).to_csv(
        output_dir / "test_summary.csv", index=False
    )
    return results
