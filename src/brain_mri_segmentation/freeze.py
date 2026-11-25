"""Freeze the method before the test set is touched.

The frozen-method manifest records SHA-256 hashes of, per run: best.pt, channel_statistics.json, config.yaml,
run_summary.json and the validation selection.json (selected threshold); plus the evaluation code and the
train/val/test manifest CSVs (the test manifest is hashed as a file; its labels are not read). A sidecar .sha256
file protects the manifest itself. Freezing is one-shot. verify_frozen_method() re-checks every hash and lists
all mismatches.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import PROJECT_ROOT
from .data import SPLITS, save_json
from .hashing import sha256_file, timestamp


class ProtocolError(SystemExit):
    """Raised when the freeze or one-time test protocol would be violated."""


RUN_FILES_HASHED = ["best.pt", "channel_statistics.json", "config.yaml", "run_summary.json"]
CODE_FILES_HASHED = [
    "src/brain_mri_segmentation/data.py",
    "src/brain_mri_segmentation/model.py",
    "src/brain_mri_segmentation/metrics.py",
    "src/brain_mri_segmentation/hashing.py",
    "src/brain_mri_segmentation/freeze.py",
    "src/brain_mri_segmentation/evaluate.py",
    "scripts/05_evaluate_test_once.py",
]


def _sidecar(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def freeze_method(
    run_names: list[str],
    results_dir: str | Path,
    data_root: str,
    output_path: str | Path,
    project_root: Path = PROJECT_ROOT,
    allow_smoke_test_runs: bool = False,
) -> dict:
    """Write the one-shot frozen-method manifest for the given finished and validation-selected runs."""
    results_dir, output_path = Path(results_dir), Path(output_path)
    if output_path.exists():
        raise ProtocolError(f"{output_path} exists; freezing is one-shot.")
    frozen = {
        "frozen_at": timestamp(),
        "data_root": os.path.abspath(data_root),
        "allow_smoke_test_runs": allow_smoke_test_runs,
        "code_sha256": {},
        "data_manifest_sha256": {},
        "runs": {},
    }
    for code_file in CODE_FILES_HASHED:
        frozen["code_sha256"][code_file] = sha256_file(project_root / code_file)
    for split in SPLITS:
        manifest_path = Path(data_root) / f"{split}_manifest.csv"
        frozen["data_manifest_sha256"][str(manifest_path.resolve())] = sha256_file(manifest_path)
    for run_name in run_names:
        run_dir = results_dir / "runs" / run_name
        selection_path = results_dir / "validation" / run_name / "selection.json"
        if not selection_path.exists():
            raise ProtocolError(f"{run_name}: no validation selection; run scripts/03_select_on_validation.py first")
        run_summary = json.load(open(run_dir / "run_summary.json"))
        if run_summary["is_smoke_test"] and not allow_smoke_test_runs:
            raise ProtocolError(f"{run_name} is a smoke-test run; it cannot be frozen for the study")
        statistics = json.load(open(run_dir / "channel_statistics.json"))
        if statistics.get("fitted_on") != "train":
            raise ProtocolError(f"{run_name}: normalisation statistics were not fitted on the training split")
        file_hashes = {name: sha256_file(run_dir / name) for name in RUN_FILES_HASHED}
        if file_hashes["best.pt"] != run_summary["best_checkpoint_sha256"]:
            raise ProtocolError(f"{run_name}: best.pt differs from the checkpoint recorded at the end of training")
        selection = json.load(open(selection_path))
        if selection["best_checkpoint_sha256"] != file_hashes["best.pt"]:
            raise ProtocolError(f"{run_name}: best.pt changed after validation selection")
        frozen["runs"][run_name] = {
            "run_dir": str(run_dir.resolve()),
            "selection_file": str(selection_path.resolve()),
            "file_sha256": file_hashes,
            "selection_sha256": sha256_file(selection_path),
            "selected_threshold": selection["selected_threshold"],
            "normalisation": statistics,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(frozen, output_path)
    _sidecar(output_path).write_text(sha256_file(output_path) + "\n")
    return frozen


def verify_frozen_method(frozen_path: str | Path, project_root: Path = PROJECT_ROOT) -> dict:
    """Return the frozen manifest if every recorded hash still matches; otherwise raise listing all mismatches."""
    frozen_path = Path(frozen_path)
    if not frozen_path.exists() or not _sidecar(frozen_path).exists():
        raise ProtocolError("Frozen-method manifest or its .sha256 file is missing; run scripts/04_freeze_method.py")
    problems = []
    if _sidecar(frozen_path).read_text().strip() != sha256_file(frozen_path):
        problems.append("frozen-method manifest was modified after freezing")
    frozen = json.load(open(frozen_path))
    for code_file, digest in frozen["code_sha256"].items():
        path = project_root / code_file
        if not path.exists() or sha256_file(path) != digest:
            problems.append(f"code changed since freezing: {code_file}")
    for manifest_path, digest in frozen["data_manifest_sha256"].items():
        if not os.path.exists(manifest_path) or sha256_file(manifest_path) != digest:
            problems.append(f"data manifest changed since freezing: {manifest_path}")
    for run_name, spec in frozen["runs"].items():
        for name, digest in spec["file_sha256"].items():
            path = Path(spec["run_dir"]) / name
            if not path.exists() or sha256_file(path) != digest:
                problems.append(f"run file changed: {run_name}/{name}")
        if (
            not os.path.exists(spec["selection_file"])
            or sha256_file(spec["selection_file"]) != spec["selection_sha256"]
        ):
            problems.append(f"validation selection changed: {run_name}")
    if problems:
        raise ProtocolError("Refusing test evaluation:\n  " + "\n  ".join(problems))
    return frozen
