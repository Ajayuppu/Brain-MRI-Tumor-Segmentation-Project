"""Freeze and one-time test protocol on synthetic run folders (no real data)."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from brain_mri_segmentation.config import PROJECT_ROOT
from brain_mri_segmentation.evaluate import LEDGER_FILE, LOCK_FILE, run_test_evaluation_once
from brain_mri_segmentation.freeze import CODE_FILES_HASHED, ProtocolError, freeze_method, verify_frozen_method
from brain_mri_segmentation.hashing import sha256_file
from brain_mri_segmentation.model import UNet

RUN = "bce_seed0"


@pytest.fixture
def study(tmp_path):
    """A fake results folder with one finished, validation-selected run and a fake data root."""
    data_root, results = tmp_path / "data", tmp_path / "results"
    data_root.mkdir()
    for split in ("train", "val", "test"):
        pd.DataFrame({"patient_id": [split], "split": [split]}).to_csv(data_root / f"{split}_manifest.csv", index=False)
    run_dir = results / "runs" / RUN
    run_dir.mkdir(parents=True)
    torch.save(UNet(base_channels=4).state_dict(), run_dir / "best.pt")
    (run_dir / "channel_statistics.json").write_text(
        json.dumps({"mean": [0] * 3, "std": [1] * 3, "fitted_on": "train"})
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"run_name": RUN, "base_channels": 4}))
    checkpoint_hash = sha256_file(run_dir / "best.pt")
    (run_dir / "run_summary.json").write_text(
        json.dumps({"is_smoke_test": False, "best_checkpoint_sha256": checkpoint_hash})
    )
    selection = results / "validation" / RUN
    selection.mkdir(parents=True)
    (selection / "selection.json").write_text(
        json.dumps({"selected_threshold": 0.5, "best_checkpoint_sha256": checkpoint_hash})
    )
    frozen = results / "frozen_method" / "frozen_method_manifest.json"
    return {"data_root": data_root, "results": results, "run_dir": run_dir, "frozen": frozen, "test": results / "test"}


def freeze(study, **options):
    return freeze_method([RUN], study["results"], str(study["data_root"]), study["frozen"], **options)


def fake_evaluation(frozen, mark_labels_loaded):
    mark_labels_loaded()
    return {"ok": True}


def test_freeze_records_hashes_and_verifies(study):
    frozen = freeze(study)
    assert frozen["runs"][RUN]["selected_threshold"] == 0.5
    assert set(frozen["code_sha256"]) == set(CODE_FILES_HASHED)
    assert len(frozen["data_manifest_sha256"]) == 3
    verify_frozen_method(study["frozen"])


def test_freezing_is_one_shot(study):
    freeze(study)
    with pytest.raises(ProtocolError, match="one-shot"):
        freeze(study)


def test_freeze_requires_validation_selection(study):
    shutil.rmtree(study["results"] / "validation")
    with pytest.raises(ProtocolError, match="no validation selection"):
        freeze(study)


def test_smoke_test_runs_cannot_be_frozen_for_the_study(study):
    summary = json.loads((study["run_dir"] / "run_summary.json").read_text())
    (study["run_dir"] / "run_summary.json").write_text(json.dumps({**summary, "is_smoke_test": True}))
    with pytest.raises(ProtocolError, match="smoke-test"):
        freeze(study)


def test_statistics_not_fitted_on_train_are_refused(study):
    (study["run_dir"] / "channel_statistics.json").write_text(
        json.dumps({"mean": [0] * 3, "std": [1] * 3, "fitted_on": "val"})
    )
    with pytest.raises(ProtocolError, match="not fitted on the training split"):
        freeze(study)


@pytest.mark.parametrize("tamper", ["checkpoint", "threshold", "manifest_record", "test_manifest"])
def test_any_change_after_freezing_is_refused_before_the_test_set_is_claimed(study, tamper):
    freeze(study)
    if tamper == "checkpoint":
        with open(study["run_dir"] / "best.pt", "ab") as f:
            f.write(b"x")
    elif tamper == "threshold":
        path = study["results"] / "validation" / RUN / "selection.json"
        path.write_text(path.read_text().replace("0.5", "0.4"))
    elif tamper == "manifest_record":
        record = json.loads(study["frozen"].read_text())
        record["runs"][RUN]["selected_threshold"] = 0.1
        study["frozen"].write_text(json.dumps(record))
    else:
        (study["data_root"] / "test_manifest.csv").write_text("patient_id,split\nOTHER,test\n")
    with pytest.raises(ProtocolError, match="Refusing"):
        run_test_evaluation_once(study["test"], study["frozen"], fake_evaluation)
    assert not study["test"].exists()


def test_evaluation_code_change_is_detected(study, tmp_path):
    project_copy = tmp_path / "project_copy"
    for code_file in CODE_FILES_HASHED:
        (project_copy / code_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PROJECT_ROOT / code_file, project_copy / code_file)
    freeze(study, project_root=project_copy)
    with open(project_copy / "src/brain_mri_segmentation/metrics.py", "a") as f:
        f.write("\n# edited after freezing\n")
    with pytest.raises(ProtocolError, match="metrics.py"):
        verify_frozen_method(study["frozen"], project_root=project_copy)


def test_test_evaluation_runs_once_only(study):
    freeze(study)
    assert run_test_evaluation_once(study["test"], study["frozen"], fake_evaluation) == {"ok": True}
    with pytest.raises(ProtocolError, match="already completed"):
        run_test_evaluation_once(study["test"], study["frozen"], fake_evaluation)
    (study["test"] / LOCK_FILE).unlink()  # deleting the lock alone does not reopen the test set
    with pytest.raises(ProtocolError, match="already completed"):
        run_test_evaluation_once(study["test"], study["frozen"], fake_evaluation)


def test_retry_only_with_reason_and_only_before_masks_were_read(study):
    freeze(study)

    def fails_before_masks(frozen, mark_labels_loaded):
        raise RuntimeError("crash before reading masks")

    def fails_after_masks(frozen, mark_labels_loaded):
        mark_labels_loaded()
        raise RuntimeError("crash after reading masks")

    with pytest.raises(RuntimeError):
        run_test_evaluation_once(study["test"], study["frozen"], fails_before_masks)
    with pytest.raises(ProtocolError, match="retry-reason"):
        run_test_evaluation_once(study["test"], study["frozen"], fake_evaluation)
    with pytest.raises(RuntimeError):
        run_test_evaluation_once(study["test"], study["frozen"], fails_after_masks, retry_reason="path fix")
    with pytest.raises(ProtocolError, match="already read test masks"):
        run_test_evaluation_once(study["test"], study["frozen"], fake_evaluation, retry_reason="again")
    attempts = json.loads((study["test"] / LEDGER_FILE).read_text())["attempts"]
    assert [a["status"] for a in attempts] == ["failed", "failed"]
    assert not Path(study["test"] / "test_results.json").exists()
