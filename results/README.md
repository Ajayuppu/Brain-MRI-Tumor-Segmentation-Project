# results/

Filled by the numbered scripts. This copy holds the outputs of the full GPU study (24 Nov 2026); the `best.pt` checkpoints are not in git because of their size, and their SHA-256 hashes are in `frozen_method/frozen_method_manifest.json`.

| Folder | Written by | Contents |
|---|---|---|
| `data_preparation/` | `01_prepare_data.py` | `data_audit.json`, `train_channel_statistics.json` (normalisation fitted on training images only) |
| `runs/<run_name>/` | `02_train_all_runs.py` | `metrics.csv` (one row per epoch), `best.pt` (selected checkpoint), `config.yaml`, `channel_statistics.json`, `run_summary.json` (seed, hardware, parameters, training time), `training.log` |
| `validation/<run_name>/` | `03_select_on_validation.py` | `selection.json` (best epoch, selected threshold), `threshold_sweep.csv`, `validation_slices.csv`, `validation_patient_dice.csv`; plus `always_empty_reference/` and `validation_summary_by_run.csv` |
| `frozen_method/` | `04_freeze_method.py` | `frozen_method_manifest.json` + `.sha256` (hashes of checkpoints, thresholds, normalisation, configs, evaluation code, data manifests) |
| `test/` | `05_evaluate_test_once.py` | `test_results.json`, `test_summary.csv`, `test_patient_dice.csv`, `<run>_test_slices.csv`, `test_evaluation_ledger.json` |
| `tables/`, `figures/` | `06_make_figures_and_tables.py` | data audit, experiment record, validation/test summaries, paired patient analysis, lesion-size subgroups, training curves, threshold curves, trade-off plots, failure overlays |
| `predicted_masks/<run>/` | `07_predict_masks_for_images.py` | binary PNG masks for any TIFF images |
