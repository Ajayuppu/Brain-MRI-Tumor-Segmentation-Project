# Decisions and claims linked to evidence

Project description, section 7 (4): "a compact table linking key decisions and claims to experiments or code".
The "Evidence" column names the saved result file that supports each row (full GPU study, 24 Nov 2026).

| Decision or claim | Why | Code | Evidence |
|---|---|---|---|
| Use the course patient split unchanged (77/17/16 patients) | required; slices of one patient are correlated | `data_audit.py`, `scripts/01_prepare_data.py` | `results/data_preparation/data_audit.json`, `results/tables/data_audit_splits_and_labels.md` |
| Normalisation fitted on training images only | no leakage from validation/test | `data.py: fit_channel_statistics` | `results/data_preparation/train_channel_statistics.json` (`fitted_on: train`) |
| Evaluate at native 256×256, masks binarised with > 0 | required metric protocol | `data.py: read_mask`, `metrics.py: per_slice_table` (asserts 256×256) | `tests/test_metrics.py`; `results/data_preparation/data_audit.json` (`all_images_256x256x3: true`) |
| Augmentation identical for image and mask, randomness per slice and epoch | keeps pairs aligned; fixes the pilot's repeated-stream bug | `data.py: augment, augmentation_generator` | `tests/test_data.py` |
| 2D U-Net from scratch, 7,763,041 parameters | credible standard baseline, no pretrained weights | `model.py` | `tests/test_model.py`, `results/tables/experiment_record.md` (`trainable_parameters` 7763041) |
| Always-empty reference | shows that empty slices alone give high slice-averaged scores | `select.py`, `evaluate.py` | `results/validation/always_empty_reference/summary.json` (mean slice Dice 0.659 but pooled Dice 0), `results/test/test_results.json` |
| Proposed: BCE + 1.0 × batch-pooled soft Dice | Dice targets overlap directly under class imbalance | `losses.py` | `results/tables/validation_summary_by_condition.md` (0.775 ± 0.008 vs BCE 0.759 ± 0.044), `results/tables/test_summary_by_condition.md` (0.791 ± 0.027 vs BCE 0.829 ± 0.010) |
| Diagnostic: weighted BCE | tests the simpler "just reweight positives" explanation | `losses.py: positive_class_weight_from_training` | `results/tables/test_summary_by_condition.md` (test 0.746 ± 0.010), `results/tables/experiment_record.md` (weight 93.58) |
| Diagnostic: positive-only Dice | tests whether empty slices in the Dice term matter | `losses.py: soft_dice_loss(positive_slices_only=True)` | `results/tables/test_summary_by_condition.md` (93.0 empty slices with false positives vs 29.7 for BCE) |
| Sensitivity: Dice weight 0.5 and 2 | is the result specific to λ = 1? | `configs/runs/bce_dice_lambda{0.5,2}_seed0.yaml` | `results/tables/test_summary_by_run.md` (test 0.792 and 0.796) |
| Stability: seeds 0, 1, 2 | gains must exceed seed spread | configs `seed` | `results/tables/validation_summary_by_condition.md`, `results/tables/test_summary_by_condition.md` (mean ± SD over seeds) |
| Same budget for every run | fair comparison | `configs/runs/*.yaml` | `tests/test_configs_and_parallel.py` |
| Checkpoint and threshold chosen on validation only | required protocol | `train.py`, `select.py` | `results/validation/<run>/selection.json`, `threshold_sweep.csv` |
| Thresholds at the grid edge are flagged | an edge choice may mean the grid was too narrow | `select.py` | `results/tables/experiment_record.md`: 7 of 14 runs at the edge (weighted BCE ×3, positive-only Dice ×2, λ = 0.5, λ = 2) |
| Method frozen before test; test evaluated once | required protocol | `freeze.py`, `evaluate.py` | `results/frozen_method/frozen_method_manifest.json(.sha256)`, `results/test/test_evaluation_ledger.json` (one attempt), `tests/test_protocol.py` |
| Claims about patients, lesion sizes, failures | error analysis required | `figures.py` | `results/tables/test_paired_patient_dice.md` (BCE + Dice worse for 11/16 patients), `results/tables/validation_dice_by_lesion_size.md`, `results/figures/validation_failure_overlays.png`, `notebooks/01_results_walkthrough.ipynb` |
