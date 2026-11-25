# Requirements checklist: ML Project 1 description → fresh project

Source: course project description (sections 1–9). Paths are relative to the repository root.
Status: ✅ covered, with code and saved results from the full GPU study (14 runs on 4 × NVIDIA L40S, 24 Nov 2026).
The report is `docs/report.md` / `docs/report.pdf`; the executed walkthrough is `notebooks/01_results_walkthrough.ipynb`.

## 1. Overview

| Requirement | Where | Status |
|---|---|---|
| Full workflow: understand data, prepare inputs, implement and train, evaluate, interpret | `scripts/01`–`07` | ✅ |
| Open-ended investigation with a specific question | `README.md` "Study design", `docs/proposal.md` | ✅ |

## 2. Task definition

| Requirement | Where | Status |
|---|---|---|
| Binary semantic segmentation; class 1 = mask pixels (> 0), class 0 = all others | `data.py: read_mask`, `model.py` (1 logit), `losses.py` | ✅ |
| Empty slice ≠ healthy patient | metrics report empty slices separately; report wording | ✅ (report §2) |
| Working image-to-mask prediction pipeline | `predict.py`, `scripts/07_predict_masks_for_images.py` (TIFF → PNG mask), smoke test runs it | ✅ |
| Self-defined investigation stated as "Does [change] affect [outcome] compared with [baseline] under [conditions]?" | `README.md`, `docs/proposal.md` | ✅ |
| Why it matters, what evidence would contradict it | `README.md`, `docs/proposal.md` | ✅ |
| Not just "train several models and pick the best" | main + diagnostic + sensitivity design, paired/subgroup analyses (`figures.py`) | ✅ |

## 3. Dataset

| Requirement | Where | Status |
|---|---|---|
| LGG dataset, 3,929 pairs / 110 patients, 256×256×3, masks binarised from nonzero | `data.py`, `data_audit.py` (checks 256×256×3) | ✅ |
| Channels are MRI sequences, not RGB | stored channel order kept (`tifffile`), named in `configs/data_preparation.yaml` | ✅ |
| Course fixed patient split and manifests | reads course `train/val/test_manifest.csv`; checked against course `patient_splits.csv` (`course_split_reference/`) | ✅ |
| Train 77/2,763, val 17/598, test 16/568 | `configs/data_preparation.yaml`, `01_prepare_data.py` stops on mismatch, `tests/test_data.py` | ✅ |
| Training: fit models and preprocessing parameters | `fit_channel_statistics` on train only | ✅ |
| Validation: select configurations, thresholds, stopping | `train.py` (best epoch), `select.py` (threshold) | ✅ |
| Test: final evaluation after freezing | `freeze.py`, `evaluate.py`, scripts 04/05 | ✅ |
| Use all training images | no subsetting except smoke-test flags, which mark runs unfreezable | ✅ |
| Never mix patients across splits | disjointness check in `data_audit.py`, `tests/test_data.py` | ✅ |
| No test cases in modelling decisions | test read only by `evaluate.py` after freeze; audit reads test IDs/sizes only | ✅ |

## 4A. Problem formulation

| Requirement | Where | Status |
|---|---|---|
| One primary question, motivation, approach, comparison, success measures, scope, plan | `docs/proposal.md` | ✅ |
| ~1-page proposal before the main experiments | `docs/proposal.md` (copy of the submitted text) | ✅ (already written) |
| Question refined with documented reason | not refined | n/a |

## 4B. Data understanding and preparation

| Requirement | Where | Status |
|---|---|---|
| Positive/empty masks, region sizes | `data_audit.py` → `tables/data_audit_splits_and_labels.md`, `figures/data_lesion_area_distribution.png` | ✅ |
| Channel differences, intensity distributions | per-channel mean/std/zero fraction → `tables/data_audit_channel_intensities.md` | ✅ |
| Data quality | duplicate image hashes, missing files, stored sizes, split-file agreement | ✅ (does not re-hash file contents against the manifest hashes) |
| Explain preprocessing and transforms | `README.md`, `data.py` docstring; report text | ✅ (`docs/report.md`) |
| Image–mask alignment | `augment` applies identical transforms; `tests/test_data.py` | ✅ |
| Fitted preprocessing on training data only | `fit_channel_statistics`; freeze refuses stats not `fitted_on: train` | ✅ |

## 4C. Baselines and implementation

| Requirement | Where | Status |
|---|---|---|
| End-to-end training and inference pipeline | scripts 01–07 | ✅ |
| (i) Trivial reference (always-empty) | `select.py: score_always_empty_reference`, `evaluate.py` | ✅ |
| (ii) Credible learned baseline (2D U-Net) | `model.py`, `configs/runs/bce_seed*.yaml` | ✅ |
| (iii) Proposed approach | `configs/runs/bce_dice_lambda1_seed*.yaml` | ✅ |
| Closest simpler alternative | weighted BCE (`weighted_bce_seed*`) | ✅ |
| Identify implemented / modified / reused / pretrained parts | `README.md` "Reused vs modified work" and "Attribution"; no pretrained weights | ✅ |

## 4D. Experimental comparisons

| Requirement | Where | Status |
|---|---|---|
| 1) Main comparison on the same split | `bce` vs `bce_dice_lambda1`, seeds 0–2 | ✅ |
| 2) Diagnostic comparison | `weighted_bce`, `bce_positive_only_dice`, seeds 0–2 | ✅ |
| 3) Stability / sensitivity | seeds 0–2; Dice weight 0.5 and 2 | ✅ |
| Consistent metrics, comparable budgets | one `metrics.py`; identical budgets enforced by `tests/test_configs_and_parallel.py` | ✅ |
| Validation-only model selection | `train.py`, `select.py` | ✅ |
| Record configs, seeds, hardware, parameter counts, training time | `results/runs/<run>/config.yaml`, `run_summary.json`; `tables/experiment_record.md` | ✅ |

## 4E. Analysis and discussion

| Requirement | Where | Status |
|---|---|---|
| Representative prediction overlays | `figures/validation_representative_overlays.png` (slices at 25/50/75% Dice quantiles) | ✅ |
| ≥1 quantitative failure or subgroup analysis | lesion-size subgroups (train-fitted tertiles), paired per-patient Dice, `figures/validation_failure_overlays.png` | ✅ |
| Trade-offs | `figures/*_positive_dice_vs_empty_false_positives.png`, threshold curves | ✅ |
| What the experiments show/don't show, limitations, negative results | report text (`docs/report_outline.md` maps sections to evidence) | ✅ (report §6) |
| Claims backed by saved results | every table/figure is written to `results/` | ✅ |

## 5. Metrics and protocol

| Requirement | Where | Status |
|---|---|---|
| Instructor partitions, common metric definitions | `data.py`, `metrics.py` | ✅ |
| Evaluate against original 256×256 masks | `per_slice_table` asserts 256×256; no resizing anywhere | ✅ |
| Foreground Dice 2TP/(2TP+FP+FN), IoU TP/(TP+FP+FN) | `metrics.py` (pooled + per slice), `tests/test_metrics.py` | ✅ |
| Positive-slice Dice | `positive_slice_dice` (+ IoU) | ✅ |
| Empty-slice errors (count and/or pixel burden) | count, rate, total and mean FP pixels | ✅ |
| Patient-level results | patient Dice mean/median/std/min, per-patient CSVs, paired analysis | ✅ |
| Pixel accuracy (optional context) | `pixel_accuracy` | ✅ |
| Empty handling: both empty = 1, one empty = 0, same for every method | `dice_score`/`iou_score`, tests | ✅ |
| Positive-slice metrics reported separately | in every summary/table | ✅ |
| Hyperparameters/thresholds/checkpoints from train/val only | `train.py`, `select.py` | ✅ |
| Test with frozen method, no repeated tuning on test | `freeze.py` hashes; `evaluate.py` runs once (lock + ledger), `tests/test_protocol.py` | ✅ |

## 6. Expected performance

| Requirement | Where | Status |
|---|---|---|
| Correct pipeline + sound investigation; negative result acceptable | design + tests | ✅ |
| Implementation errors identified and corrected or disclosed | pilot augmentation bug kept fixed and documented (`data.py`, README, proposal) | ✅ (report §2) |

## 7. Deliverables

| Deliverable | Where | Status |
|---|---|---|
| 1) Proposal (~1 page, six headings) | `docs/proposal.md` | ✅ |
| 2) Final report (6–8 pages) | `docs/report.md`, `docs/report.pdf` (8 pages incl. references and appendices) | ✅ |
| 3) Runnable training/evaluation/inference scripts | `scripts/01`–`07` | ✅ |
| 3) Requirements / environment information | `requirements.txt`, `environment.yml`, `requirements-development.txt`, `run_summary.json` environment record | ✅ |
| 3) README with reproduction steps | `README.md` | ✅ |
| 4) Saved configurations and metrics | `configs/runs/`, `results/runs/<run>/{config.yaml,metrics.csv}`, `results/validation`, `results/test` | ✅ |
| 4) Relevant figures | `results/figures/` | ✅ |
| 4) Table linking decisions/claims to experiments or code | `docs/decision_evidence_table.md` | ✅ |
| 4) Model artifacts / checkpoints | `best.pt` files (~31 MB each) are kept out of git; SHA-256 of each is recorded in `results/frozen_method/frozen_method_manifest.json` and the files are shared separately (Drive) | ✅ |

## 8. Grading criteria (100 points)

Each row of the official rubric is mapped to the file that satisfies it. Status: ✅ covered by code, saved results and the report.

### Problem formulation and motivation (15)

| Rubric expectation | Evidence | Status |
|---|---|---|
| Specific testable question | `docs/proposal.md`, `README.md` (question in the form "Does [change] affect [outcome] compared with [baseline] under [conditions]?") | ✅ |
| Dataset-relevant motivation | `docs/proposal.md` (class imbalance, pilot positive-slice Dice 0.6061, empty-slice false positives 24/394, a patient with Dice 0) | ✅ |
| Clear scope, comparison and measurable objective | `docs/proposal.md` comparison plan; what would contradict the hypothesis (seed spread, similar weighted-BCE gains, or more empty-slice false positives) | ✅ |

### Data understanding and preparation (15)

| Rubric expectation | Evidence | Status |
|---|---|---|
| Relevant data audit | `scripts/01_prepare_data.py`, `src/brain_mri_segmentation/data_audit.py`; `results/tables/data_audit_splits_and_labels.md`, `data_audit_channel_intensities.md`, `results/figures/data_lesion_area_distribution.png` | ✅ |
| Well-justified preprocessing | `src/brain_mri_segmentation/data.py` (tifffile channel order, masks binarised with > 0, native 256×256, joint flip/rot90); explained in `README.md` | ✅ |
| Correct masks and patient-aware handling | course `patient_splits.csv` checked in `01_prepare_data.py`; `tests/test_data.py` (77/17/16, no shared patients); aligned augmentation tests | ✅ |
| No leakage | normalisation fitted on training images only (`fitted_on: train`); checkpoint and threshold chosen on validation only (`train.py`, `select.py`); test read once after freezing (`evaluate.py`); test-label statistics are never computed in the audit | ✅ |

### Approach and implementation (15)

| Rubric expectation | Evidence | Status |
|---|---|---|
| Technically sound pipeline and proposed approach | `scripts/01`–`07`; 2D U-Net from scratch (`model.py`, 7,763,041 parameters); proposed loss in `losses.py` and `configs/runs/bce_dice_lambda1_seed*.yaml` | ✅ |
| Appropriate implementation choices | identical budgets in every run config; per-slice augmentation RNG; validation-only selection; one-time test protocol | ✅ |
| Clear account of reused or modified work | `README.md` section "Reused vs modified work"; no pretrained weights | ✅ |

### Experimental evaluation (25)

| Rubric item (points) | Evidence | Status |
|---|---|---|
| Relevant baselines (5) | trivial always-empty reference (`select.py`, `evaluate.py`); learned baseline = same U-Net with BCE (`configs/runs/bce_seed0..2.yaml`) | ✅ |
| Fair protocol (5) | same data, model, budget and validation rule for every run (`tests/test_configs_and_parallel.py`); method frozen before test; test evaluated once (`freeze.py`, `evaluate.py`, `tests/test_protocol.py`) | ✅ |
| Main comparison (5) | BCE vs BCE+Dice (λ=1), seeds 0, 1 and 2; `results/tables/validation_summary_by_condition.md`, `test_summary_by_condition.md` | ✅ |
| Diagnostic experiment (5) | weighted BCE (simpler reweighting) and BCE + positive-only Dice (role of empty slices), seeds 0–2 | ✅ |
| Stability or sensitivity analysis (5) | seeds 0–2 for the four loss conditions; Dice weights 0.5 and 2 at seed 0; seed mean/std in `*_summary_by_condition.md` | ✅ |
| Record of configs, seeds, hardware, parameter counts and training time | `results/runs/<run>/{config.yaml,run_summary.json}`; `results/tables/experiment_record.md` | ✅ |

### Analysis and discussion (15)

| Rubric expectation | Evidence | Status |
|---|---|---|
| Evidence-based interpretation | all claims can be tied to `results/tables/` and `results/figures/`; `docs/decision_evidence_table.md` and `docs/report_outline.md` | ✅ (report §5–6, Appendix A) |
| Substantive error analysis | `results/figures/validation_failure_overlays.png`, `validation_representative_overlays.png`; `results/tables/validation_dice_by_lesion_size.md`; `*_paired_patient_dice.md` | ✅ (report §5–6) |
| Trade-offs | `results/figures/*_positive_dice_vs_empty_false_positives.png`, `validation_threshold_curves.png`; grid-edge flag in `experiment_record.md` | ✅ (report §5–6) |
| Alternative explanations | diagnostic runs (weighted BCE; positive-only Dice) are the planned alternative explanations | ✅ (report §5–6) |
| Honest limitations and negative results | proposal already states the pilot does not promise an improvement; report must discuss seed spread, grid-edge thresholds and any negative result | ✅ (report §6 Limitations) |

### Evidence of work and reproducibility (10)

| Rubric expectation | Evidence | Status |
|---|---|---|
| Traceable experiments, configurations and outcomes | `configs/runs/*.yaml`, `results/runs/`, `results/validation/`, `results/frozen_method/`, `results/test/test_evaluation_ledger.json`, `results/tables/experiment_record.md` | ✅ |
| Readable code and enough instructions to reproduce | `README.md` (setup and exact commands), `studio_setup.sh`, `tests/` (55 unit tests), `smoke_test/run_smoke_test.sh` | ✅ |

### Report quality and attribution (5)

| Rubric expectation | Evidence | Status |
|---|---|---|
| Clear organization, readable tables and figures | `docs/report_outline.md` (6–8 page structure); tables/figures from `scripts/06_make_figures_and_tables.py` | ✅ (`docs/report.md`) |
| Proper references | `docs/proposal.md`, `README.md` Attribution (U-Net, V-Net, Buda/LGG, libraries) | ✅ |
| Disclosure of reused code, pretrained resources and assistance | `README.md` "Reused vs modified work" and "Attribution"; no pretrained weights; AI assistance disclosed | ✅ (report Appendix B) |


## 9. Technical and academic requirements

| Requirement | Where | Status |
|---|---|---|
| GPU training; report hardware and runtime | GPU used automatically; `run_summary.json` records GPU, CUDA, time; parallel multi-GPU option | ✅ |
| Complete training partition for the main method | default; smoke-test subsets cannot be frozen | ✅ |
| No other datasets, no test-label-informed development | only course files; test read once after freezing | ✅ |
| Disclose external code, pretrained weights, AI assistance | `README.md` "Reused vs modified work" and "Attribution" (no pretrained weights) | ✅ |

## Remaining notes

1. **File-content hash check**: the audit checks duplicate hashes, missing files and the split, but does not
   recompute each TIFF's SHA-256 against the manifest.
2. **Checkpoints** are not in the repository (size); their hashes are in the frozen-method manifest.
