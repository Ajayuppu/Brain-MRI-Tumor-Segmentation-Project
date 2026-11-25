# Brain MRI Segmentation (LGG): does adding soft Dice to BCE help?

ML Project 1, Ajay Uppu, MS Data Science & Analytics, Georgia State University.

This repository is a clean, fresh implementation of the project. It trains a 2D U-Net to segment
tumour-related regions in the LGG brain MRI dataset and tests one question:

> Does adding batch-pooled soft Dice loss to binary cross-entropy (BCE) improve positive-slice Dice without
> increasing false positives on empty slices, compared with BCE alone, when the U-Net and the
> patient-disjoint data are kept the same?

What would contradict the hypothesis: gains within seed-to-seed variation, similar gains from weighted BCE
(a simpler reweighting), or more empty slices with false positives.

## Results

Full study: 14 runs on 4 × NVIDIA L40S (SageMaker Studio, ml.g6e.12xlarge), 707–731 s per run, 2.79 GPU-hours,
24 Nov 2026. Epoch and threshold chosen on validation only; the frozen methods were evaluated on the test set once
(`results/test/test_evaluation_ledger.json` has one attempt). Mean ± SD over seeds 0–2 (n = 1 for λ = 0.5 / 2).

| Condition | Validation pooled Dice | Test pooled Dice | Test positive-slice Dice | Test empty slices with FP (of 368) |
|---|---:|---:|---:|---:|
| Always-empty mask | 0.000 | 0.000 | 0.000 | 0 |
| BCE (baseline) | 0.759 ± 0.044 | **0.829 ± 0.010** | **0.730** | **29.7** |
| BCE + Dice, λ = 1 (proposed) | **0.775 ± 0.008** | 0.791 ± 0.027 | 0.699 | 35.0 |
| Weighted BCE (diagnostic) | 0.685 ± 0.028 | 0.746 ± 0.010 | 0.684 | 66.7 |
| BCE + positive-only Dice (diagnostic) | 0.744 ± 0.046 | 0.788 ± 0.048 | 0.725 | 93.0 |
| BCE + Dice, λ = 0.5 | 0.738 | 0.792 | 0.676 | 29 |
| BCE + Dice, λ = 2 | 0.744 | 0.796 | 0.684 | 28 |

**Answer: not supported.** Validation slightly favoured BCE + Dice (smaller seed spread), but the gap was inside
BCE's seed spread and positive-slice Dice did not improve. On the test set plain BCE was better on every primary
metric and BCE + Dice was worse for 11 of 16 test patients. Weighted BCE did not explain any effect; removing
empty slices from the Dice term tripled empty-slice false positives. 7 of 14 runs picked a threshold at the
edge of the 0.05–0.95 grid (all weighted BCE runs), and small lesions remain the main failure mode for every loss
(BCE seed 0: validation slice Dice 0.36 on small vs 0.83 on large lesions).

- Report: [`docs/report.md`](docs/report.md) / [`docs/report.pdf`](docs/report.pdf)
- Proposal (pre-experiment plan): [`docs/proposal.md`](docs/proposal.md) / [`docs/proposal.pdf`](docs/proposal.pdf)
- Executed results notebook: [`notebooks/01_results_walkthrough.ipynb`](notebooks/01_results_walkthrough.ipynb)
- Claims and decisions linked to evidence: [`docs/decision_evidence_table.md`](docs/decision_evidence_table.md)
- All saved outputs: [`results/`](results/) (checkpoints `best.pt` are not in git because of their size; their
  SHA-256 hashes are in `results/frozen_method/frozen_method_manifest.json`)

## Study design

| Part | Runs (configs in `configs/runs/`) | Purpose |
|---|---|---|
| Trivial reference | always-empty mask (no training) | lower bound; shows why positive-slice metrics are reported |
| Learned baseline | `bce_seed0`, `bce_seed1`, `bce_seed2` | standard 2D U-Net with BCE |
| Proposed approach | `bce_dice_lambda1_seed0..2` | BCE + 1.0 × batch-pooled soft Dice |
| Diagnostic: simpler reweighting | `weighted_bce_seed0..2` | BCE with positive-class weight min((1−f)/f, 100), f = training foreground fraction |
| Diagnostic: role of empty slices | `bce_positive_only_dice_seed0..2` | Dice term computed only on slices that contain a lesion |
| Sensitivity: loss weight | `bce_dice_lambda0.5_seed0`, `bce_dice_lambda2_seed0` | Dice weight 0.5 and 2 |
| Stability: random seed | seeds 0, 1, 2 for the four loss conditions | seed-to-seed spread |

Every run uses the same model, data and budget; only the loss (and the seed) changes:
2D U-Net from scratch (7,763,041 parameters, widths 32–512, no pretrained weights), native 256×256 three-channel
input, 50 epochs, batch size 16, AdamW (learning rate 0.001, weight decay 0.0001), cosine schedule, 4 data-loader
workers, training augmentation = identical horizontal flip and 90° rotations of image and mask.

**Data split.** The course-provided patient-level split is used unchanged: train 77 patients / 2,763 slices,
validation 17 / 598, test 16 / 568, read from the course files `train_manifest.csv`, `val_manifest.csv` and
`test_manifest.csv`. `scripts/01_prepare_data.py` checks the split against the course `patient_splits.csv`
(copied in `course_split_reference/`) and stops if anything differs. All training slices are used.

**Validation-only selection.** After every epoch the validation set is scored for thresholds 0.05–0.95
(step 0.05); the checkpoint kept is the epoch with the highest pooled foreground Dice at its best threshold, and the
threshold is then re-selected on that checkpoint. The same rule applies to every run.

**Test once.** `04_freeze_method.py` hashes every checkpoint, threshold, normalisation file, config, the
evaluation code and the data manifests. `05_evaluate_test_once.py` refuses to run if anything changed, claims the
test set atomically, predicts from test images first, records when test masks are read, and refuses to run a
second time.

**Metrics** (all at the original 256×256): foreground Dice and IoU (pooled), mean positive-slice Dice and IoU,
empty slices with any false-positive pixel (count, rate, pixel burden), patient-level Dice (mean, median, standard
deviation, minimum), and pixel accuracy as optional context. Per-slice convention: both empty = 1, only one
empty = 0.

## Project layout

```
configs/paths.yaml               where the data is and where results go  <- edit data_root here
configs/runs/<run_name>.yaml     one file per training run (14 files)
configs/data_preparation.yaml    expected split sizes, audit settings
configs/analysis.yaml            settings for tables and figures
course_split_reference/          course patient_splits.csv and split_summary.csv (unchanged copies)
src/brain_mri_segmentation/      the code (see "Code map")
scripts/01..07                   the pipeline, run in order
smoke_test/                      tiny synthetic dataset + end-to-end check (CPU, about 1 minute)
tests/                           unit tests (CPU, about 25 seconds)
docs/                            proposal, report (md + pdf), decision/evidence table, requirements checklist
notebooks/                       executed results walkthrough (reads results/, CPU only)
results/                         everything the scripts write (see results/README.md)
```

## Setup in SageMaker Studio

1. In the new SageMaker Studio JupyterLab (home `/home/sagemaker-user`), unpack the code into the project folder:
   ```bash
   mkdir -p ~/mlproject1-brain-mri-segmentation
   tar -xzf ~/brain-mri-segmentation.tar.gz -C ~/mlproject1-brain-mri-segmentation --strip-components=1
   cd ~/mlproject1-brain-mri-segmentation
   ```
   Or run `bash ~/studio_setup.sh` after the code tar is in `~`. That script looks for the dataset in this order:
   a copy already at `data/`, a local `~/lgg-mri-dataset.tar.gz`, then S3 (only when you set `DATA_BUCKET=<your-bucket-name>`; it syncs a prepared train/val/test
   folder if present, otherwise it copies the staged archives `data/trainval/trainval.tar` and
   `data/test_holdout/test_holdout.tar`), and finally the public course Drive folder via `gdown`.
2. Check the dataset path in `configs/paths.yaml`. It is set to
   `/home/sagemaker-user/mlproject1-brain-mri-segmentation/data`, which must contain `train_manifest.csv`,
   `val_manifest.csv`, `test_manifest.csv`, `patient_splits.csv` and the `train/`, `val/`, `test/` folders
   (the course dataset folder `LGG-MRI-Segmentation/dataset`). Any script also accepts `--data-root <folder>`.
3. Install the Python packages (PyTorch comes from the Studio image) and run the checks:
   ```bash
   pip install -r requirements.txt
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), 'GPUs')"
   python -m pytest -q                 # unit tests, ~25 s
   bash smoke_test/run_smoke_test.sh   # end-to-end on synthetic data, ~1 min
   ```

## Run the study

```bash
python scripts/01_prepare_data.py                       # data audit + training-only normalisation (~2 min)

# Train the 14 runs. Choose ONE:
python scripts/02_train_all_runs.py                     # one GPU, one run after another (~12 min per run)
python scripts/02_train_all_runs.py --parallel-gpus     # every visible GPU, one run per GPU at a time
python scripts/02_train_all_runs.py --parallel-gpus 0 1 --runs-per-gpu 2   # chosen GPUs, two runs per GPU

python scripts/03_select_on_validation.py               # thresholds on validation + always-empty reference
python scripts/06_make_figures_and_tables.py            # validation tables/figures (review before freezing)

# Only when every method decision is final:
python scripts/04_freeze_method.py --confirm-method-is-final
python scripts/05_evaluate_test_once.py --confirm-method-is-frozen
python scripts/06_make_figures_and_tables.py            # adds the test tables and figures

# Inference on any slice (binary PNG mask):
python scripts/07_predict_masks_for_images.py --run bce_dice_lambda1_seed0 /path/to/slice.tif
```

`02_train_all_runs.py` can be stopped and started again at any time: finished runs are skipped and an
interrupted run continues from its last finished epoch. In parallel mode each run is its own process pinned to
one GPU (`CUDA_VISIBLE_DEVICES`), so results do not depend on the schedule; its console output goes to
`results/runs/<run>/console.log`. Run it inside `tmux` or `nohup ... &` so closing the browser does not stop it.

### Expected runtime

About 12 minutes per run on one NVIDIA L40S (the earlier study measured 726–744 s per run), so:

| Hardware | Schedule | 14 runs |
|---|---|---|
| 1 × L40S (e.g. ml.g6e.2xlarge) | sequential | about 2 h 50 min |
| 1 × L40S | `--parallel-gpus --runs-per-gpu 2` | roughly 1 h 45 min–2 h 30 min (limited by CPU data loading) |
| 4 × L40S (e.g. ml.g6e.12xlarge) | `--parallel-gpus` | about 45–50 min (4 rounds) |
| 8 GPUs | `--parallel-gpus` | about 25–30 min (2 rounds) |

Each run uses 4 data-loader worker processes; give each concurrent run at least 4 CPU cores. Steps 1, 3, 5
and 6 take a few minutes in total.

### Re-check the saved results (CPU, no data or GPU needed)

```bash
pip install pandas matplotlib scipy jupyter
(cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_results_walkthrough.ipynb)
sha256sum results/frozen_method/frozen_method_manifest.json   # equals the .sha256 file and the test ledger
```

## Expected outputs

See `results/README.md`. The main evidence for the report:
`results/tables/validation_summary_by_condition.md`, `test_summary_by_condition.md`, `experiment_record.md`
(seeds, hardware, parameters, training time), `validation_paired_patient_dice.md`,
`validation_dice_by_lesion_size.md`, `data_audit_*.md`, and `results/figures/*.png`.

## Code map

| File | What it does |
|---|---|
| `src/brain_mri_segmentation/config.py` | loads `configs/`; the 14-run order |
| `src/brain_mri_segmentation/data.py` | manifests, TIFF reading, binary masks, training-only normalisation, aligned augmentation with per-slice randomness |
| `src/brain_mri_segmentation/data_audit.py` | split checks against the course split, label and intensity statistics (no test labels) |
| `src/brain_mri_segmentation/model.py` | 2D U-Net, 7,763,041 parameters |
| `src/brain_mri_segmentation/losses.py` | BCE, BCE + soft Dice, BCE + positive-only Dice, weighted BCE |
| `src/brain_mri_segmentation/metrics.py` | Dice, IoU, positive-slice and empty-slice metrics, patient Dice, threshold sweep |
| `src/brain_mri_segmentation/train.py` | one run: training, per-epoch validation, best checkpoint, resumable |
| `src/brain_mri_segmentation/parallel.py` | runs several trainings at once across GPUs |
| `src/brain_mri_segmentation/select.py` | validation threshold selection, always-empty reference |
| `src/brain_mri_segmentation/freeze.py` | frozen-method manifest and its verification |
| `src/brain_mri_segmentation/evaluate.py` | one-time, two-pass test evaluation with a ledger |
| `src/brain_mri_segmentation/figures.py` | tables and figures |
| `src/brain_mri_segmentation/predict.py` | inference to PNG masks |

## Reused vs modified work

This section is the account of reused, modified and original work asked for in the project description
(section 4C and the 15-point implementation criterion).

**Reused, unchanged**
- The course dataset, patient-level split and manifests (`train_manifest.csv`, `val_manifest.csv`,
  `test_manifest.csv`, `patient_splits.csv`). They are used as provided; `scripts/01_prepare_data.py`
  checks them and stops if they differ from `course_split_reference/`.
- The U-Net *architecture* of Ronneberger et al. (2015): 3 input channels, widths 32/64/128/256/512,
  four pooling stages, skip connections. Implemented from scratch here; **no pretrained weights** and no
  copied official implementation.
- The soft-Dice formula of Milletari et al. (V-Net, 2016), used as a batch-pooled term added to BCE.
- Python libraries: PyTorch, NumPy, pandas, tifffile, Matplotlib, PyYAML, Pillow.

**Reused in substance, rewritten and renamed**
- The scientific protocol and hyperparameters of the earlier study code (`versions/study_v2` of the first
  Studio project): the same 14 runs, losses, 50-epoch budget, AdamW + cosine schedule, batch size 16,
  validation-only epoch and threshold selection (grid 0.05–0.95, step 0.05), metric conventions
  (both-empty = 1, one-empty = 0), and the freeze-then-evaluate-once test rule.
- The U-Net parameter *names* (`downs`, `ups`, `decs`, `head`) were kept so an old `best.pt` still loads.
- On CPU, with the same seed and settings, this code produces bit-identical weights to that earlier code
  (checked for BCE, BCE+Dice, weighted BCE and positive-only Dice on a synthetic dataset). GPU training
  is not bit-for-bit deterministic (cuDNN), so new GPU numbers will be close to, but not exactly the same as,
  the earlier study.

**Modified or added in this package**
- Clearer module and file names (`src/brain_mri_segmentation/`, one YAML file per run, numbered scripts).
- Training can be resumed from the last finished epoch, and `02_train_all_runs.py --parallel-gpus` can
  run several trainings at once (one process per GPU).
- Threshold selection is its own step (`03_select_on_validation.py`); the rule is unchanged.
- The frozen-method manifest also hashes each run's validation `selection.json`.
- A course-split data audit, experiment-record table, lesion-size and paired-patient analyses,
  representative and failure overlays, and a TIFF-to-PNG inference script.
- The earlier pilot's augmentation bug (DataLoader workers repeating one random stream) stays fixed:
  each sample is drawn from a generator keyed by (seed, epoch, index).

**Not used**
- No pretrained weights, no other datasets, and no test labels for any modelling decision.
- Studio-only extras from the earlier project (old audit notes, snapshot tools) were not carried over.


## Attribution

- Data: M. Buda, A. Saha, M. A. Mazurowski, Computers in Biology and Medicine 109 (2019) 218–225; TCIA TCGA-LGG;
  Kaggle `mateuszbuda/lgg-mri-segmentation`, CC BY-NC-SA 4.0. Patient split and manifests provided by the course.
- U-Net: O. Ronneberger, P. Fischer, T. Brox, MICCAI 2015 (implemented here from scratch; no pretrained weights).
- Soft Dice loss: F. Milletari, N. Navab, S.-A. Ahmadi, V-Net, 3DV 2016.
- Libraries: PyTorch, NumPy, pandas, tifffile, Matplotlib, PyYAML, Pillow.
- AI assistance: an AI assistant helped write and restructure the code, tests and documentation. The student
  reviews, runs and verifies every result and claim.
