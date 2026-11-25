#!/usr/bin/env bash
# End-to-end smoke test on a tiny SYNTHETIC dataset (CPU, about 1-2 minutes):
# data preparation -> train two runs for 2 epochs -> validation selection -> freeze -> test once
# -> second test attempt must be refused -> figures and tables -> inference on one image.
# Nothing here touches the real dataset or the real results/ folder.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${1:-$(mktemp -d)}"
DATA="$WORK_DIR/synthetic_dataset"
RESULTS="$WORK_DIR/smoke_test_results"
RUNS="bce_seed0 bce_dice_lambda1_seed0"
cd "$PROJECT_ROOT"
SMOKE_TRAINING="--smoke-test-epochs 2 --smoke-test-batch-size 4 --smoke-test-data-loader-workers 0"

python smoke_test/make_synthetic_dataset.py --output-dir "$DATA"
python scripts/01_prepare_data.py --data-root "$DATA" --results-dir "$RESULTS" \
  --skip-course-size-check --course-patient-split-file "$DATA/patient_splits.csv"
python scripts/02_train_all_runs.py --data-root "$DATA" --results-dir "$RESULTS" --runs $RUNS --device cpu $SMOKE_TRAINING
echo "--- second call must skip finished runs"
python scripts/02_train_all_runs.py --data-root "$DATA" --results-dir "$RESULTS" --runs $RUNS --device cpu $SMOKE_TRAINING
python scripts/03_select_on_validation.py --data-root "$DATA" --results-dir "$RESULTS" --runs $RUNS --device cpu
echo "--- test evaluation before freezing must be refused"
if python scripts/05_evaluate_test_once.py --data-root "$DATA" --results-dir "$RESULTS" --confirm-method-is-frozen --device cpu; then
  echo "ERROR: test evaluation ran without a frozen method"; exit 1; fi
echo "--- freezing smoke-test runs without the smoke-test flag must be refused"
if python scripts/04_freeze_method.py --data-root "$DATA" --results-dir "$RESULTS" --runs $RUNS --confirm-method-is-final; then
  echo "ERROR: smoke-test runs were frozen for the study"; exit 1; fi
python scripts/04_freeze_method.py --data-root "$DATA" --results-dir "$RESULTS" --runs $RUNS --confirm-method-is-final --allow-smoke-test-runs
python scripts/05_evaluate_test_once.py --data-root "$DATA" --results-dir "$RESULTS" --confirm-method-is-frozen --device cpu
echo "--- second test evaluation must be refused"
if python scripts/05_evaluate_test_once.py --data-root "$DATA" --results-dir "$RESULTS" --confirm-method-is-frozen --device cpu; then
  echo "ERROR: test evaluation ran twice"; exit 1; fi
python scripts/06_make_figures_and_tables.py --data-root "$DATA" --results-dir "$RESULTS" --runs $RUNS
python scripts/07_predict_masks_for_images.py --data-root "$DATA" --results-dir "$RESULTS" --run bce_dice_lambda1_seed0 \
  --device cpu "$DATA/val/SYNTHETIC_VAL_00/images/SYNTHETIC_VAL_00_20000101_1.tif"
echo "SMOKE TEST PASSED. Outputs: $RESULTS"
