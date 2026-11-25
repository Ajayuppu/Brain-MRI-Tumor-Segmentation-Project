#!/usr/bin/env bash
# Run this in the SageMaker Studio JupyterLab terminal (home: /home/sagemaker-user).
# It unpacks the project, brings in the course dataset, installs Python packages,
# checks the code, prepares the data, then starts the 14 training runs in the
# background across every visible GPU.
#
# Put this in your home folder first, if you have it:
#   ~/brain-mri-segmentation.tar.gz   (the code package)
#   ~/lgg-mri-dataset.tar.gz          (optional verified data package)
#   ~/studio_setup.sh                 (this script)
#
# Data is obtained in this order:
#   1. Already present at $DATA_ROOT
#   2. ~/lgg-mri-dataset.tar.gz, if you uploaded it
#   3. S3, only if you set DATA_BUCKET=<your-bucket-name> before running this script:
#      sync a prepared train/val/test folder if one is there, otherwise copy the known
#      archives data/trainval/trainval.tar and data/test_holdout/test_holdout.tar
#   4. Google Drive (gdown) of the course dataset folder (anyone with the link can view)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/sagemaker-user/mlproject1-brain-mri-segmentation}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
CODE_TAR="${CODE_TAR:-$HOME/brain-mri-segmentation.tar.gz}"
DATA_TAR="${DATA_TAR:-$HOME/lgg-mri-dataset.tar.gz}"
# Course Drive folder LGG-MRI-Segmentation/dataset (anyone with the link can view).
DATASET_DRIVE_ID="${DATASET_DRIVE_ID:-1CcO1Ou27_h6_NEl1pA9A145a_Y3QFWC2}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
TRAIN_LOG="${PROJECT_ROOT}/results/logs/train_all.log"
# Known SHA-256 of the staged archives (from the original transfer verification).
TRAINVAL_TAR_SHA256="${TRAINVAL_TAR_SHA256:-d20e09fe1930086d6c913a73206038abca362d98c12f5e68ee6b3d3e921ad0ae}"
TEST_HOLDOUT_TAR_SHA256="${TEST_HOLDOUT_TAR_SHA256:-c7c58e3957dff311782e98ae2d547c6fa5d5ee3f7c18c5c49c0354f0829985e9}"

echo "=== GPU status ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. Training will fall back to CPU and will be very slow."
fi

unpack_code() {
  if [[ -f "${PROJECT_ROOT}/scripts/02_train_all_runs.py" ]]; then
    echo "Code already present at ${PROJECT_ROOT}"
    return
  fi
  if [[ ! -f "${CODE_TAR}" ]]; then
    echo "ERROR: ${CODE_TAR} not found. Upload brain-mri-segmentation.tar.gz to your home folder first."
    exit 1
  fi
  echo "Unpacking ${CODE_TAR} into ${PROJECT_ROOT}"
  mkdir -p "${PROJECT_ROOT}"
  tar -xzf "${CODE_TAR}" -C "${PROJECT_ROOT}" --strip-components=1
}

dataset_is_complete() {
  [[ -f "${DATA_ROOT}/train_manifest.csv" && -f "${DATA_ROOT}/val_manifest.csv" && -f "${DATA_ROOT}/test_manifest.csv" && -f "${DATA_ROOT}/patient_splits.csv" ]] \
    && [[ -d "${DATA_ROOT}/train" && -d "${DATA_ROOT}/val" && -d "${DATA_ROOT}/test" ]]
}

# The bucket comes from the DATA_BUCKET environment variable; its name is never printed.
discover_data_buckets() {
  if [[ -n "${DATA_BUCKET:-}" ]]; then
    printf '%s\n' "${DATA_BUCKET}"
  fi
}

s3_object_exists() {
  local uri="$1"
  aws s3 ls --region "${AWS_REGION}" "${uri}" >/dev/null 2>&1
}

s3_prefix_has_prepared_layout() {
  local prefix="$1"
  s3_object_exists "${prefix}train_manifest.csv" \
    && s3_object_exists "${prefix}val_manifest.csv" \
    && s3_object_exists "${prefix}test_manifest.csv" \
    && s3_object_exists "${prefix}patient_splits.csv" \
    && aws s3 ls --region "${AWS_REGION}" "${prefix}train/" >/dev/null 2>&1 \
    && aws s3 ls --region "${AWS_REGION}" "${prefix}val/" >/dev/null 2>&1 \
    && aws s3 ls --region "${AWS_REGION}" "${prefix}test/" >/dev/null 2>&1
}

promote_extracted_tree() {
  # Move train/val/test and the course CSV/README files from a temp extract into DATA_ROOT.
  local src="$1"
  mkdir -p "${DATA_ROOT}"
  # Some archives wrap the files in one top-level folder.
  if [[ ! -d "${src}/train" && ! -d "${src}/test" && ! -d "${src}/test_holdout" ]]; then
    local only
    only="$(find "${src}" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
    [[ -n "${only}" ]] && src="${only}"
  fi
  if [[ -d "${src}/test_holdout" && ! -d "${src}/test" ]]; then
    mv "${src}/test_holdout" "${src}/test"
  fi
  if [[ -f "${src}/test_holdout_manifest.csv" && ! -f "${src}/test_manifest.csv" ]]; then
    mv "${src}/test_holdout_manifest.csv" "${src}/test_manifest.csv"
  fi
  local name
  for name in train val test train_manifest.csv val_manifest.csv test_manifest.csv \
              patient_splits.csv split_summary.csv manifest.csv README.md; do
    if [[ -e "${src}/${name}" ]]; then
      rm -rf "${DATA_ROOT}/${name}"
      mv "${src}/${name}" "${DATA_ROOT}/${name}"
    fi
  done
}

check_archive_sha256() {
  local path="$1" expected="$2" label="$3"
  [[ -z "${expected}" ]] && return 0
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "WARNING: ${label} SHA-256 ${actual} does not match the recorded ${expected}."
    return 1
  fi
  echo "${label} SHA-256 matches the recorded archive hash."
  return 0
}

try_s3_prepared_prefix() {
  local bucket="$1"
  local prefix
  # Prefer a already-assembled train/val/test folder if one exists.
  for prefix in \
      "s3://${bucket}/data/" \
      "s3://${bucket}/dataset/" \
      "s3://${bucket}/" \
      "s3://${bucket}/lgg-mri-segmentation/dataset/" \
      "s3://${bucket}/mlproject1-brain-mri-segmentation/data/"; do
    if s3_prefix_has_prepared_layout "${prefix}"; then
      echo "Found a prepared dataset prefix on S3; syncing into ${DATA_ROOT}."
      mkdir -p "${DATA_ROOT}"
      aws s3 sync --region "${AWS_REGION}" "${prefix}" "${DATA_ROOT}/" \
        --exclude "*" --include "train/*" --include "val/*" --include "test/*" \
        --include "train_manifest.csv" --include "val_manifest.csv" --include "test_manifest.csv" \
        --include "patient_splits.csv" --include "split_summary.csv" --include "manifest.csv" \
        --include "README.md"
      return 0
    fi
  done
  return 1
}

try_s3_known_archives() {
  local bucket="$1"
  local work trainval_uri holdout_uri
  trainval_uri="s3://${bucket}/data/trainval/trainval.tar"
  holdout_uri="s3://${bucket}/data/test_holdout/test_holdout.tar"
  if ! s3_object_exists "${trainval_uri}"; then
    # The holdout archive might also sit directly under the prefix.
    if s3_object_exists "s3://${bucket}/data/trainval/trainval.tar"; then
      :
    else
      return 1
    fi
  fi
  if ! s3_object_exists "${holdout_uri}"; then
    if s3_object_exists "s3://${bucket}/data/test_holdout/test.tar"; then
      holdout_uri="s3://${bucket}/data/test_holdout/test.tar"
    elif s3_object_exists "s3://${bucket}/test_holdout.tar"; then
      holdout_uri="s3://${bucket}/test_holdout.tar"
    else
      echo "Found the train/val archive on S3 but not the test holdout archive; will try another source."
      return 1
    fi
  fi
  echo "Copying the staged train/val and test-holdout archives from S3 (about 1.06 GB)."
  work="$(mktemp -d "${TMPDIR:-/tmp}/mlp1-s3-data.XXXXXX")"
  aws s3 cp --region "${AWS_REGION}" "${trainval_uri}" "${work}/trainval.tar"
  aws s3 cp --region "${AWS_REGION}" "${holdout_uri}" "${work}/test_holdout.tar"
  check_archive_sha256 "${work}/trainval.tar" "${TRAINVAL_TAR_SHA256}" "trainval.tar" || true
  check_archive_sha256 "${work}/test_holdout.tar" "${TEST_HOLDOUT_TAR_SHA256}" "test_holdout.tar" || true
  mkdir -p "${work}/extract"
  tar -xf "${work}/trainval.tar" -C "${work}/extract"
  tar -xf "${work}/test_holdout.tar" -C "${work}/extract"
  promote_extracted_tree "${work}/extract"
  rm -rf "${work}"
  return 0
}

fetch_data_from_s3() {
  if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI not found; skipping S3."
    return 1
  fi
  local buckets bucket
  buckets="$(discover_data_buckets | tr '\n' ' ')"
  if [[ -z "${buckets// }" ]]; then
    echo "DATA_BUCKET is not set; skipping S3."
    return 1
  fi
  local count
  count="$(discover_data_buckets | grep -c . || true)"
  echo "Using the S3 bucket from DATA_BUCKET (name not printed)."
  while IFS= read -r bucket; do
    [[ -z "${bucket}" ]] && continue
    DATA_BUCKET="${bucket}"
    export DATA_BUCKET
    if try_s3_prepared_prefix "${bucket}"; then
      return 0
    fi
    if try_s3_known_archives "${bucket}"; then
      return 0
    fi
    echo "This bucket does not have a prepared data folder or the known archives; trying the next match."
  done < <(discover_data_buckets)
  return 1
}

download_data_from_drive() {
  echo "Downloading the course dataset folder from Google Drive (id ${DATASET_DRIVE_ID})."
  echo "This is about 1.0 GB and 7,858 TIFF files; allow 20-60 minutes."
  python -m pip install -q gdown
  mkdir -p "${DATA_ROOT}"
  # --continue lets you re-run this script if the download is interrupted.
  python -m gdown --folder --continue --retries 8 --output "${DATA_ROOT}/" "${DATASET_DRIVE_ID}"
}

unpack_or_download_data() {
  if dataset_is_complete; then
    echo "Dataset already present at ${DATA_ROOT}"
    return
  fi
  if [[ -f "${DATA_TAR}" ]]; then
    echo "Unpacking ${DATA_TAR} into ${DATA_ROOT}"
    mkdir -p "${DATA_ROOT}"
    tar -xzf "${DATA_TAR}" -C "${DATA_ROOT}" --strip-components=1
    dataset_is_complete && return
    echo "The local data tar did not produce a complete dataset; trying S3."
  fi
  if fetch_data_from_s3 && dataset_is_complete; then
    echo "Dataset is in place from S3."
    return
  fi
  echo "S3 transfer did not produce a complete dataset; falling back to Google Drive."
  download_data_from_drive
}

verify_data_counts() {
  python - <<PY
from pathlib import Path
import pandas as pd
root = Path("${DATA_ROOT}")
expected = {"train": (77, 2763), "val": (17, 598), "test": (16, 568)}
problems = []
for split, (n_patients, n_pairs) in expected.items():
    manifest = pd.read_csv(root / f"{split}_manifest.csv")
    n_images = sum(1 for _ in (root / split).glob("*/images/*.tif"))
    n_masks = sum(1 for _ in (root / split).glob("*/masks/*.tif"))
    patients = manifest.patient_id.nunique()
    print(f"{split}: manifest {patients} patients / {len(manifest)} pairs; files {n_images} images, {n_masks} masks")
    if (patients, len(manifest)) != (n_patients, n_pairs):
        problems.append(f"{split} manifest {patients}/{len(manifest)} != {n_patients}/{n_pairs}")
    if n_images != n_pairs or n_masks != n_pairs:
        problems.append(f"{split} files images={n_images} masks={n_masks}, expected {n_pairs}")
    missing = [p for p in list(manifest.image_path) + list(manifest.mask_path) if not (root / p).exists()]
    if missing:
        problems.append(f"{split}: {len(missing)} paths in the manifest are missing, e.g. {missing[:3]}")
if problems:
    raise SystemExit("Dataset check failed:\\n  " + "\\n  ".join(problems))
print("Dataset counts match the course split (77/17/16 patients, 2,763/598/568 slices).")
PY
}

unpack_code
cd "${PROJECT_ROOT}"
unpack_or_download_data
if ! dataset_is_complete; then
  echo "ERROR: dataset is incomplete at ${DATA_ROOT}. Re-run this script to resume the download."
  exit 1
fi
verify_data_counts

echo "=== Installing Python packages (PyTorch comes from the Studio image) ==="
python -m pip install -q -r "${PROJECT_ROOT}/requirements.txt"

echo "=== Unit tests ==="
python -m pytest -q "${PROJECT_ROOT}/tests"

echo "=== Data audit and training-only normalisation ==="
python "${PROJECT_ROOT}/scripts/01_prepare_data.py" --data-root "${DATA_ROOT}" --results-dir "${PROJECT_ROOT}/results"

mkdir -p "${PROJECT_ROOT}/results/logs"
if [[ -f "${PROJECT_ROOT}/results/logs/train_all.pid" ]] && ps -p "$(cat "${PROJECT_ROOT}/results/logs/train_all.pid")" >/dev/null 2>&1; then
  echo "Training is already running (pid $(cat "${PROJECT_ROOT}/results/logs/train_all.pid")). Log: ${TRAIN_LOG}"
  exit 0
fi

echo "=== Starting the 14 training runs across every visible GPU ==="
echo "Log: ${TRAIN_LOG}"
echo "Re-run this script, or just scripts/02_train_all_runs.py, to resume skipped/interrupted runs."

TRAIN_CMD=(python "${PROJECT_ROOT}/scripts/02_train_all_runs.py"
  --data-root "${DATA_ROOT}"
  --results-dir "${PROJECT_ROOT}/results"
  --parallel-gpus)

if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t mlp1_train 2>/dev/null && tmux kill-session -t mlp1_train
  tmux new-session -d -s mlp1_train "cd ${PROJECT_ROOT} && ${TRAIN_CMD[*]} 2>&1 | tee -a ${TRAIN_LOG}; echo exit_code=\$? | tee -a ${TRAIN_LOG}"
  echo "Started in tmux session mlp1_train. Attach with: tmux attach -t mlp1_train"
  sleep 1
  tmux list-sessions
else
  nohup "${TRAIN_CMD[@]}" >> "${TRAIN_LOG}" 2>&1 &
  echo $! > "${PROJECT_ROOT}/results/logs/train_all.pid"
  echo "Started with nohup, pid $(cat "${PROJECT_ROOT}/results/logs/train_all.pid")"
fi

echo
echo "Follow the log with:  tail -f ${TRAIN_LOG}"
echo "When all 14 runs have finished:"
echo "  python scripts/03_select_on_validation.py"
echo "  python scripts/06_make_figures_and_tables.py"
echo "  # only after every method decision is final:"
echo "  python scripts/04_freeze_method.py --confirm-method-is-final"
echo "  python scripts/05_evaluate_test_once.py --confirm-method-is-frozen"
echo "  python scripts/06_make_figures_and_tables.py"
