#!/usr/bin/env bash

ADHH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ADHH_MODEL_DIR="$(cd -- "${ADHH_SCRIPT_DIR}/.." && pwd)"
ADHH_REPO_ROOT="$(cd -- "${ADHH_MODEL_DIR}/.." && pwd)"
ADHH_WORKSPACE_ROOT="$(cd -- "${ADHH_REPO_ROOT}/.." && pwd)"

cd "${ADHH_MODEL_DIR}"

adhh_default_results_root() {
  printf '%s\n' "${ADHH_REPO_ROOT}/results_deact"
}

adhh_default_data_path() {
  if [[ -d "${ADHH_REPO_ROOT}/dataset/coco/val2014" ]]; then
    printf '%s\n' "${ADHH_REPO_ROOT}/dataset"
  elif [[ -d "${ADHH_WORKSPACE_ROOT}/dataset/coco/val2014" ]]; then
    printf '%s\n' "${ADHH_WORKSPACE_ROOT}/dataset"
  elif [[ -d "${ADHH_REPO_ROOT}/dataset" ]]; then
    printf '%s\n' "${ADHH_REPO_ROOT}/dataset"
  else
    printf '%s\n' "${ADHH_MODEL_DIR}/../dataset"
  fi
}

adhh_default_amber_root() {
  if [[ -d "${ADHH_REPO_ROOT}/third_party/AMBER" ]]; then
    printf '%s\n' "${ADHH_REPO_ROOT}/third_party/AMBER"
  elif [[ -d "${ADHH_WORKSPACE_ROOT}/third_party/AMBER" ]]; then
    printf '%s\n' "${ADHH_WORKSPACE_ROOT}/third_party/AMBER"
  else
    printf '%s\n' "${ADHH_MODEL_DIR}/../third_party/AMBER"
  fi
}

adhh_default_amber_image_folder() {
  if [[ -d "${ADHH_REPO_ROOT}/dataset/AMBER/images" ]]; then
    printf '%s\n' "${ADHH_REPO_ROOT}/dataset/AMBER/images"
  elif [[ -d "${ADHH_WORKSPACE_ROOT}/dataset/AMBER/images" ]]; then
    printf '%s\n' "${ADHH_WORKSPACE_ROOT}/dataset/AMBER/images"
  else
    printf '%s\n' "$(adhh_default_data_path)/AMBER/images"
  fi
}

adhh_python_bin() {
  local candidate="${PYTHON_BIN:-${PYTHON:-python}}"
  if command -v "${candidate}" >/dev/null 2>&1; then
    printf '%s\n' "${candidate}"
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
  else
    printf '%s\n' "${candidate}"
  fi
}
