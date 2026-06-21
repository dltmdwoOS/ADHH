#!/usr/bin/env bash
# 실행 방법: bash download_coco.sh false true true => validation, annotations 다운로드, train은 건너뜀

set -euo pipefail

BASE_DIR="$(pwd)"
TARGET_DIR="${BASE_DIR}/dataset/coco"
DOWNLOAD_TRAIN="$1"
DOWNLOAD_VAL="$2"
DOWNLOAD_ANNOTATIONS="$3"

echo "Target directory: ${TARGET_DIR}"
echo "Download: Train: ${DOWNLOAD_TRAIN}, Val: ${DOWNLOAD_VAL}, Annotations: ${DOWNLOAD_ANNOTATIONS}"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

URLS=()

if [ "${DOWNLOAD_TRAIN}" = "true" ]; then
  URLS+=("http://images.cocodataset.org/zips/train2014.zip")
fi

if [ "${DOWNLOAD_VAL}" = "true" ]; then
  URLS+=("http://images.cocodataset.org/zips/val2014.zip")
fi

if [ "${DOWNLOAD_ANNOTATIONS}" = "true" ]; then
  URLS+=("http://images.cocodataset.org/annotations/annotations_trainval2014.zip")
fi

command -v wget >/dev/null 2>&1 || { echo "wget이 설치되어 있지 않습니다. sudo apt-get install wget 로 설치하세요."; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "unzip이 설치되어 있지 않습니다. sudo apt-get install unzip 로 설치하세요."; exit 1; }

for url in "${URLS[@]}"; do
  file="$(basename "${url}")"

  echo "===== 처리: ${file} ====="
  if [ ! -f "${file}" ]; then
    echo "다운로드 중: ${url}"
    wget -c "${url}"
  else
    echo "이미 존재: ${file}, 다운로드 건너뜀"
  fi

  echo "압축 해제 중: ${file}"
  unzip -q -o "${file}"
done

echo "모든 작업이 완료되었습니다. 이미지들은 ${TARGET_DIR}/train2014, ${TARGET_DIR}/val2014에 위치합니다."