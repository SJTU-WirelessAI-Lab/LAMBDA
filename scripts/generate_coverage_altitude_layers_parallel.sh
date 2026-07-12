#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SCENARIO="${SCENARIO:-block1_4p9g}"
SIM="${SIM:-sub6_4p9_clear}"
XML_SCENE_PATH="${XML_SCENE_PATH:-region_files/San Francisco/Full SF/fullsf_0-10G.xml}"
SITES_CSV="${SITES_CSV:-configs/full_sf_recommended_sites.csv}"
BASE_STATION_NAMES="${BASE_STATION_NAMES:-BS1,BS2,BS3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data_export/coverage/fullsf_4p9g_altitude_layers_parallel}"
ALTITUDES="${ALTITUDES:-50 80 120}"

SIZE_M="${SIZE_M:-2000}"
STEP_M="${STEP_M:-100}"
CENTER_M="${CENTER_M:-}"
TX_POWER_DBM="${TX_POWER_DBM:-30}"
BANDWIDTH_HZ="${BANDWIDTH_HZ:-20000000}"
NOISE_FIGURE_DB="${NOISE_FIGURE_DB:-7}"
MAX_DEPTH="${MAX_DEPTH:-5}"
MI_VARIANT="${MI_VARIANT:-cuda_ad_mono_polarized}"
SEED="${SEED:-1000}"
LIMIT="${LIMIT:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
DIFFRACTION="${DIFFRACTION:-1}"
DIFFUSE="${DIFFUSE:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
INCLUDE_OPENGROUND="${INCLUDE_OPENGROUND:-0}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
SHARDS_PER_ALTITUDE="${SHARDS_PER_ALTITUDE:-3}"
read -r -a GPU_IDS <<< "${GPUS}"
GPU_COUNT="${#GPU_IDS[@]}"
MAX_PARALLEL="${MAX_PARALLEL:-${GPU_COUNT}}"

if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "[error] GPUS must contain at least one GPU id." >&2
  exit 2
fi
if [[ "${SHARDS_PER_ALTITUDE}" -lt 1 ]]; then
  echo "[error] SHARDS_PER_ALTITUDE must be >= 1." >&2
  exit 2
fi
if [[ "${MAX_PARALLEL}" -lt 1 ]]; then
  echo "[error] MAX_PARALLEL must be >= 1." >&2
  exit 2
fi

if [[ "${SKIP_MERGE}" != "1" ]]; then
  MERGE_ARGS=()
  if [[ "${INCLUDE_OPENGROUND}" == "1" ]]; then
    MERGE_ARGS+=(--include-openground)
  fi
  python scripts/merge_sf_region_xml.py "${MERGE_ARGS[@]}"
fi

mkdir -p "${OUTPUT_ROOT}"
SHARD_ROOT="${SHARD_ROOT:-${OUTPUT_ROOT}/_shards}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
FAIL_DIR="${OUTPUT_ROOT}/_failures"
mkdir -p "${SHARD_ROOT}" "${LOG_DIR}" "${FAIL_DIR}"

COMMON_ARGS=(
  --scenario "${SCENARIO}"
  --sim "${SIM}"
  --xml-scene-path "${XML_SCENE_PATH}"
  --sites-csv "${SITES_CSV}"
  --base-station-names "${BASE_STATION_NAMES}"
  --size-m "${SIZE_M}"
  --step-m "${STEP_M}"
  --tx-power-dbm "${TX_POWER_DBM}"
  --bandwidth-hz "${BANDWIDTH_HZ}"
  --noise-figure-db "${NOISE_FIGURE_DB}"
  --max-depth "${MAX_DEPTH}"
  --mi-variant "${MI_VARIANT}"
  --seed "${SEED}"
  --progress-every "${PROGRESS_EVERY}"
)

if [[ -n "${CENTER_M}" ]]; then
  COMMON_ARGS+=(--center-m "${CENTER_M}")
fi

if [[ "${DIFFRACTION}" == "1" ]]; then
  COMMON_ARGS+=(--diffraction)
fi

if [[ "${DIFFUSE}" == "1" ]]; then
  COMMON_ARGS+=(--diffuse)
fi

if [[ "${LIMIT}" != "0" ]]; then
  COMMON_ARGS+=(--limit "${LIMIT}")
fi

run_shard() {
  local altitude_m="$1"
  local shard_index="$2"
  local gpu_id="$3"
  local shard_name
  printf -v shard_name "shard_%03d" "${shard_index}"
  local out_dir="${SHARD_ROOT}/alt_${altitude_m}m/${shard_name}"
  local log_path="${LOG_DIR}/alt_${altitude_m}m_${shard_name}_gpu${gpu_id}.log"

  mkdir -p "${out_dir}"
  echo "[start] alt=${altitude_m} shard=${shard_index}/${SHARDS_PER_ALTITUDE} gpu=${gpu_id} log=${log_path}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export LAMBDA_SCENARIO="${SCENARIO}"
    export LAMBDA_SIM="${SIM}"
    python scripts/generate_coverage_map.py \
      "${COMMON_ARGS[@]}" \
      --altitude-m "${altitude_m}" \
      --num-shards "${SHARDS_PER_ALTITUDE}" \
      --shard-index "${shard_index}" \
      --output-dir "${out_dir}"
  ) > "${log_path}" 2>&1 || {
    echo "[failed] alt=${altitude_m} shard=${shard_index} gpu=${gpu_id}; see ${log_path}" >&2
    touch "${FAIL_DIR}/alt_${altitude_m}_shard_${shard_index}.failed"
  }
}

task_id=0
for altitude_m in ${ALTITUDES}; do
  for shard_index in $(seq 0 $((SHARDS_PER_ALTITUDE - 1))); do
    while [[ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]]; do
      wait -n || true
    done
    gpu_id="${GPU_IDS[$((task_id % GPU_COUNT))]}"
    run_shard "${altitude_m}" "${shard_index}" "${gpu_id}" &
    task_id=$((task_id + 1))
  done
done

while [[ "$(jobs -rp | wc -l)" -gt 0 ]]; do
  wait -n || true
done

if compgen -G "${FAIL_DIR}/*.failed" > /dev/null; then
  echo "[error] One or more shard jobs failed:" >&2
  ls -1 "${FAIL_DIR}"/*.failed >&2
  exit 1
fi

for altitude_m in ${ALTITUDES}; do
  echo "[merge] altitude=${altitude_m} m"
  python scripts/merge_coverage_shards.py \
    --input-root "${SHARD_ROOT}/alt_${altitude_m}m" \
    --output-dir "${OUTPUT_ROOT}/alt_${altitude_m}m" \
    --num-shards "${SHARDS_PER_ALTITUDE}"
done

python scripts/summarize_coverage_layers.py \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --altitudes ${ALTITUDES}

echo "[done] parallel altitude-layer coverage maps: ${OUTPUT_ROOT}"
