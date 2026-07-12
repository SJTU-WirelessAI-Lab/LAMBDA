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
OUTPUT_ROOT="${OUTPUT_ROOT:-data_export/coverage/fullsf_4p9g_altitude_layers}"
ALTITUDES="${ALTITUDES:-50 80 120}"

SIZE_M="${SIZE_M:-2000}"
STEP_M="${STEP_M:-50}"
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

if [[ "${SKIP_MERGE}" != "1" ]]; then
  MERGE_ARGS=()
  if [[ "${INCLUDE_OPENGROUND}" == "1" ]]; then
    MERGE_ARGS+=(--include-openground)
  fi
  python scripts/merge_sf_region_xml.py "${MERGE_ARGS[@]}"
fi

mkdir -p "${OUTPUT_ROOT}"

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

for ALTITUDE_M in ${ALTITUDES}; do
  OUT_DIR="${OUTPUT_ROOT}/alt_${ALTITUDE_M}m"
  echo "[coverage] altitude=${ALTITUDE_M} m -> ${OUT_DIR}"
  python scripts/generate_coverage_map.py \
    "${COMMON_ARGS[@]}" \
    --altitude-m "${ALTITUDE_M}" \
    --output-dir "${OUT_DIR}"
done

python scripts/summarize_coverage_layers.py \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --altitudes ${ALTITUDES}

echo "[done] altitude-layer coverage maps: ${OUTPUT_ROOT}"
