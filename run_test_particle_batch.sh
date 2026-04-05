#!/usr/bin/env bash
set -euo pipefail

VALUES=(50 100 500 1000 5000 10000 50000 100000)

TARGET=bench_particle

BENCHMARKS=(
  # BM_MultiMap_Create
  # BM_BMI_Create
  # BM_PoolBMI_Create
  # BM_MultiMap_FindPrimary
  # BM_BMI_FindPrimary
  # BM_PoolBMI_FindPrimary
  # BM_MultiMap_Remove
  # BM_BMI_Remove
  # BM_PoolBMI_Remove
  # BM_MultiMap_BulkIterate
  # BM_BMI_BulkIterate
  # BM_PoolBMI_BulkIterate
  BM_MultiMap_MassRange
  # BM_BMI_MassRange
  # BM_PoolBMI_MassRange
  # BM_MultiMap_OrderedIterate
  # BM_BMI_OrderedIterate
  # BM_PoolBMI_OrderedIterate
  # BM_MultiMap_Modify
  # BM_BMI_Modify
  # BM_PoolBMI_Modify
  # BM_MultiMap_LevelWalk
  # BM_BMI_LevelWalk
  # BM_PoolBMI_LevelWalk
  # BM_MultiMap_Mixed
  # BM_BMI_Mixed
  # BM_PoolBMI_Mixed
)

RUN_MODE="${1:-default}"
BENCH_CPU="${BENCH_CPU:-0}"

COMMON_BENCH_ARGS=(
  --benchmark_repetitions=200
  --benchmark_out_format=json
)

run_cmd() {
  local exe="$1"
  shift

  case "$RUN_MODE" in
    default)
      "$exe" "$@"
      ;;
    pinned)
      taskset -c "${BENCH_CPU}" "$exe" "$@"
      ;;
    background)
      taskpolicy -b "$exe" "$@"
      ;;
    *)
      echo "Unknown RUN_MODE: $RUN_MODE"
      echo "Use: default, pinned, or background"
      exit 1
      ;;
  esac
}

find_exe() {
  local build_dir="$1"

  local candidates=(
    "${build_dir}/${TARGET}"
    "${build_dir}/src/${TARGET}"
    "${build_dir}/bench/${TARGET}"
    "${build_dir}/benchmark/${TARGET}"
    "${build_dir}/Debug/${TARGET}"
    "${build_dir}/Release/${TARGET}"
  )

  for p in "${candidates[@]}"; do
    if [[ -x "$p" ]]; then
      echo "$p"
      return 0
    fi
  done

  return 1
}

for kn in "${VALUES[@]}"; do
  build_dir="build_kn_${kn}"
  result_dir="${build_dir}/results"

  mkdir -p "${result_dir}"

  echo "============================================================"
  echo "Building ${TARGET} with SET_KN=${kn}"
  echo "============================================================"

  cmake -S . -B "${build_dir}" -DSET_KN_VALUE="${kn}"
  cmake --build "${build_dir}" --target "${TARGET}" -j

  exe="$(find_exe "${build_dir}")" || {
    echo "Could not find executable ${TARGET} under ${build_dir}"
    exit 1
  }

  echo "Using executable: ${exe}"

  for bench in "${BENCHMARKS[@]}"; do
    out_json="${result_dir}/result_${kn}_${bench}.json"
    out_log="${result_dir}/result_${kn}_${bench}.log"

    echo
    echo "--- Running ${bench} with SET_KN=${kn} ---"

    run_cmd "${exe}" \
      "${COMMON_BENCH_ARGS[@]}" \
      "--benchmark_filter=^${bench}$" \
      "--benchmark_out=${out_json}" \
      2>&1 | tee "${out_log}"
  done
done
