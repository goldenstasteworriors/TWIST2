#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="/tmp/twist2_play_compare"
mkdir -p "${log_dir}"

# 默认用你本地已下载的数据集动作，找不到则回退到项目示例动作。
motion_file="/home/ykj/Downloads/dataset/TWIST2_full/OMOMO_g1_GMR/sub1_clothesstand_000.pkl"
if [[ ! -f "${motion_file}" ]]; then
  motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_001.pkl"
fi

export DISPLAY="${DISPLAY:-:1}"

echo "[TWIST2] motion_file=${motion_file}"
echo "[TWIST2] DISPLAY=${DISPLAY}"
echo "[TWIST2] logs=${log_dir}"

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "${script_dir}/deploy_real"

conda run -n twist2 python server_motion_lib.py \
  --motion_file "${motion_file}" \
  --robot unitree_g1_with_hands \
  --vis \
  --redis_ip localhost > "${log_dir}/g1.log" 2>&1 &
pid_g1=$!

conda run -n twist2 python server_motion_lib.py \
  --motion_file "${motion_file}" \
  --robot jingchu03_upper_body \
  --vis \
  --redis_ip localhost > "${log_dir}/jingchu03.log" 2>&1 &
pid_jc=$!

wait "${pid_g1}"
wait "${pid_jc}"

echo "[TWIST2] compare playback done."
echo "[TWIST2] g1 log: ${log_dir}/g1.log"
echo "[TWIST2] jingchu03 log: ${log_dir}/jingchu03.log"
