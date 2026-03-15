#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_001.pkl"

motion_file="${1:-${default_motion_file}}"
robot="${2:-unitree_g1_with_hands}"
redis_ip="${REDIS_IP:-localhost}"

if [[ ! -f "${motion_file}" ]]; then
  echo "Motion file not found: ${motion_file}" >&2
  echo "Usage: bash visualize_g1_motion.sh [motion_file] [robot]" >&2
  exit 1
fi

if [[ "${robot}" != "unitree_g1" && "${robot}" != "unitree_g1_with_hands" ]]; then
  echo "Unsupported robot: ${robot}" >&2
  echo "Supported robots: unitree_g1, unitree_g1_with_hands" >&2
  exit 1
fi

cd "${script_dir}/deploy_real"

conda run -n twist2 python server_motion_lib.py \
  --motion_file "${motion_file}" \
  --robot "${robot}" \
  --redis_ip "${redis_ip}"
