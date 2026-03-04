#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_001.pkl"

cd deploy_real

redis_ip="localhost"

python server_motion_lib.py \
    --motion_file "${motion_file}" \
    --robot jingchu03_upper_body \
    --vis \
    --redis_ip "${redis_ip}"
