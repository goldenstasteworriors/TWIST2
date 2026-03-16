#!/bin/bash

source ~/miniconda3/bin/activate gmr_for_twist2

cd "$(dirname "$0")/deploy_real"

python xrobot_headset_controller_to_smplx_upper_body.py \
    --vis \
    --report_every 10 \
    "$@"
