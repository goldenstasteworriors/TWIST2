#!/bin/bash

cd deploy_real

python server_low_level_jingchu03_sim.py \
    --xml ../assets/jingchu03/jingchu03_upper_body.xml \
    --robot jingchu03_upper_body \
    --measure_fps 1 \
    --control_frequency 100 \
    --base_height 0.8 \
    --limit_fps 1
