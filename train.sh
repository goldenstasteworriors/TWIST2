#!/bin/bash

# Usage: bash train.sh <experiment_id> <device> [robot_name] [task_suffix] [teacher_exptid]

# bash train.sh 1103_twist2 cuda:0
# bash train.sh 0304_jingchu03_priv cuda:0 jingchu03 priv_mimic
# bash train.sh 0304_jingchu03_stu cuda:0 jingchu03 stu_future 0304_jingchu03_priv


cd legged_gym/legged_gym/scripts

robot_name=${3:-g1}
task_suffix=${4:-stu_future}
teacher_exptid=${5:-None}
exptid=$1
device=$2

task_name="${robot_name}_${task_suffix}"
proj_name="${task_name}"


# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "${teacher_exptid}" \
                # --resume \
                # --debug \
