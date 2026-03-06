
# bash eval.sh 1002_twist2 cuda:1
# bash eval.sh 0304_jingchu03_stu cuda:0 jingchu03 stu_future



script_dir=$(dirname "$(realpath "$0")")
motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_001.pkl"

robot_name=${3:-g1}
task_suffix=${4:-stu_future}
task_name="${robot_name}_${task_suffix}"
proj_name="${task_name}"
exptid=$1
device=$2

cd legged_gym/legged_gym/scripts

echo "Evaluating student policy with future motion support..."
echo "Task: ${task_name}"
echo "Project: ${proj_name}"
echo "Experiment ID: ${exptid}"
echo ""

# Run the evaluation script
python play.py --task "${task_name}" \
               --proj_name "${proj_name}" \
               --teacher_exptid "None" \
               --exptid "${exptid}" \
               --num_envs 1 \
               --record_video \
               --device "${device}" \
               --env.motion.motion_file "${motion_file}" \
               # --checkpoint 13000 \
               # --record_log \
               # --use_jit \
               # --teleop_mode
