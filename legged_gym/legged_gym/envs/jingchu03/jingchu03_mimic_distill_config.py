from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO


JINGCHU03_NUM_ACTIONS = 16
JINGCHU03_NUM_KEY_BODIES = 5

class Jingchu03MimicPrivCfg(G1MimicPrivCfg):
    class env(G1MimicPrivCfg.env):
        tar_motion_steps_priv = G1MimicPrivCfg.env.tar_motion_steps_priv
        tar_motion_steps = G1MimicPrivCfg.env.tar_motion_steps
        history_len = G1MimicPrivCfg.env.history_len

        num_actions = JINGCHU03_NUM_ACTIONS
        obs_type = 'priv'
        pose_termination_dist = 1.0

        n_priv_latent = 4 + 1 + 2 * num_actions
        n_proprio = 3 + 2 + 3 * num_actions
        n_priv_mimic_obs = len(tar_motion_steps_priv) * (21 + num_actions + 3 * JINGCHU03_NUM_KEY_BODIES)
        n_mimic_obs_single = 6 + num_actions
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_priv_info = 3 + 3 + 4 + 3 * JINGCHU03_NUM_KEY_BODIES + 2 + 4 + 1 + 2 * num_actions

        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_obs_single

        num_observations = n_priv_obs_single
        num_privileged_obs = n_priv_obs_single

        dof_err_w = [0.0] + [1.0] * (num_actions - 1)

    class init_state(G1MimicPrivCfg.init_state):
        pos = [0.0, 0.0, 0.8]
        default_joint_angles = {
            'waist_roll': 0.0,
            'waist_yaw': 0.0,
            'left_shoulder_pitch': 0.0,
            'left_shoulder_roll': 0.4,
            'left_shoulder_yaw': 0.0,
            'left_elbow_pitch': 1.2,
            'left_elbow_yaw': 0.0,
            'left_wrist_pitch': 0.0,
            'left_wrist_roll': 0.0,
            'right_shoulder_pitch': 0.0,
            'right_shoulder_roll': 0.4,
            'right_shoulder_yaw': 0.0,
            'right_elbow_pitch': 1.2,
            'right_elbow_yaw': 0.0,
            'right_wrist_pitch': 0.0,
            'right_wrist_roll': 0.0,
        }

    class control(G1MimicPrivCfg.control):
        stiffness = {
            'waist': 150,
            'shoulder': 40,
            'elbow': 40,
            'wrist': 40,
        }
        damping = {
            'waist': 4,
            'shoulder': 5,
            'elbow': 5,
            'wrist': 5,
        }

    class asset(G1MimicPrivCfg.asset):
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/jingchu03/urdf_jc01.urdf'

        torso_name: str = 'Robotbase'
        chest_name: str = 'waist_yaw'

        thigh_name: str = 'shoulder_pitch'
        shank_name: str = 'elbow'
        foot_name: str = 'wrist_roll'
        waist_name: list = ['waist_roll', 'waist_yaw']
        upper_arm_name: str = 'shoulder'
        lower_arm_name: str = 'elbow'
        hand_name: list = ['left_gripper', 'right_gripper']

        feet_bodies = ['left_wrist_roll', 'right_wrist_roll']
        n_lower_body_dofs: int = 0
        disable_dof_vel_indices = []
        ankle_dof_indices = []
        waist_dof_indices = [0, 1]
        fixed_dof_indices = [0]
        reset_to_hard_dof_limits = True

        penalize_contacts_on = []
        terminate_after_contacts_on = []

        dof_armature = [0.01] * JINGCHU03_NUM_ACTIONS
        collapse_fixed_joints = False
        fix_base_link = True

    class rewards(G1MimicPrivCfg.rewards):
        ignore_dof_pos_limit_indices = [0, 1]

        class scales(G1MimicPrivCfg.rewards.scales):
            tracking_root_translation_z = 0.0
            tracking_root_rotation = 0.0
            tracking_root_linear_vel = 0.0
            tracking_root_angular_vel = 0.0
            tracking_keybody_pos_global = 0.0
            feet_slip = 0.0
            feet_contact_forces = 0.0
            feet_stumble = 0.0
            feet_air_time = 0.0
            dof_torque_limits = 0.0
            ankle_dof_acc = 0.0
            ankle_dof_vel = 0.0

    class evaluations(G1MimicPrivCfg.evaluations):
        tracking_joint_dof = True
        tracking_joint_vel = True
        tracking_root_translation = False
        tracking_root_rotation = False
        tracking_root_vel = False
        tracking_root_ang_vel = False
        tracking_keybody_pos = True
        tracking_root_pose_delta_local = False
        tracking_root_rotation_delta_local = False

    class motion(G1MimicPrivCfg.motion):
        key_bodies = [
            'left_wrist_roll',
            'right_wrist_roll',
            'left_elbow_pitch',
            'right_elbow_pitch',
            'waist_yaw',
        ]
        upper_key_bodies = [
            'left_wrist_roll',
            'right_wrist_roll',
            'left_elbow_pitch',
            'right_elbow_pitch',
        ]

        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/twist2_dataset.yaml"
        height_offset = 0.0
        use_adaptive_pose_termination = True


class Jingchu03MimicPrivCfgPPO(G1MimicPrivCfgPPO):
    class runner(G1MimicPrivCfgPPO.runner):
        experiment_name = 'jingchu03_priv_mimic'

    class policy(G1MimicPrivCfgPPO.policy):
        action_std = [0.4] * 2 + [0.5] * 14
