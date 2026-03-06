from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym.envs.g1.g1_mimic_future_config import TAR_MOTION_STEPS_FUTURE

from .jingchu03_mimic_distill_config import Jingchu03MimicPrivCfg, Jingchu03MimicPrivCfgPPO


class Jingchu03MimicStuFutureCfg(Jingchu03MimicPrivCfg):
    class env(Jingchu03MimicPrivCfg.env):
        obs_type = 'student_future'

        tar_motion_steps = [0]
        tar_motion_steps_future = TAR_MOTION_STEPS_FUTURE

        n_mimic_obs_single = 6 + Jingchu03MimicPrivCfg.env.num_actions
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = Jingchu03MimicPrivCfg.env.n_proprio

        n_future_obs_single = 6 + Jingchu03MimicPrivCfg.env.num_actions
        n_future_obs = len(tar_motion_steps_future) * n_future_obs_single

        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (Jingchu03MimicPrivCfg.env.history_len + 1) + n_future_obs

        enable_force_curriculum = False

        class force_curriculum:
            force_apply_links = ['left_wrist_roll', 'right_wrist_roll']

            force_scale_curriculum = True
            force_scale_initial_scale = 1.0
            force_scale_up_threshold = 210
            force_scale_down_threshold = 200
            force_scale_up = 0.02
            force_scale_down = 0.02
            force_scale_max = 1.0
            force_scale_min = 0.0

            apply_force_x_range = [-40.0, 40.0]
            apply_force_y_range = [-40.0, 40.0]
            apply_force_z_range = [-50.0, 5.0]

            zero_force_prob = [0.25, 0.25, 0.25]
            randomize_force_duration = [10, 50]

            max_force_estimation = True
            use_lpf = False
            force_filter_alpha = 0.05

            only_apply_z_force_when_walking = False
            only_apply_resistance_when_walking = True

    class motion(Jingchu03MimicPrivCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        motion_decompose = False

        motion_dr_enabled = False
        root_position_noise = [0.01, 0.05]
        root_orientation_noise = [0.1, 0.2]
        root_velocity_noise = [0.05, 0.1]
        joint_position_noise = [0.05, 0.1]
        motion_dr_resampling = True

        use_error_aware_sampling = False
        error_sampling_power = 5.0
        error_sampling_threshold = 0.15

    class rewards(Jingchu03MimicPrivCfg.rewards):
        class scales(Jingchu03MimicPrivCfg.rewards.scales):
            action_rate = -0.05


class Jingchu03MimicStuFutureCfgDAgger(Jingchu03MimicStuFutureCfg):
    seed = 1

    class teachercfg(Jingchu03MimicPrivCfgPPO):
        pass

    class runner(Jingchu03MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 30_001
        warm_iters = 100

        save_interval = 500
        experiment_name = 'jingchu03_stu_future'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        teacher_experiment_name = 'test'
        teacher_proj_name = 'jingchu03_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

        save_to_wandb = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005

        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1

        future_weight_decay = 0.95
        future_consistency_loss = 0.1

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.4] * 2 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128

        future_encoder_dims = [256, 256, 128]
        future_attention_heads = 4
        future_dropout = 0.1
        temporal_embedding_dim = 64
        future_latent_dim = 128
        num_future_steps = len(TAR_MOTION_STEPS_FUTURE)

        num_future_observations = Jingchu03MimicStuFutureCfg.env.n_future_obs

        num_experts = 4
        expert_hidden_dims = [256, 128]
        gating_hidden_dim = 128
        moe_temperature = 1.0
        moe_topk = None
        load_balancing_loss_weight = 0.01
