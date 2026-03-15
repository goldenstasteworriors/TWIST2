import argparse
import faulthandler

import isaacgym  # noqa: F401
import torch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse
from termcolor import cprint

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.base.humanoid_char import (
    convert_to_global_root_body_pos,
    convert_to_local_root_body_pos,
)
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.gym_utils import get_args, task_registry


def set_debug_cfg(env_cfg):
    env_cfg.env.num_envs = 1
    env_cfg.env.debug_viz = True
    env_cfg.env.rand_reset = False
    env_cfg.env.episode_length_s = 60

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False

    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False


def update_reference_buffers(env, motion_time):
    motion_ids = env._motion_ids
    motion_times = torch.full_like(motion_ids, motion_time, dtype=torch.float)
    (
        root_pos,
        root_rot,
        root_vel,
        root_ang_vel,
        dof_pos,
        dof_vel,
        local_body_pos,
        root_pos_delta_local,
        root_rot_delta_local,
    ) = env._motion_lib.calc_motion_frame(motion_ids, motion_times)

    dof_pos = env._map_motion_dof(dof_pos)
    dof_vel = env._map_motion_dof(dof_vel)
    root_pos[:, 2] += env.cfg.motion.height_offset
    root_pos[:, :2] += env.episode_init_origin[:, :2]

    env._ref_root_pos[:] = root_pos
    env._ref_root_rot[:] = root_rot
    env._ref_root_vel[:] = root_vel
    env._ref_root_ang_vel[:] = root_ang_vel
    env._ref_dof_pos[:] = dof_pos
    env._ref_dof_vel[:] = dof_vel
    env._ref_body_pos[:] = convert_to_global_root_body_pos(
        root_pos=root_pos,
        root_rot=root_rot,
        body_pos=local_body_pos,
    )
    env._ref_root_pos_delta_local[:] = root_pos_delta_local
    env._ref_root_rot_delta_local[:] = root_rot_delta_local


def set_robot_to_reference(env, env_ids):
    zero_dof_vel = torch.zeros_like(env._ref_dof_vel)
    zero_root_vel = torch.zeros_like(env._ref_root_vel)
    zero_root_ang_vel = torch.zeros_like(env._ref_root_ang_vel)

    env._reset_dofs(env_ids, env._ref_dof_pos, zero_dof_vel)
    env._reset_root_states(
        env_ids=env_ids,
        root_vel=zero_root_vel,
        root_quat=env._ref_root_rot,
        root_pos=env._ref_root_pos,
        root_ang_vel=zero_root_ang_vel,
    )

    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)


def update_base_buffers(env):
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)


def compute_keybody_error_per_part(env):
    key_body_pos = env.rigid_body_states[:, env._key_body_ids, 0:3]
    key_body_pos = key_body_pos - env.root_states[:, 0:3].unsqueeze(1)

    tar_key_body_pos = env._ref_body_pos[:, env._key_body_ids_motion, :]
    tar_key_body_pos = tar_key_body_pos - env._ref_root_pos.unsqueeze(1)

    if not env.global_obs:
        base_yaw_quat = quat_from_euler_xyz(0 * env.yaw, 0 * env.yaw, env.yaw)
        key_body_pos = convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)

        _, _, ref_yaw = euler_from_quaternion(env._ref_root_rot)
        ref_yaw_quat = quat_from_euler_xyz(0 * ref_yaw, 0 * ref_yaw, ref_yaw)
        tar_key_body_pos = convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)

    return torch.mean(torch.abs(key_body_pos - tar_key_body_pos), dim=-1)


def parse_extra_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--motion_index", type=int, default=0, help="Motion index when using a yaml motion dataset.")
    parser.add_argument("--loop_reference", action="store_true", help="Loop the selected reference motion.")
    parser.add_argument("--print_every", type=int, default=30, help="Print key body errors every N frames.")
    return parser.parse_known_args()[0]


def main():
    faulthandler.enable()
    args = get_args()
    extra_args = parse_extra_args()

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    set_debug_cfg(env_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    num_motions = env._motion_lib.num_motions()
    motion_index = min(max(extra_args.motion_index, 0), num_motions - 1)
    env._motion_ids[:] = motion_index
    env._motion_time_offsets[:] = 0.0

    env_ids = torch.arange(env.num_envs, device=env.device)
    motion_name = env.motion_names[motion_index]
    motion_length = env._motion_lib.get_motion_length(env._motion_ids[:1])[0].item()

    cprint(f"[debug_motion_keypoints] motion_index={motion_index}", "green")
    cprint(f"[debug_motion_keypoints] motion_name={motion_name}", "green")
    cprint(f"[debug_motion_keypoints] motion_length={motion_length:.3f}s", "green")
    cprint(f"[debug_motion_keypoints] key_bodies={env.cfg.motion.key_bodies}", "green")

    frame_idx = 0
    motion_time = 0.0
    while True:
        update_reference_buffers(env, motion_time)
        set_robot_to_reference(env, env_ids)
        update_base_buffers(env)

        if env.viewer and env.enable_viewer_sync and env.debug_viz:
            env.gym.clear_lines(env.viewer)
            env.draw_key_bodies_actual()
            env.draw_key_bodies_motion()

        if extra_args.print_every > 0 and frame_idx % extra_args.print_every == 0:
            body_errors = compute_keybody_error_per_part(env)[0].detach().cpu().tolist()
            error_str = ", ".join(
                f"{name}: {err:.5f}"
                for name, err in zip(env.cfg.motion.key_bodies, body_errors)
            )
            cprint(f"[keybody_error] t={motion_time:.3f}s | {error_str}", "cyan")

        env.render()

        frame_idx += 1
        motion_time += env.dt
        if motion_time >= motion_length:
            if extra_args.loop_reference:
                motion_time = 0.0
            else:
                break


if __name__ == "__main__":
    main()
