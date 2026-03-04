#!/usr/bin/env python
import argparse
import time
import redis
import json
import numpy as np
import torch
from rich import print
import os
import mujoco
from mujoco.viewer import launch_passive
import matplotlib.pyplot as plt
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import euler_from_quaternion_torch, quat_rotate_inverse_torch

from data_utils.params import DEFAULT_MIMIC_OBS

try:
    import isaacgym  # noqa: F401
except Exception:
    isaacgym = None


ROBOT_SIM_CONFIGS = {
    "unitree_g1": {
        "xml_file": "../assets/g1/g1_mocap_29dof.xml",
        "robot_base": "pelvis",
        "floating_base": True,
        "base_height_offset": 0.0,
    },
    "unitree_g1_with_hands": {
        "xml_file": "../assets/g1/g1_mocap_29dof.xml",
        "robot_base": "pelvis",
        "floating_base": True,
        "base_height_offset": 0.0,
    },
    "jingchu03_upper_body": {
        "xml_file": "../assets/jingchu03/jingchu03_upper_body.xml",
        "robot_base": "waist_yaw",
        "floating_base": False,
        "base_height_offset": 0.8,
    },
}


def apply_fixed_base_visual_offset(sim_model: mujoco.MjModel, z_offset: float):
    """Apply world-frame z offset for fixed-base upper-body visualization."""
    if abs(z_offset) < 1e-9:
        return

    root_body_id = sim_model.body("waist_roll").id
    sim_model.body_pos[root_body_id, 2] += z_offset

    # Shift world-attached visual mesh geoms (e.g. Robotbase) together.
    for geom_id in range(sim_model.ngeom):
        if (
            sim_model.geom_bodyid[geom_id] == 0
            and sim_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        ):
            sim_model.geom_pos[geom_id, 2] += z_offset


def map_motion_dof_to_robot(robot_type: str, dof_pos: torch.Tensor) -> torch.Tensor:
    """Map source motion dof to target robot dof."""
    if robot_type in ("unitree_g1", "unitree_g1_with_hands"):
        return dof_pos

    if robot_type != "jingchu03_upper_body":
        raise ValueError(f"robot type {robot_type} not supported")

    src_dofs = dof_pos.shape[-1]
    if src_dofs == 16:
        return dof_pos
    if src_dofs != 29:
        raise ValueError(f"jingchu03_upper_body expects source dof=29 or 16, got {src_dofs}")

    # Source is G1-29DoF motion, target is Jingchu03 upper-body 16DoF.
    # Target order:
    # [waist_roll, waist_yaw, l_shoulder_pitch, l_shoulder_roll, l_shoulder_yaw,
    #  l_elbow_pitch, l_elbow_yaw, l_wrist_pitch, l_wrist_roll,
    #  r_shoulder_pitch, r_shoulder_roll, r_shoulder_yaw,
    #  r_elbow_pitch, r_elbow_yaw, r_wrist_pitch, r_wrist_roll]
    zero_ref = dof_pos[..., 0] * 0.0
    mapped = torch.stack([
        dof_pos[..., 13],  # waist_roll
        dof_pos[..., 12],  # waist_yaw
        dof_pos[..., 15],  # left_shoulder_pitch
        dof_pos[..., 16],  # left_shoulder_roll
        dof_pos[..., 17],  # left_shoulder_yaw
        dof_pos[..., 18],  # left_elbow_pitch
        zero_ref,          # left_elbow_yaw (no corresponding DoF in G1)
        dof_pos[..., 20],  # left_wrist_pitch
        dof_pos[..., 19],  # left_wrist_roll
        dof_pos[..., 22],  # right_shoulder_pitch
        dof_pos[..., 23],  # right_shoulder_roll
        dof_pos[..., 24],  # right_shoulder_yaw
        dof_pos[..., 25],  # right_elbow_pitch
        zero_ref,          # right_elbow_yaw (no corresponding DoF in G1)
        dof_pos[..., 27],  # right_wrist_pitch
        dof_pos[..., 26],  # right_wrist_roll
    ], dim=-1)
    return mapped


def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_motion_steps,
    robot_type: str = "g1",
    mask_indicator: bool = False
):
    """
    Build the mimic_obs at time-step t_step, referencing the code in MimicRunner.
    """
    device = torch.device("cuda")
    # Build times
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = tar_motion_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()
    
    # Suppose we only have a single motion in the .pkl
    motion_ids = torch.zeros(len(tar_motion_steps), dtype=torch.int, device=device)
    
    # Retrieve motion frames
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local = motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

    # Convert to euler (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion_torch(root_rot, scalar_first=False)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    # Transform velocities to root frame
    root_vel_local = quat_rotate_inverse_torch(root_rot, root_vel, scalar_first=False).reshape(1, -1, 3)
    root_ang_vel_local = quat_rotate_inverse_torch(root_rot, root_ang_vel, scalar_first=False).reshape(1, -1, 3)
    root_vel = root_vel.reshape(1, -1, 3)
    root_ang_vel = root_ang_vel.reshape(1, -1, 3)

    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = map_motion_dof_to_robot(robot_type, dof_pos)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])
    
    # mimic_obs_buf = torch.cat((
    #             root_pos,
    #             roll, pitch, yaw,
    #             # root_vel,
    #             # root_ang_vel,
    #             root_vel_local,
    #             root_ang_vel_local,
    #             dof_pos 
    #         ), dim=-1)[:, 0:1]  # shape (1, 1, ?)
    # print("root_vel_local: ", root_vel_local)
    # Modified for better observability: root_vel_xy + root_pos_z + roll_pitch + yaw_ang_vel + dof_pos
    if mask_indicator:
        mimic_obs_buf = torch.cat((
                    # root position: xy velocity + z position
                    root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
                    root_pos[..., 2:3], # 1 dim (z position)
                    # root rotation: roll/pitch + yaw angular velocity
                    roll, pitch, # 2 dims (roll/pitch orientation)
                    root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
                    dof_pos,
                ), dim=-1)[:, :]  # shape (1, 1, 6 + num_dof)
        # append mask indicator 1
        mask_indicator = torch.ones(1, mimic_obs_buf.shape[1], 1).to(device)
        mimic_obs_buf = torch.cat((mimic_obs_buf, mask_indicator), dim=-1)
    else:
        mimic_obs_buf = torch.cat((
                    # root position: xy velocity + z position
                    root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
                    root_pos[..., 2:3], # 1 dim (z position)
                    # root rotation: roll/pitch + yaw angular velocity
                    roll, pitch, # 2 dims (roll/pitch orientation)
                    root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
                    dof_pos,
                ), dim=-1)[:, :]  # shape (1, 1, 6 + num_dof)

    # print("root height: ", root_pos[..., 2:3].detach().cpu().numpy().squeeze())
    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    
    return mimic_obs_buf.detach().cpu().numpy().squeeze(), root_pos.detach().cpu().numpy().squeeze(), \
        root_rot.detach().cpu().numpy().squeeze(), dof_pos.detach().cpu().numpy().squeeze(), \
            root_vel.detach().cpu().numpy().squeeze(), root_ang_vel.detach().cpu().numpy().squeeze()


def main(args, xml_file, robot_base, floating_base, base_height_offset):
    # Remote control state  
    motion_started = False if args.use_remote_control else True
    
    if args.use_remote_control:
        print("[Motion Server] Remote control enabled. Waiting for start signal from robot controller...")

    viewer = None
    sim_model = None
    sim_data = None
    if args.vis:
        sim_model = mujoco.MjModel.from_xml_path(xml_file)
        if not floating_base:
            apply_fixed_base_visual_offset(sim_model, base_height_offset)
        sim_data = mujoco.MjData(sim_model)
        viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)
            
    # 1. Connect to Redis
    redis_ip = args.redis_ip
    # redis_client = redis.Redis(host="localhost", port=6379, db=0)
    # redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0)
    # redis_client = redis.Redis(host="192.168.110.24", port=6379, db=0)
    redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
    redis_client.ping()


    # 2. Load motion library
    device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_lib = MotionLib(args.motion_file, device=device)
    
    # 3. Prepare the steps array
    tar_motion_steps = [int(x.strip()) for x in args.steps.split(",")]
    tar_motion_steps_tensor = torch.tensor(tar_motion_steps, device=device, dtype=torch.int)

    # 4. Loop over time steps and publish mimic obs
    control_dt = 0.02
    
    # 4.5 Extract start frame for end frame if option is enabled
    start_frame_mimic_obs = None
    if args.send_start_frame_as_end_frame:
        start_frame_mimic_obs, _, _, _, _, _ = build_mimic_obs(
            motion_lib=motion_lib,
            t_step=0,
            control_dt=control_dt,
            tar_motion_steps=tar_motion_steps_tensor,
            robot_type=args.robot
        )
    # compute num_steps based on motion length
    motion_id = torch.tensor([0], device=device, dtype=torch.long)
    motion_length = motion_lib.get_motion_length(motion_id)
    num_steps = int(motion_length / control_dt)
    
    print(f"[Motion Server] Streaming for {num_steps} steps at dt={control_dt:.3f} seconds...")

    last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
    
    # Helper function to check remote control signals
    def check_remote_control_signals():
        if not args.use_remote_control:
            return True, False  # motion_active, should_exit
        
        try:
            # Check for start signal (B button from robot controller)
            start_signal = redis_client.get("motion_start_signal")
            start_pressed = start_signal == b"1" if start_signal else False
            
            # Check for exit signal (Select button from robot controller)
            exit_signal = redis_client.get("motion_exit_signal") 
            exit_pressed = exit_signal == b"1" if exit_signal else False
            
            return start_pressed, exit_pressed
        except Exception as e:
            return False, False
    
    if args.use_remote_control:
        # reset start and exit signal to 0
        redis_client.set("motion_start_signal", "0")
        redis_client.set("motion_exit_signal", "0")
    
    try:
        # for t_step in range(num_steps):
        t_step = 0
        while True:
            t0 = time.time()
            
            # Handle remote control logic
            if args.use_remote_control:
                # Check remote control signals
                start_pressed, exit_pressed = check_remote_control_signals()

                if exit_pressed:
                    print("[Motion Server] Exit signal received, stopping...")
                    break
                    
                if not motion_started and start_pressed:
                    print("[Motion Server] Start signal received, beginning motion...")
                    motion_started = True
                elif not motion_started:
                    # Keep sending default pose while waiting for start signal
                    idle_mimic_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame and start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[args.robot]
                    redis_client.set(f"action_body_{args.robot}", json.dumps(idle_mimic_obs.tolist()))
                    redis_client.set(f"action_hand_left_{args.robot}", json.dumps(np.zeros(7).tolist()))
                    redis_client.set(f"action_hand_right_{args.robot}", json.dumps(np.zeros(7).tolist()))

                    # Sleep and continue to next iteration
                    elapsed = time.time() - t0
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
                    continue

            # Build a mimic obs from the motion library
            mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs(
                motion_lib=motion_lib,
                t_step=t_step,
                control_dt=control_dt,
                tar_motion_steps=tar_motion_steps_tensor,
                robot_type=args.robot
            )   
            
            # Convert to JSON (list) to put into Redis
            mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
            redis_client.set(f"action_body_{args.robot}", json.dumps(mimic_obs_list))
            redis_client.set(f"action_hand_left_{args.robot}", json.dumps(np.zeros(7).tolist()))
            redis_client.set(f"action_hand_right_{args.robot}", json.dumps(np.zeros(7).tolist()))
            redis_client.set(f"action_neck_{args.robot}", json.dumps(np.zeros(2).tolist()))
            last_mimic_obs = mimic_obs
            
            # Print or log it
            print(f"Step {t_step:4d} => mimic_obs shape = {mimic_obs.shape} published...", end="\r")

            if args.vis:
                if floating_base:
                    sim_data.qpos[:3] = root_pos
                    root_rot = root_rot[[3, 0, 1, 2]]
                    sim_data.qpos[3:7] = root_rot
                    sim_data.qpos[7:7 + dof_pos.shape[0]] = dof_pos
                else:
                    sim_data.qpos[:dof_pos.shape[0]] = dof_pos
                mujoco.mj_forward(sim_model, sim_data)
                robot_base_pos = sim_data.xpos[sim_model.body(robot_base).id]
                viewer.cam.lookat = robot_base_pos
                # set distance to pelvis
                viewer.cam.distance = 2.0
                viewer.sync()
            
            t_step += 1
            if t_step >= num_steps:
                break
            # Sleep to maintain real-time pace
            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    
      
    except Exception as e:
        print(f"[Motion Server] Error: {e}")
        print("[Motion Server] Keyboard interrupt. Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        target_mimic_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame and start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[args.robot]
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (target_mimic_obs - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_body_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_body_{args.robot}", json.dumps(target_mimic_obs.tolist()))
        last_mimic_obs = target_mimic_obs
        if viewer is not None:
            viewer.close()
        time.sleep(0.5)
        exit()
    finally:
        print("[Motion Server] Exiting...Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        target_mimic_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame and start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[args.robot]
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (target_mimic_obs - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_body_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_body_{args.robot}", json.dumps(target_mimic_obs.tolist()))
        last_mimic_obs = target_mimic_obs
        if viewer is not None:
            viewer.close()
        time.sleep(0.5)
        exit()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", help="Path to your *.pkl motion file for MotionLib", 
                        default="../motion_data/OMOMO_g1_GMR/sub1_clothesstand_067.pkl"
                        )
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1_with_hands",
        choices=list(ROBOT_SIM_CONFIGS.keys()),
    )
    parser.add_argument("--steps", type=str,
                        # default="1,3,5,10,15,20,30,40,50",
                        default="1",
                        help="Comma-separated steps for future frames (tar_motion_steps)")
    parser.add_argument("--vis", action="store_true", help="Visualize the motion")
    parser.add_argument("--use_remote_control", action="store_true", help="Use remote control signals from robot controller")
    parser.add_argument("--send_start_frame_as_end_frame", action="store_true", help="Use motion's first frame as end frame instead of default pose")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP")
    args = parser.parse_args()

    args.vis = True
    

    print("Robot type: ", args.robot)
    print("Motion file: ", args.motion_file)
    print("Steps: ", args.steps)
    
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    robot_cfg = ROBOT_SIM_CONFIGS[args.robot]
    xml_file = f"{HERE}/{robot_cfg['xml_file']}"
    robot_base = robot_cfg["robot_base"]
    floating_base = robot_cfg["floating_base"]
    base_height_offset = robot_cfg.get("base_height_offset", 0.0)

    main(args, xml_file, robot_base, floating_base, base_height_offset)
