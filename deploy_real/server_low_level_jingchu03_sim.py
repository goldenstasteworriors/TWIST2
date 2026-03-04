import argparse
import json
import os
import time

import mujoco
import mujoco.viewer as mjv
import numpy as np
import redis
from rich import print
from tqdm import tqdm

from data_utils.params import DEFAULT_MIMIC_OBS


class Jingchu03SimController:
    def __init__(
        self,
        xml_file,
        robot_name="jingchu03_upper_body",
        measure_fps=False,
        limit_fps=True,
        control_frequency=100,
        max_steps=None,
    ):
        self.robot_name = robot_name
        self.measure_fps = measure_fps
        self.limit_fps = limit_fps
        self.max_steps = max_steps

        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.redis_pipeline = self.redis_client.pipeline()

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)

        self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        self.viewer.cam.distance = 1.6

        self.num_actions = 16
        self.sim_dt = 0.001
        self.sim_duration = 100000.0
        self.sim_decimation = max(1, int(round(1.0 / (control_frequency * self.sim_dt))))

        self.default_dof_pos = np.array([
            0.0, 0.0,
            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
        ], dtype=np.float32)

        # A moderate PD setting for stable upper-body tracking.
        self.stiffness = np.array([
            120.0, 120.0,
            80.0, 80.0, 80.0, 80.0, 60.0, 50.0, 50.0,
            80.0, 80.0, 80.0, 80.0, 60.0, 50.0, 50.0,
        ], dtype=np.float32)
        self.damping = np.array([
            6.0, 6.0,
            4.0, 4.0, 4.0, 4.0, 3.0, 2.0, 2.0,
            4.0, 4.0, 4.0, 4.0, 3.0, 2.0, 2.0,
        ], dtype=np.float32)
        self.torque_limits = np.array([
            150.0, 150.0,
            100.0, 100.0, 100.0, 100.0, 80.0, 60.0, 60.0,
            100.0, 100.0, 100.0, 100.0, 80.0, 60.0, 60.0,
        ], dtype=np.float32)

        self.default_mimic_obs = DEFAULT_MIMIC_OBS[self.robot_name].astype(np.float32)
        self.pd_target = self.default_dof_pos.copy()

        print("Jingchu03 Sim Controller Configuration:")
        print(f"  XML file: {xml_file}")
        print(f"  num_actions: {self.num_actions}")
        print(f"  sim_dt: {self.sim_dt}")
        print(f"  control_frequency: {control_frequency}")
        print(f"  sim_decimation: {self.sim_decimation}")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.default_dof_pos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _get_action_from_redis(self):
        keys = [
            f"action_body_{self.robot_name}",
            f"action_hand_left_{self.robot_name}",
            f"action_hand_right_{self.robot_name}",
            f"action_neck_{self.robot_name}",
        ]
        for key in keys:
            self.redis_pipeline.get(key)
        result = self.redis_pipeline.execute()

        body = result[0]
        if body is None:
            return self.default_mimic_obs

        try:
            action_mimic = np.array(json.loads(body), dtype=np.float32)
        except Exception:
            return self.default_mimic_obs

        expected_dim = 6 + self.num_actions
        if action_mimic.shape[0] != expected_dim:
            print(f"[warning] action dim mismatch, expected {expected_dim}, got {action_mimic.shape[0]}. fallback default")
            return self.default_mimic_obs

        return action_mimic

    def run(self):
        self.reset()

        initial_state = np.concatenate([np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32), self.default_dof_pos])
        self.redis_pipeline.set(f"state_body_{self.robot_name}", json.dumps(initial_state.tolist()))
        self.redis_pipeline.set(f"state_hand_left_{self.robot_name}", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set(f"state_hand_right_{self.robot_name}", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set(f"state_neck_{self.robot_name}", json.dumps(np.zeros(2).tolist()))
        self.redis_pipeline.execute()

        total_steps = int(self.sim_duration / self.sim_dt)
        if self.max_steps is not None:
            total_steps = min(total_steps, self.max_steps)

        fps_measurements = []
        fps_target = 1000
        last_ctrl_time = None

        pbar = tqdm(range(total_steps), desc="Simulating jingchu03 upper body")
        for i in pbar:
            step_start = time.time()

            dof_pos = self.data.qpos[:self.num_actions].copy()
            dof_vel = self.data.qvel[:self.num_actions].copy()

            if i % self.sim_decimation == 0:
                state_body = np.concatenate([
                    np.zeros(3, dtype=np.float32),
                    np.zeros(2, dtype=np.float32),
                    dof_pos.astype(np.float32),
                ])

                self.redis_pipeline.set(f"state_body_{self.robot_name}", json.dumps(state_body.tolist()))
                self.redis_pipeline.set(f"state_hand_left_{self.robot_name}", json.dumps(np.zeros(7).tolist()))
                self.redis_pipeline.set(f"state_hand_right_{self.robot_name}", json.dumps(np.zeros(7).tolist()))
                self.redis_pipeline.set(f"state_neck_{self.robot_name}", json.dumps(np.zeros(2).tolist()))
                self.redis_pipeline.set("t_state", int(time.time() * 1000))
                self.redis_pipeline.execute()

                action_mimic = self._get_action_from_redis()
                self.pd_target = action_mimic[-self.num_actions:]

                current_ctrl_time = time.time()
                if last_ctrl_time is not None:
                    ctrl_fps = 1.0 / max(1e-6, current_ctrl_time - last_ctrl_time)
                    if self.measure_fps:
                        fps_measurements.append(ctrl_fps)
                        if len(fps_measurements) >= fps_target:
                            print("\n=== Control FPS Results ===")
                            print(f"Average FPS: {np.mean(fps_measurements):.2f}")
                            print(f"Max FPS: {np.max(fps_measurements):.2f}")
                            print(f"Min FPS: {np.min(fps_measurements):.2f}")
                            print(f"Std FPS: {np.std(fps_measurements):.2f}")
                            print("===========================\n")
                            fps_measurements.clear()
                last_ctrl_time = current_ctrl_time

                body_id = self.model.body("waist_yaw").id
                self.viewer.cam.lookat = self.data.xpos[body_id]
                self.viewer.sync()

            torque = (self.pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)

            if self.limit_fps:
                elapsed = time.time() - step_start
                if elapsed < self.sim_dt:
                    time.sleep(self.sim_dt - elapsed)

        self.viewer.close()
        print("Simulation finished.")


def main():
    parser = argparse.ArgumentParser(description="Low-level sim server for jingchu03 upper-body robot")
    parser.add_argument(
        "--xml",
        type=str,
        default="../assets/jingchu03/jingchu03_upper_body.xml",
        help="Path to MuJoCo XML file",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="jingchu03_upper_body",
        choices=["jingchu03_upper_body"],
        help="Robot redis key prefix",
    )
    parser.add_argument("--measure_fps", help="Measure control FPS", default=0, type=int)
    parser.add_argument("--limit_fps", help="Limit sim FPS with sleep", default=1, type=int)
    parser.add_argument("--control_frequency", help="Control frequency", default=100, type=int)
    parser.add_argument("--max_steps", help="Maximum sim steps", default=None, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.xml):
        print(f"Error: XML file {args.xml} does not exist")
        return

    print("Starting jingchu03 simulation controller...")
    print(f"  XML file: {args.xml}")
    print(f"  Robot: {args.robot}")
    print(f"  Measure FPS: {args.measure_fps}")
    print(f"  Limit FPS: {args.limit_fps}")
    print(f"  Control frequency: {args.control_frequency}")
    print(f"  Max steps: {args.max_steps}")

    controller = Jingchu03SimController(
        xml_file=args.xml,
        robot_name=args.robot,
        measure_fps=bool(args.measure_fps),
        limit_fps=bool(args.limit_fps),
        control_frequency=args.control_frequency,
        max_steps=args.max_steps,
    )
    controller.run()


if __name__ == "__main__":
    main()
