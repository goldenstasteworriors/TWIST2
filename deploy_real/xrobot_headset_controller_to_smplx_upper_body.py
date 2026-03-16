#!/usr/bin/env python3
"""Synthesize upper-body SMPLX-style joints from headset/controllers and retarget with GMR.

This script is intended for the `jingchu03` upper-body pipeline:
1. Read headset and controller poses from XRoboToolkit SDK.
2. Synthesize a sparse SMPLX-style upper-body joint dictionary.
3. Retarget the sparse human pose to `jingchu03` with GMR.
4. Optionally stream the resulting 16-DoF joint targets to Redis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import redis
import xrobotoolkit_sdk as xrt
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_utils.params import DEFAULT_MIMIC_OBS
from general_motion_retargeting import GeneralMotionRetargeting, ROBOT_XML_DICT


SMPLXFrame = Dict[str, Tuple[np.ndarray, np.ndarray]]


class HeadsetControllerUpperBodySynthesizer:
    """Build a sparse SMPLX-style upper-body pose from headset and controller poses."""

    _UNITY_TO_RIGHT_HAND = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    _UNITY_TO_RIGHT_HAND_ROT = R.from_matrix(_UNITY_TO_RIGHT_HAND)
    _IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def __init__(
        self,
        shoulder_width: float = 0.38,
        neck_to_shoulder_center: float = 0.20,
        shoulder_center_to_pelvis: float = 0.42,
        upper_arm_length: float = 0.28,
        lower_arm_length: float = 0.26,
        shoulder_forward_offset: float = 0.02,
        pelvis_forward_offset: float = -0.02,
        spine3_ratio: float = 0.45,
    ) -> None:
        self.shoulder_width = shoulder_width
        self.neck_to_shoulder_center = neck_to_shoulder_center
        self.shoulder_center_to_pelvis = shoulder_center_to_pelvis
        self.upper_arm_length = upper_arm_length
        self.lower_arm_length = lower_arm_length
        self.shoulder_forward_offset = shoulder_forward_offset
        self.pelvis_forward_offset = pelvis_forward_offset
        self.spine3_ratio = spine3_ratio

    @staticmethod
    def _is_raw_pose_valid(pose: object) -> bool:
        if pose is None:
            return False
        pose_arr = np.asarray(pose, dtype=np.float64)
        if pose_arr.shape != (7,) or not np.all(np.isfinite(pose_arr)):
            return False
        quat_norm = np.linalg.norm(pose_arr[3:7])
        return quat_norm > 1e-6

    def _transform_unity_pose(self, pose: object) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self._is_raw_pose_valid(pose):
            return None

        pose_arr = np.asarray(pose, dtype=np.float64)
        pos_unity = pose_arr[:3]
        quat_xyzw = pose_arr[3:7]
        quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)

        rot_unity = R.from_quat(quat_xyzw)
        rot_world = self._UNITY_TO_RIGHT_HAND_ROT * rot_unity
        pos_world = pos_unity @ self._UNITY_TO_RIGHT_HAND.T
        quat_world_wxyz = rot_world.as_quat(scalar_first=True)
        return pos_world, quat_world_wxyz

    @staticmethod
    def _yaw_only_rotation(quat_wxyz: np.ndarray) -> R:
        roll, pitch, yaw = R.from_quat(quat_wxyz, scalar_first=True).as_euler("xyz", degrees=False)
        _ = roll, pitch
        return R.from_euler("z", yaw, degrees=False)

    def _solve_elbow(
        self,
        shoulder: np.ndarray,
        wrist: np.ndarray,
        torso_right: np.ndarray,
        torso_forward: np.ndarray,
        is_left: bool,
    ) -> np.ndarray:
        arm_vec = wrist - shoulder
        dist = float(np.linalg.norm(arm_vec))
        if dist < 1e-6:
            return shoulder + np.array([0.0, 0.0, -self.upper_arm_length], dtype=np.float64)

        min_reach = abs(self.upper_arm_length - self.lower_arm_length) + 1e-6
        max_reach = self.upper_arm_length + self.lower_arm_length - 1e-6
        reach = float(np.clip(dist, min_reach, max_reach))
        arm_dir = arm_vec / dist

        a = (self.upper_arm_length**2 - self.lower_arm_length**2 + reach**2) / (2.0 * reach)
        h_sq = max(self.upper_arm_length**2 - a**2, 0.0)
        base_point = shoulder + arm_dir * a

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        side_axis = -torso_right if is_left else torso_right
        bias = -1.0 * world_up + 0.30 * side_axis - 0.10 * torso_forward
        bias = bias - arm_dir * np.dot(bias, arm_dir)
        bias_norm = np.linalg.norm(bias)
        if bias_norm < 1e-6:
            bias = np.cross(arm_dir, side_axis)
            bias_norm = np.linalg.norm(bias)
        if bias_norm < 1e-6:
            bias = np.cross(arm_dir, world_up)
            bias_norm = np.linalg.norm(bias)
        if bias_norm < 1e-6:
            bias = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            bias_norm = 1.0

        bend_dir = bias / bias_norm
        return base_point + bend_dir * math.sqrt(h_sq)

    def build_from_world_poses(
        self,
        head_pos: np.ndarray,
        head_quat_wxyz: np.ndarray,
        left_wrist_pos: np.ndarray,
        left_wrist_quat_wxyz: np.ndarray,
        right_wrist_pos: np.ndarray,
        right_wrist_quat_wxyz: np.ndarray,
    ) -> SMPLXFrame:
        torso_rot = self._yaw_only_rotation(head_quat_wxyz)
        torso_right = torso_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        torso_forward = torso_rot.apply(np.array([0.0, 1.0, 0.0], dtype=np.float64))
        torso_quat = torso_rot.as_quat(scalar_first=True)

        shoulder_center = (
            head_pos
            + torso_forward * self.shoulder_forward_offset
            + np.array([0.0, 0.0, -self.neck_to_shoulder_center], dtype=np.float64)
        )
        pelvis = (
            shoulder_center
            + torso_forward * self.pelvis_forward_offset
            + np.array([0.0, 0.0, -self.shoulder_center_to_pelvis], dtype=np.float64)
        )
        spine3 = pelvis + (shoulder_center - pelvis) * self.spine3_ratio

        left_shoulder = shoulder_center - torso_right * (self.shoulder_width * 0.5)
        right_shoulder = shoulder_center + torso_right * (self.shoulder_width * 0.5)

        left_elbow = self._solve_elbow(
            shoulder=left_shoulder,
            wrist=left_wrist_pos,
            torso_right=torso_right,
            torso_forward=torso_forward,
            is_left=True,
        )
        right_elbow = self._solve_elbow(
            shoulder=right_shoulder,
            wrist=right_wrist_pos,
            torso_right=torso_right,
            torso_forward=torso_forward,
            is_left=False,
        )

        return {
            "pelvis": (pelvis, torso_quat.copy()),
            "spine3": (spine3, torso_quat.copy()),
            "head": (head_pos, head_quat_wxyz.copy()),
            "left_shoulder": (left_shoulder, torso_quat.copy()),
            "right_shoulder": (right_shoulder, torso_quat.copy()),
            "left_elbow": (left_elbow, self._IDENTITY_WXYZ.copy()),
            "right_elbow": (right_elbow, self._IDENTITY_WXYZ.copy()),
            "left_wrist": (left_wrist_pos, left_wrist_quat_wxyz.copy()),
            "right_wrist": (right_wrist_pos, right_wrist_quat_wxyz.copy()),
        }

    def get_live_upper_body_frame(self) -> Optional[SMPLXFrame]:
        headset = self._transform_unity_pose(xrt.get_headset_pose())
        left_controller = self._transform_unity_pose(xrt.get_left_controller_pose())
        right_controller = self._transform_unity_pose(xrt.get_right_controller_pose())
        if headset is None or left_controller is None or right_controller is None:
            return None

        return self.build_from_world_poses(
            head_pos=headset[0],
            head_quat_wxyz=headset[1],
            left_wrist_pos=left_controller[0],
            left_wrist_quat_wxyz=left_controller[1],
            right_wrist_pos=right_controller[0],
            right_wrist_quat_wxyz=right_controller[1],
        )

    def get_mock_upper_body_frame(self, t_sec: float) -> SMPLXFrame:
        head_pos = np.array([0.0, 0.0, 1.60], dtype=np.float64)
        head_quat = R.from_euler(
            "xyz",
            [
                0.05 * math.sin(0.7 * t_sec),
                0.08 * math.sin(0.5 * t_sec),
                0.20 * math.sin(0.35 * t_sec),
            ],
        ).as_quat(scalar_first=True)

        left_wrist_pos = np.array(
            [
                -0.34 + 0.04 * math.sin(1.1 * t_sec),
                0.18 + 0.05 * math.sin(0.9 * t_sec),
                1.08 + 0.06 * math.sin(0.8 * t_sec),
            ],
            dtype=np.float64,
        )
        right_wrist_pos = np.array(
            [
                0.34 + 0.04 * math.sin(1.0 * t_sec + 0.5),
                0.18 + 0.05 * math.sin(0.7 * t_sec + 0.8),
                1.08 + 0.06 * math.sin(0.6 * t_sec + 0.3),
            ],
            dtype=np.float64,
        )
        left_wrist_quat = R.from_euler(
            "xyz",
            [0.1, -0.2 + 0.1 * math.sin(0.6 * t_sec), -0.4],
        ).as_quat(scalar_first=True)
        right_wrist_quat = R.from_euler(
            "xyz",
            [0.1, 0.2 + 0.1 * math.sin(0.6 * t_sec + 0.5), 0.4],
        ).as_quat(scalar_first=True)

        return self.build_from_world_poses(
            head_pos=head_pos,
            head_quat_wxyz=head_quat,
            left_wrist_pos=left_wrist_pos,
            left_wrist_quat_wxyz=left_wrist_quat,
            right_wrist_pos=right_wrist_pos,
            right_wrist_quat_wxyz=right_wrist_quat,
        )


def frame_to_jsonable(frame: SMPLXFrame) -> Dict[str, list]:
    return {
        joint_name: [pos.tolist(), quat.tolist()]
        for joint_name, (pos, quat) in frame.items()
    }


class UpperBodyGMRRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.redis_client = None
        self.redis_pipeline = None
        self.jsonl_file = None

        self.synthesizer = HeadsetControllerUpperBodySynthesizer(
            shoulder_width=args.shoulder_width,
            neck_to_shoulder_center=args.neck_to_shoulder_center,
            shoulder_center_to_pelvis=args.shoulder_center_to_pelvis,
            upper_arm_length=args.upper_arm_length,
            lower_arm_length=args.lower_arm_length,
            shoulder_forward_offset=args.shoulder_forward_offset,
            pelvis_forward_offset=args.pelvis_forward_offset,
            spine3_ratio=args.spine3_ratio,
        )
        self.gmr = GeneralMotionRetargeting(
            src_human="smplx",
            tgt_robot="jingchu03",
            actual_human_height=args.actual_human_height,
            verbose=args.verbose,
        )
        self.default_obs_prefix = DEFAULT_MIMIC_OBS["jingchu03_upper_body"][:6].astype(np.float32)

        if args.use_redis:
            self.redis_client = redis.Redis(host=args.redis_ip, port=6379, db=0)
            self.redis_client.ping()
            self.redis_pipeline = self.redis_client.pipeline()

        if args.save_pose_jsonl is not None:
            save_path = Path(args.save_pose_jsonl).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_file = save_path.open("w", encoding="utf-8")

        if not args.mock_input:
            xrt.init()

    def close(self) -> None:
        if self.jsonl_file is not None:
            self.jsonl_file.close()
        if not self.args.mock_input:
            try:
                xrt.close()
            except Exception:
                pass

    def _publish_to_redis(self, joint_qpos: np.ndarray) -> None:
        if self.redis_pipeline is None:
            return

        action_body = np.concatenate(
            [self.default_obs_prefix, joint_qpos.astype(np.float32)],
        ).astype(np.float32)
        self.redis_pipeline.set(
            "action_body_jingchu03_upper_body",
            json.dumps(action_body.tolist()),
        )
        self.redis_pipeline.set(
            "action_hand_left_jingchu03_upper_body",
            json.dumps(np.zeros(7, dtype=np.float32).tolist()),
        )
        self.redis_pipeline.set(
            "action_hand_right_jingchu03_upper_body",
            json.dumps(np.zeros(7, dtype=np.float32).tolist()),
        )
        self.redis_pipeline.set(
            "action_neck_jingchu03_upper_body",
            json.dumps(np.zeros(2, dtype=np.float32).tolist()),
        )
        self.redis_pipeline.set("t_action", int(time.time() * 1000))
        self.redis_pipeline.execute()

    def _save_pose_frame(self, frame_idx: int, frame: SMPLXFrame) -> None:
        if self.jsonl_file is None:
            return

        record = {
            "frame_idx": frame_idx,
            "timestamp_ns": time.time_ns(),
            "smplx_upper_body": frame_to_jsonable(frame),
        }
        self.jsonl_file.write(json.dumps(record) + "\n")
        self.jsonl_file.flush()

    def _get_smplx_frame(self, step_idx: int, t_start: float) -> Optional[SMPLXFrame]:
        if self.args.mock_input:
            t_sec = time.time() - t_start
            return self.synthesizer.get_mock_upper_body_frame(t_sec)
        return self.synthesizer.get_live_upper_body_frame()

    def run(self) -> None:
        xml_path = str(ROBOT_XML_DICT["jingchu03"])
        model = data = viewer = None
        if self.args.vis:
            model = mj.MjModel.from_xml_path(xml_path)
            data = mj.MjData(model)
            viewer = mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
            viewer.cam.distance = 1.8

        t_start = time.time()
        step_idx = 0
        loop_idx = 0
        invalid_frame_count = 0
        try:
            while True:
                if self.args.max_steps is not None and loop_idx >= self.args.max_steps:
                    break
                if viewer is not None and not viewer.is_running():
                    break

                loop_start = time.time()
                loop_idx += 1
                smplx_frame = self._get_smplx_frame(step_idx, t_start)
                if smplx_frame is None:
                    invalid_frame_count += 1
                    if invalid_frame_count % 30 == 1:
                        print("[upper_body_gmr] waiting for valid headset/controller poses ...")
                    time.sleep(1.0 / max(self.args.rate_hz, 1.0))
                    continue

                invalid_frame_count = 0
                qpos = self.gmr.retarget(smplx_frame)
                joint_qpos = np.asarray(qpos[-16:], dtype=np.float32)

                self._publish_to_redis(joint_qpos)
                self._save_pose_frame(step_idx, smplx_frame)

                if viewer is not None and model is not None and data is not None:
                    data.qpos[:] = qpos
                    mj.mj_forward(model, data)
                    viewer.cam.lookat = data.xpos[model.body("Robotbase").id]
                    viewer.sync()

                if step_idx % self.args.report_every == 0:
                    print(
                        f"[upper_body_gmr] step={step_idx} pelvis={smplx_frame['pelvis'][0]} "
                        f"left_wrist={smplx_frame['left_wrist'][0]} right_wrist={smplx_frame['right_wrist'][0]}"
                    )

                step_idx += 1
                elapsed = time.time() - loop_start
                target_dt = 1.0 / max(self.args.rate_hz, 1.0)
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
        finally:
            if viewer is not None:
                viewer.close()
            self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use headset + controllers to synthesize an upper-body SMPLX-style pose for GMR.",
    )
    parser.add_argument("--mock_input", action="store_true", help="Use synthetic headset/controller motions for testing.")
    parser.add_argument("--vis", action="store_true", help="Visualize GMR retargeting in MuJoCo.")
    parser.add_argument("--use_redis", action="store_true", help="Publish jingchu03 upper-body targets to Redis.")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis host when --use_redis is enabled.")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many frames.")
    parser.add_argument("--rate_hz", type=float, default=60.0, help="Loop frequency.")
    parser.add_argument("--report_every", type=int, default=30, help="Print one status line every N steps.")
    parser.add_argument("--actual_human_height", type=float, default=1.70, help="Human height used by GMR.")
    parser.add_argument("--save_pose_jsonl", type=str, default=None, help="Optional path to save synthesized SMPLX-style frames as JSONL.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose GMR logs.")

    parser.add_argument("--shoulder_width", type=float, default=0.38, help="Shoulder width used by the upper-body prior.")
    parser.add_argument("--neck_to_shoulder_center", type=float, default=0.20, help="Vertical offset from head to shoulder center.")
    parser.add_argument("--shoulder_center_to_pelvis", type=float, default=0.42, help="Vertical offset from shoulder center to pelvis.")
    parser.add_argument("--upper_arm_length", type=float, default=0.28, help="Upper arm length used by elbow synthesis.")
    parser.add_argument("--lower_arm_length", type=float, default=0.26, help="Lower arm length used by elbow synthesis.")
    parser.add_argument("--shoulder_forward_offset", type=float, default=0.02, help="Forward offset from head to shoulder center.")
    parser.add_argument("--pelvis_forward_offset", type=float, default=-0.02, help="Forward offset from shoulder center to pelvis.")
    parser.add_argument("--spine3_ratio", type=float, default=0.45, help="Interpolation ratio from pelvis to shoulder center for spine3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = UpperBodyGMRRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
