# Copyright 2026 YIng-Chun Lee. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MolmoAct2 on a calibrated LeRobot SO100/SO101 follower arm.

This is a policy-inference script for the Hugging Face
``allenai/MolmoAct2-SO100_101`` checkpoint. It reads two camera views and the
current follower joint state, asks MolmoAct2 for an action chunk, converts the
checkpoint action convention to current LeRobot SO commands, and sends safe
joint-position targets to the follower.

Requires: a LeRobot hardware environment with MolmoAct2 dependencies installed.

Default behavior:
- The follower arm is required.
- SO100 and SO101 use the same script. Use ``--robot.type=so100_follower``
  or ``--robot.type=so101_follower`` to select the hardware.
- The leader arm is optional. If connected, the follower first matches the
  leader and that pose becomes the home pose. If the leader is disconnected or
  unavailable, the follower's initial pose becomes the home pose.
- On Ctrl+C, timeout, or normal exit after ``max_steps``, the follower smoothly
  returns to the saved home pose before disconnecting.
- Commands are clipped to the follower's calibration limits and to
  ``max_relative_target`` per control tick.

Example with the default SO100 follower and optional SO100 leader:

```shell
run_molmoact2_lerobot.py
```

Example with a different prompt:

```shell
run_molmoact2_lerobot.py --prompt="Place the fork on the plate"
```
"""

import logging
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.motors import MotorNormMode
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from lerobot.robots.so_follower import SO100FollowerConfig
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so_leader,
)
from lerobot.teleoperators.so_leader import SO100LeaderConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


MOTOR_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30

_emergency_stop = False


@dataclass
class MolmoAct2LeRobotConfig:
    """Configuration for MolmoAct2 rollout on a LeRobot SO follower.

    The default values are chosen for a common SO100 setup:
    follower on ``/dev/ttyACM0``, optional leader on ``/dev/ttyACM1``, top
    camera index ``6``, and wrist camera index ``0``. All of these can be
    overridden with LeRobot/draccus CLI fields such as ``--robot.port``,
    ``--robot.cameras``, ``--teleop.port``, and ``--prompt``.
    """

    # Follower robot. This must be connected for the script to run.
    robot: RobotConfig = field(
        default_factory=lambda: SO100FollowerConfig(
            port="/dev/ttyACM0",
            id="so100_follower_arm",
            cameras={
                # Change these OpenCV indexes to match your top and wrist cameras.
                "top": OpenCVCameraConfig(
                    index_or_path=6,
                    width=DEFAULT_CAMERA_WIDTH,
                    height=DEFAULT_CAMERA_HEIGHT,
                    fps=DEFAULT_CAMERA_FPS,
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=0,
                    width=DEFAULT_CAMERA_WIDTH,
                    height=DEFAULT_CAMERA_HEIGHT,
                    fps=DEFAULT_CAMERA_FPS,
                ),
            },
        )
    )
    # Optional leader. If unavailable, the script keeps running from follower home.
    teleop: TeleoperatorConfig | None = field(
        default_factory=lambda: SO100LeaderConfig(port="/dev/ttyACM1", id="so100_leader_arm")
    )
    # MolmoAct2 checkpoint source. Can be a Hugging Face repo id or local path.
    model_id: str = "allenai/MolmoAct2-SO100_101"
    # Normalization tag stored in the checkpoint's norm_stats.json.
    norm_tag: str = "so100_so101_molmoact2"
    # Task instruction sent to MolmoAct2 on every action-chunk request.
    prompt: str = "Place the block on the plate"
    # Use "cuda" when available; CPU works for testing but is slower.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Control-loop frequency for sending robot commands.
    fps: int = 30
    # Maximum number of control steps before normal shutdown.
    max_steps: int = 10_000
    # Per-tick joint delta cap applied after model prediction.
    max_relative_target: float = 4.0
    # Number of actions to execute from each predicted MolmoAct2 chunk.
    open_loop_horizon: int = 30
    # Image size passed to MolmoAct2.
    image_width: int = DEFAULT_CAMERA_WIDTH
    image_height: int = DEFAULT_CAMERA_HEIGHT
    # Return-home timing on shutdown.
    shutdown_return_time_s: float = 5.0
    home_tolerance: float = 1.0


def stop_handler(signum: int, frame: Any) -> None:
    del signum, frame
    global _emergency_stop
    _emergency_stop = True
    raise KeyboardInterrupt


def observation_state(obs: dict[str, Any]) -> np.ndarray:
    """Extract the six SO arm joint positions from a LeRobot observation."""

    missing = [key for key in MOTOR_KEYS if key not in obs]
    if missing:
        raise KeyError(f"Missing motor keys: {missing}")

    state = []
    for key in MOTOR_KEYS:
        value = obs[key]
        if isinstance(value, (list, np.ndarray)):
            value = np.asarray(value).squeeze()
            if value.size != 1:
                raise ValueError(f"Key {key} is not scalar: shape {np.shape(value)}")
            value = float(value)
        state.append(value)
    return np.asarray(state, dtype=np.float32)


def action_state(action: dict[str, Any]) -> np.ndarray:
    """Extract the six SO arm joint targets from a LeRobot action."""

    return observation_state(action)


def vector_to_action(values: np.ndarray) -> dict[str, float]:
    """Convert a six-element joint vector into LeRobot's keyed action dict."""

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) != len(MOTOR_KEYS):
        raise ValueError(f"Expected {len(MOTOR_KEYS)} action values, got {len(values)}")
    return {key: float(value) for key, value in zip(MOTOR_KEYS, values, strict=True)}


class MolmoAct2CalibrationAdapter:
    """Convert between LeRobot SO joint values and MolmoAct2 SO joint values.

    MolmoAct2-SO100_101 expects shoulder lift and elbow flex in the SO100/SO101
    convention used by the checkpoint. The follower receives commands in the
    current LeRobot calibrated convention, so state and action vectors pass
    through this adapter at the model boundary.
    """

    @staticmethod
    def robot_state_to_model_state(state: np.ndarray) -> np.ndarray:
        mapped = np.asarray(state, dtype=np.float32).copy()
        shoulder_lift = MOTOR_KEYS.index("shoulder_lift.pos")
        elbow_flex = MOTOR_KEYS.index("elbow_flex.pos")
        mapped[shoulder_lift] = 90.0 - mapped[shoulder_lift]
        mapped[elbow_flex] = mapped[elbow_flex] + 90.0
        return mapped

    @staticmethod
    def model_action_to_robot_action(action: np.ndarray) -> np.ndarray:
        mapped = np.asarray(action, dtype=np.float32).copy()
        shoulder_lift = MOTOR_KEYS.index("shoulder_lift.pos")
        elbow_flex = MOTOR_KEYS.index("elbow_flex.pos")
        mapped[shoulder_lift] = 90.0 - mapped[shoulder_lift]
        mapped[elbow_flex] = mapped[elbow_flex] - 90.0
        return mapped


def read_robot_state(robot: Robot) -> np.ndarray:
    """Read only joint positions, avoiding camera reads when the motor bus is available."""

    bus = getattr(robot, "bus", None)
    if bus is not None and getattr(bus, "is_connected", False):
        positions = bus.sync_read("Present_Position")
        return observation_state({f"{motor}.pos": value for motor, value in positions.items()})
    return observation_state(robot.get_observation())


def calibration_limits(robot: Robot) -> tuple[np.ndarray, np.ndarray]:
    """Return low/high calibrated joint limits in LeRobot command units."""

    low = np.full(len(MOTOR_KEYS), -np.inf, dtype=np.float32)
    high = np.full(len(MOTOR_KEYS), np.inf, dtype=np.float32)

    bus = getattr(robot, "bus", None)
    calibration = getattr(bus, "calibration", None)
    motors = getattr(bus, "motors", None)
    model_resolution_table = getattr(bus, "model_resolution_table", None)
    if not calibration or not motors or not model_resolution_table:
        return low, high

    for i, key in enumerate(MOTOR_KEYS):
        motor_name = key.removesuffix(".pos")
        motor = motors[motor_name]
        if motor.norm_mode is MotorNormMode.RANGE_0_100:
            low[i], high[i] = 0.0, 100.0
            continue
        if motor.norm_mode is MotorNormMode.RANGE_M100_100:
            low[i], high[i] = -100.0, 100.0
            continue
        if motor.norm_mode is MotorNormMode.DEGREES:
            cal = calibration[motor_name]
            mid = (cal.range_min + cal.range_max) / 2
            max_res = model_resolution_table[motor.model] - 1
            low[i] = (cal.range_min - mid) * 360 / max_res
            high[i] = (cal.range_max - mid) * 360 / max_res

    return low, high


def sanitize_action(raw_action: np.ndarray, limits: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Validate action shape/values and clip it to calibrated joint limits."""

    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if len(action) != len(MOTOR_KEYS):
        raise ValueError(f"Expected {len(MOTOR_KEYS)} action values, got {len(action)}")
    if not np.all(np.isfinite(action)):
        raise ValueError(f"MolmoAct2 returned non-finite action values: {action}")

    low, high = limits
    return np.clip(action, low, high).astype(np.float32)


def return_home(
    robot: Robot,
    home_position: np.ndarray,
    action_limits: tuple[np.ndarray, np.ndarray],
    duration_s: float,
    tolerance: float,
) -> None:
    """Move the follower back to the saved home pose with linear interpolation."""

    if not robot.is_connected:
        return

    home_position = sanitize_action(home_position, action_limits)
    current_position = sanitize_action(read_robot_state(robot), action_limits)
    initial_error = float(np.max(np.abs(home_position - current_position)))
    if initial_error <= tolerance:
        robot.send_action(vector_to_action(home_position))
        logging.info("Return-home skipped; already within %.2f degrees of home.", initial_error)
        return

    control_fps = 50
    steps = max(int(duration_s * control_fps), 1)

    for step_idx in range(1, steps + 1):
        alpha = step_idx / steps
        target = current_position * (1.0 - alpha) + home_position * alpha
        robot.send_action(vector_to_action(sanitize_action(target, action_limits)))
        precise_sleep(1.0 / control_fps)

    robot.send_action(vector_to_action(home_position))
    precise_sleep(0.5)
    final_error = float(np.max(np.abs(home_position - sanitize_action(read_robot_state(robot), action_limits))))
    logging.info("Return-home finished with max joint error %.2f.", final_error)


def image_to_pil(image: Any) -> Image.Image:
    """Normalize LeRobot camera output to a PIL image for MolmoAct2."""

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC or CHW image, got shape {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def first_action(output: Any) -> np.ndarray:
    """Normalize MolmoAct2 output into an action array."""

    actions = getattr(output, "actions", output)
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()

    array = np.asarray(actions, dtype=np.float32).squeeze()
    if array.ndim == 0:
        raise ValueError("MolmoAct2 returned a scalar action; expected a 6D joint target.")
    if array.ndim == 1:
        action = array
    else:
        action = array.reshape(-1, array.shape[-1])
    if action.ndim == 1 and len(action) != len(MOTOR_KEYS):
        raise ValueError(f"MolmoAct2 returned action shape {array.shape}; expected last dimension 6.")
    return action.astype(np.float32)


def predict_action(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    task: str,
    state: np.ndarray,
    norm_tag: str,
) -> np.ndarray:
    """Run one MolmoAct2 action-chunk prediction."""

    use_cuda_graph = any(parameter.is_cuda for parameter in model.parameters())
    autocast_enabled = str(next(model.parameters()).device).startswith("cuda")
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast_enabled else nullcontext()
    )

    with torch.inference_mode(), autocast_context:
        output = model.predict_action(
            processor=processor,
            images=images,
            task=task,
            state=np.asarray(state, dtype=np.float32),
            norm_tag=norm_tag,
            action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=10,
            normalize_language=True,
            enable_cuda_graph=use_cuda_graph,
        )
    return first_action(output)


def load_molmoact2(cfg: MolmoAct2LeRobotConfig) -> tuple[Any, Any]:
    """Load MolmoAct2 processor/model from Hugging Face or local cache."""

    dtype = torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32

    def resolve_snapshot(*, local_files_only: bool) -> str:
        source_path = Path(cfg.model_id).expanduser()
        if source_path.exists():
            return str(source_path)
        return snapshot_download(
            repo_id=cfg.model_id,
            local_files_only=local_files_only,
            allow_patterns=["*.json", "*.py", "*.jinja", "*.safetensors", "tokenizer.*"],
        )

    def load_from(source: str) -> tuple[Any, Any]:
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            source,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(cfg.device)
        model.eval()
        return processor, model

    online_error: Exception | None = None
    try:
        snapshot_path = resolve_snapshot(local_files_only=False)
        logging.info("Loading MolmoAct2 from snapshot: %s", snapshot_path)
        return load_from(snapshot_path)
    except Exception as exc:
        online_error = exc
        logging.warning("Hugging Face load failed, trying local cache: %s", exc)

    try:
        snapshot_path = resolve_snapshot(local_files_only=True)
        logging.info("Loading MolmoAct2 from local cache: %s", snapshot_path)
        return load_from(snapshot_path)
    except Exception as cache_exc:
        raise RuntimeError(
            "Could not load MolmoAct2 from Hugging Face or local cache. "
            "Connect to the internet once to download it, or make sure the cache exists."
        ) from online_error or cache_exc


def molmoact2_loop(
    robot: Robot,
    processor: Any,
    model: Any,
    cfg: MolmoAct2LeRobotConfig,
    adapter: MolmoAct2CalibrationAdapter,
    action_limits: tuple[np.ndarray, np.ndarray],
) -> bool:
    """Run the main MolmoAct2 closed-loop rollout.

    Returns:
        True if at least one robot action was sent. The shutdown code uses this
        to decide whether it should return the arm to home.
    """

    action_queue: list[np.ndarray] = []
    robot_actions_sent = False

    for step in range(cfg.max_steps):
        if _emergency_stop:
            break

        loop_start = time.perf_counter()
        obs = robot.get_observation()
        state = observation_state(obs)
        top_pil = image_to_pil(obs["top"]).resize((cfg.image_width, cfg.image_height))
        wrist_pil = image_to_pil(obs["wrist"]).resize((cfg.image_width, cfg.image_height))

        if len(action_queue) == 0:
            model_state = adapter.robot_state_to_model_state(state)
            model_actions = predict_action(
                model,
                processor,
                [top_pil, wrist_pil],
                cfg.prompt,
                model_state,
                cfg.norm_tag,
            )
            if model_actions.ndim == 1:
                model_actions = model_actions.reshape(1, -1)
            robot_actions = np.asarray(
                [adapter.model_action_to_robot_action(action) for action in model_actions],
                dtype=np.float32,
            )
            horizon = min(cfg.open_loop_horizon, len(robot_actions))
            action_queue = [robot_actions[i] for i in range(horizon)]

        raw_action = sanitize_action(action_queue.pop(0), action_limits)
        safe_action = state + np.clip(raw_action - state, -cfg.max_relative_target, cfg.max_relative_target)
        safe_action = sanitize_action(safe_action, action_limits)
        _ = robot.send_action(vector_to_action(safe_action))
        robot_actions_sent = True

        logging.info("step=%d state=%s target=%s", step, np.round(state, 2), np.round(safe_action, 2))
        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / cfg.fps - dt_s, 0.0))

    return robot_actions_sent


@parser.wrap()
def run_molmoact2(cfg: MolmoAct2LeRobotConfig) -> None:
    """Connect hardware, run MolmoAct2, and cleanly return home on shutdown."""

    init_logging()
    logging.info(pformat(asdict(cfg)))

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    adapter = MolmoAct2CalibrationAdapter()

    home_position: np.ndarray | None = None
    action_limits = (
        np.full(len(MOTOR_KEYS), -np.inf, dtype=np.float32),
        np.full(len(MOTOR_KEYS), np.inf, dtype=np.float32),
    )

    robot.connect()
    try:
        action_limits = calibration_limits(robot)
        home_position = observation_state(robot.get_observation())

        if teleop is not None:
            try:
                teleop.connect()
                home_position = sanitize_action(action_state(teleop.get_action()), action_limits)
                _ = robot.send_action(vector_to_action(home_position))
                time.sleep(1.0)
                logging.info("Follower matched teleoperator home position.")
            except ConnectionError as exc:
                logging.warning(
                    "Teleoperator unavailable; continuing without leader arm. "
                    "The follower initial pose will be used as home: %s",
                    exc,
                )
                teleop = None

        processor, model = load_molmoact2(cfg)

        dummy = Image.fromarray(np.zeros((cfg.image_height, cfg.image_width, 3), dtype=np.uint8))
        dummy_state = adapter.robot_state_to_model_state(home_position)
        for _ in range(3):
            predict_action(model, processor, [dummy, dummy], "test", dummy_state, cfg.norm_tag)

        molmoact2_loop(robot, processor, model, cfg, adapter, action_limits)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if home_position is not None:
            try:
                logging.info("Returning follower to home before shutdown.")
                return_home(
                    robot=robot,
                    home_position=home_position,
                    action_limits=action_limits,
                    duration_s=cfg.shutdown_return_time_s,
                    tolerance=cfg.home_tolerance,
                )
            except Exception as exc:
                logging.warning("Could not return home before shutdown: %s", exc)
        logging.info("Disconnecting follower. Torque release is controlled by robot.disable_torque_on_disconnect.")
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    run_molmoact2()


if __name__ == "__main__":
    main()
