# MolmoAct2-LeRobot-bridge

A small bridge script for running
[`allenai/MolmoAct2-SO100_101`](https://huggingface.co/allenai/MolmoAct2-SO100_101)
on a calibrated LeRobot SO100 or SO101 follower arm.

The script connects a LeRobot follower, reads the top and wrist camera views,
queries MolmoAct2 for action chunks, converts the checkpoint action convention
to the current LeRobot SO calibration convention, and sends bounded joint
position targets to the robot.

## Compatibility

Recommended setup:

- Python `>=3.12`
- LeRobot `0.5.2` or a compatible LeRobot source checkout with SO100/SO101 support
- SO100 or SO101 follower arm with Feetech motors
- Optional SO100/SO101 leader arm
- Two cameras named `top` and `wrist`
- A CUDA GPU is strongly recommended for model inference

This bridge is intended to be run inside a working LeRobot environment. Before
using MolmoAct2, make sure normal LeRobot calibration and teleoperation work on
your robot.

## Install

Create or activate a LeRobot environment first. Follow the official LeRobot
installation instructions for your platform, then make sure the SO follower
hardware dependencies are installed.

For a source checkout, a typical editable install is:

```shell
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[feetech]"
```

Check whether `transformers` is already installed:

```shell
python -c "import transformers; print(transformers.__version__)"
```

If that command fails, install `transformers`. A hardware-only LeRobot install
does not always include it, but some policy extras already do.

```shell
pip install transformers
```

## Script Location

Recommended: put `run_molmoact2_lerobot.py` in the LeRobot repository root and
run it from there:

```text
lerobot/
├── src/
├── pyproject.toml
└── run_molmoact2_lerobot.py
```

```shell
cd /path/to/lerobot
python run_molmoact2_lerobot.py
```

The script does not have to live inside `src/lerobot/scripts`. It only needs to
run in an environment where Python imports the same LeRobot installation that
you used for calibration and teleoperation. Keeping it in the repository root is
the safest option because it avoids accidentally importing a different installed
copy of `lerobot`.

## Calibrate First

Calibrate the follower before running the policy:

```shell
lerobot-calibrate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so100_follower_arm
```

For SO101, use:

```shell
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so101_follower_arm
```

If you use a leader arm, calibrate it too:

```shell
lerobot-calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so100_leader_arm
```

## Check Ports and Cameras

Find motor USB ports:

```shell
lerobot-find-port
```

Find camera indexes:

```shell
lerobot-find-cameras opencv
```

The script defaults to:

- follower port: `/dev/ttyACM0`
- leader port: `/dev/ttyACM1`
- top camera: OpenCV index `6`
- wrist camera: OpenCV index `0`

Edit the camera indexes in `run_molmoact2_lerobot.py` if your cameras appear at
different indexes:

```python
"top": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=30)
"wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)
```

## Verify LeRobot Control

Before running MolmoAct2, verify teleoperation with LeRobot:

```shell
lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so100_follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so100_leader_arm
```

If this does not work, fix the LeRobot hardware setup before running the bridge.

## Run

From the LeRobot repo root:

```shell
python run_molmoact2_lerobot.py
```

Use a different task prompt:

```shell
python run_molmoact2_lerobot.py --prompt="Place the fork on the plate"
```

Use SO101:

```shell
python run_molmoact2_lerobot.py \
    --robot.type=so101_follower \
    --robot.id=so101_follower_arm \
    --teleop.type=so101_leader \
    --teleop.id=so101_leader_arm
```

Run without a leader arm:

```shell
python run_molmoact2_lerobot.py --teleop=null
```

When no leader is connected, the follower's initial pose is used as the home
pose. When a leader is connected, the follower first matches the leader pose and
uses that pose as home.

## Safety Behavior

The script:

- clips commands to the follower calibration limits
- limits per-step joint movement with `max_relative_target`
- returns the arm to the saved home pose on Ctrl+C, timeout, or normal exit
- disconnects through LeRobot, which may release motor torque depending on
  `robot.disable_torque_on_disconnect`

Keep a hand near power during first tests. Start with a clear workspace and a
short run.

## Important Defaults

Model:

```text
allenai/MolmoAct2-SO100_101
```

Normalization tag:

```text
so100_so101_molmoact2
```

Control settings:

```text
fps = 30
open_loop_horizon = 30
max_relative_target = 4.0
```

Camera image size:

```text
640 x 480
```

## Notes

The MolmoAct2 checkpoint and current LeRobot SO calibration convention do not
use identical shoulder and elbow values. The script handles this at the model
boundary:

- current LeRobot state is converted before being sent to MolmoAct2
- MolmoAct2 actions are converted back before being sent to the robot

Do not remove this conversion unless the checkpoint and LeRobot calibration
conventions are known to match.
