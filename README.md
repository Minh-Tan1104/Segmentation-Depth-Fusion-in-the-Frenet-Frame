# RL_CAR — Segmentation + Depth Fusion in the Frenet Frame

A full self-driving stack for a differential-drive car on ROS2 Humble /
Jetson Orin. Lane geometry and obstacles from a segmentation and a detection
YOLO model are fused with depth into a single **Frenet-frame** state
(`s, d, heading`), which a Frenet Optimal Trajectory planner and pure-pursuit
controller then track in real time.

> **On the name:** despite "RL_CAR", the pipeline is **rule-based + classical
> planning** — YOLO perception, an Extended Kalman Filter, and a Frenet
> Optimal Trajectory planner. There is no reinforcement-learning component
> (no gym environment, policy network, or training loop). The name is a
> holdover from an earlier direction of the project.

## Overview

The car carries a depth camera and a GPS unit and drives with only one
control authority: a single node that owns `/cmd_vel`. Everything upstream
of it — perception, GPS, odometry — only supplies data.

```
Sensor Data                Perception module            Lane-frame     Local planner   Control module
 ├─ GPS ────────┐           ├─ Depth Estimation  ┐            │              │               │
 ├─ RealSense ───┼─Camera──►├─ Object Detection  ┼──────────► ├─ vision +    │               │
 │               │Encoder   └─ Line Segmentation ┘            │  encoder     ├─ optimal   ──► ├─ path
 └─ Encoder ─────┘                                            │  fusion      │  path            tracking ──► Actuator
                  └─GPS/Encoder─► Route representation ──────►│  Route-frame │  generation    │  algorithm
                                   (waypoints, curve)          └─ encoder +   │                │
                                                                   gated GPS ─┘                │
```

![Perception → planning → control pipeline](docs/images/pipeline-overview.png)

Two independent state estimates feed the planner, each with its own EKF:

- **Lane-frame** — camera segmentation fused with wheel-encoder odometry
  (`FrenetEKF`, state `[s, d, psi, v]`). This is the primary estimate during
  normal lane following.
- **Route-frame** — wheel-encoder odometry anchored by GPS map-matching
  against a recorded CSV route (`RouteEKF`, state `[x, y, psi]`). This
  estimate only gates in when the car enters a pre-mapped curve zone, where
  vision alone is less reliable; GPS is never allowed to correct the
  lane-frame estimate.

Both feed a **Frenet Optimal Trajectory** planner (quintic lateral × quartic
longitudinal candidate generation, cost-ranked by jerk, time, center offset,
and obstacle clearance), tracked by a pure-pursuit controller.

## Demo

Left: RGB frame with YOLO detection (car bounding box) and lane segmentation
(green centerline) overlaid. Right: the corresponding Frenet/EKF view —
candidate trajectories in gray, the selected path in cyan, the pure-pursuit
lookahead point in magenta.

![Camera view and Frenet/EKF planner visualization](docs/images/frenet-ekf-demo.jpg)

## Hardware

![Hardware architecture](docs/images/hardware-architecture.png)

| | Component | Role |
|---|---|---|
| a | Jetson Orin | GPU compute — perception inference, planning, control |
| b | u-blox M10 GPS module | UBX NAV-PVT fixes for route-frame localization |
| c | Intel RealSense D4xx | RGB + depth for lane segmentation and obstacle detection |
| d | Hoverboard motor controller mainboard | Differential-drive actuation, wheel encoder feedback |
| e | Hub motor wheel | Drive output |

## Key features

- **Segmentation + depth → Frenet frame.** Lane segmentation and depth are
  fused into a lane-relative `(s, d, heading)` state instead of a raw pixel
  or occupancy-grid representation.
- **Detection → Frenet-frame obstacles.** Detected obstacles are projected
  into the same `(s, x)` frame with EMA smoothing and TTL-based tracking.
- **Dual EKF localization.** A camera-driven lane-frame EKF and a
  GPS-driven route-frame EKF run independently; only one is ever allowed to
  steer at a time, and GPS never overwrites the vision estimate outside
  mapped curve zones.
- **Frenet Optimal Trajectory planning** with a curved-reference mode
  (`scipy.CubicSpline` over a recorded GPS route) for sharp mapped curves,
  and a straight-reference mode for ordinary lane following.
- **Single point of control authority.** Only one node ever publishes
  `/cmd_vel`; every other node is read-only with respect to actuation.

## System architecture

7 ROS2 executables across 6 Python subpackages of one `ament_python`
package. Each node splits pure logic (`logic.py` — no `rclpy` import,
testable offline) from ROS wiring (`node.py` — subscribe / publish /
parameters only).

```
joy_pygame_node ──/joy──────┐
encoder_node ────/odom──────┤
perception_node ─/frenet_state──┤──► control_node ──/cmd_vel──► encoder_node → motor controller
gps_node ────────/route_state───┘         │
                                            ├──► visualization_node (read-only debug grid)
planner_motion_node (parallel planner instance, viz/rosbag only — not in the drive path)
```

| Node | Role |
|---|---|
| `perception_node` | Two YOLO models (GPU) on RealSense RGB+Depth: segmentation → lane Frenet state, detection → obstacles projected into Frenet space |
| `control_node` | The only `/cmd_vel` publisher. Two timers on a `MultiThreadedExecutor`: a 50 Hz EKF-predict/manual-drive tick, and a 15 Hz planner tick (Frenet Optimal Planner + pure pursuit + GPS curvature feed-forward) |
| `gps_node` | Reads UBX NAV-PVT, map-matches against a recorded CSV route, publishes route-frame state and detected curve zones |
| `encoder_node` | Owns the motor-controller serial link, decodes wheel RPM, integrates dead-reckoning odometry → `/odom` + TF |
| `planner_motion_node` | A second, independent Frenet planner instance driven only by live camera data — visualization/rosbag use only, never drives the car |
| `visualization_node` | Read-only 2×2 debug grid (camera+segmentation, Frenet/EKF planner, encoder trail, GPS route) |

For the full technical write-up — EKF gating logic, curve-zone handoff,
tuning notes, and known gotchas discovered during field testing — see
[`docs/ARCHITECTURE.vi.md`](docs/ARCHITECTURE.vi.md) (Vietnamese).

## Repository structure

```
RL_CAR/
├── perception/        YOLO segmentation + detection, depth fusion, Frenet projection
├── control/            EKF, planner invocation, pure pursuit, /cmd_vel — the only actuation path
├── planner_motion/     Frenet Optimal Trajectory planner (shared) + a viz-only node
├── Gps/                UBX GPS reader, map matching, route EKF
├── Encoder/            Motor-controller serial link, odometry integration
├── visualization/      Read-only debug overlay
├── config/              rl_car_params.yaml — single source of truth for all node parameters
├── launch/              full_stack.launch.py, rgb_dual_model.launch.py, realsense.launch.py
├── map/                 Recorded GPS route CSVs + curve-zone map
├── scripts/             Build/run scripts, rosbag tooling, offline plotting utilities
├── docs/                 Detailed architecture doc + images used in this README
└── frenet_optimal_trajectory.py   Standalone offline reference planner (not part of the live ROS pipeline)
```

## Getting started

```bash
# Build (from the workspace root, the parent of src/)
colcon build --packages-select RL_CAR
source install/setup.bash

# Full stack: camera + perception + control + encoder + joystick + GPS
ros2 launch RL_CAR full_stack.launch.py launch_gps:=true

# Camera pipeline only (e.g. against a rosbag, no hardware GPS/encoder)
ros2 launch RL_CAR rgb_dual_model.launch.py launch_realsense:=false
```

All node parameters — speed limits, EKF noise, planner cost weights, model
paths, topics, GPS gating — live in one file:
[`config/rl_car_params.yaml`](config/rl_car_params.yaml). Rebuild after any
edit (`install/` is a static copy, not a symlink).

### Model weights

Detection and segmentation checkpoints are not committed to this
repository. Place your own weights at:

```
model/detection/yolo11n.pt
model/seg/best.pt
```

or point the `RL_CAR_SOURCE_ROOT` environment variable at a source tree that
already has them (see `perception/model_paths.py`).

## Known limitations

- Encoder odometry is unfiltered dead reckoning — it accumulates drift,
  especially wheel slip on sharp turns. This is the reason both EKFs exist.
- GPS is only trusted to correct lateral position once, on entry to a
  pre-mapped curve zone — treating it as a continuous correction source was
  found to let a single bad reading teleport the pose by several meters.
- `frenet_optimal_trajectory.py` at the package root is an offline reference
  implementation (curved cubic-spline planner), kept for experimentation —
  it is not wired into the live ROS pipeline.

## License

MIT — see [LICENSE](LICENSE).
