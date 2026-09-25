# RL_CAR — Segmentation + Depth Fusion in the Frenet Frame

A full self-driving stack for a differential-drive car on ROS2 Humble /
Jetson Orin. Lane geometry and obstacles from a segmentation and a detection
YOLO model are fused with depth into a single **Frenet-frame** state
(`s, d, heading`), which a Frenet Optimal Trajectory planner and pure-pursuit
controller then track in real time.

> **On the name:** by default the pipeline is **rule-based + classical
> planning** — YOLO perception, an Extended Kalman Filter, and a Frenet
> Optimal Trajectory planner. Reinforcement learning is an **optional,
> config-switched** component: a SAC policy can replace the straight-mode
> planner's cost-based choice of lateral offset (see
> [RL lateral planner](#optional-rl-lateral-planner-sac)).

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
- **Optional RL lateral planner.** A SAC policy, trained on the planner's
  own cost function, can pick the straight-mode avoidance offset instead of
  enumerating candidates — switched by one config flag, with a per-tick
  fallback to the classical planner.

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
├── models/               Trained SAC policy for the optional RL lateral planner (+ .meta.json)
├── train_frenet_rl.py    Gymnasium env + SAC training for the RL lateral planner (offline, not a ROS node)
├── test_frenet_rl.py     Evaluates the SAC policy against the classical planner on fixed scenarios
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

## Optional: RL lateral planner (SAC)

In straight mode the classical planner enumerates ~117 candidate
trajectories (lateral offset × horizon × speed) and keeps the lowest-cost
one. The RL option replaces that **choice** with a Soft Actor-Critic policy
that outputs the target lateral offset `d_target` and lateral horizon `Ti`
directly. Everything else is unchanged: the same quintic/quartic polynomials
build the single chosen path, the same pure pursuit tracks it, and curve mode
always stays classical.

**Switching it on** — in [`config/rl_car_params.yaml`](config/rl_car_params.yaml),
under `control_node`:

```yaml
plan_use_rl: true        # false = classical Frenet cost-based (default)
plan_rl_model_path: "~/ros2_ws/src/RL_CAR/models/sac_frenet_straight_parity_s1_150k.zip"
```

It needs `stable-baselines3` and `torch` on the car; they are only imported
when the flag is on. Every RL path still goes through the planner's hard
checks (collision, speed, acceleration, curvature). If a path fails, that
tick falls back to the classical planner and `control_node` logs a warning,
so an RL mistake never reaches the motors.

Obstacles are only shown to the policy when they matter. Each tick first
asks the policy for its plain lane-keeping path. The gate extends that path
to the end of the vision range at its final offset. If the path is feasible
and every obstacle stays at least 0.9 m (robot radius 0.6 + 0.3 m margin,
the same gap the feasibility rules require) from it, the path is used as is.
Without this gate the policy swerved around obstacles it could have passed
going straight. The extension matters with the robot's settings: a path is
only 2–3 m long, while obstacles are visible up to 8 m. Checking the short
path alone let the car drive straight at obstacles 3–6 m ahead until it was
too late to avoid them (9 of 20 multi-obstacle failures).

**How it was trained** — `train_frenet_rl.py` wraps the straight-mode
planner in a Gymnasium environment:

- **Observation:** `[d, psi]` plus, for each of the 3 nearest obstacles
  ahead, `[distance, lateral offset, present flag]`, seen in a mirrored frame
  (below). The mirror side is latched on the nearest obstacle.
- **Reward:** the negative of the classical planner's cost (jerk, time,
  centre offset, obstacle clearance) plus a constant per-step bonus, with
  terminal penalties for collision and leaving the lane. A constant shift
  does not change which action is cheaper, so the policy optimises the same
  objective the classical planner minimises by search.
- **Episodes:** a stream of static obstacle clusters along the road
  (clusters 12–25 m apart, each present with probability 0.7). Each cluster
  holds 1–3 obstacles within 6 m, so up to 3 can be in view at once; 20 %
  of obstacles are near the centreline. Every layout is checked for
  feasibility and regenerated if it fails (see below).
- **Robot settings:** planner (`plan_*`) and pure pursuit (`lookahead_distance`,
  `max_wheel_speed`) parameters are read from
  [`config/rl_car_params.yaml`](config/rl_car_params.yaml), so the policy
  optimises the same cost as the classical planner on the car.
- **Timing:** each RL decision is held for 0.5 s. Inside it, pure pursuit
  runs at 40 Hz and the path is rebuilt from the current state on every tick.
  On the car the policy is queried on every planner tick.
- **Action shaping:** the offset output goes through a signed square
  (`action_power: 2`) before scaling to `d_target`, so small outputs map
  to a near-zero offset and the empty-road lane keeping is exact.
- **Terminal penalties** (collision, leaving the lane) are 3000, and all
  rewards are scaled by 0.05. With the robot's obstacle cost, passing one
  obstacle costs ~460 (clearance 1.2 m), so smaller penalties would make
  crashing cheaper than passing.

**Feasibility constraints** (`feasible_layout` in `train_frenet_rl.py`).
These are geometric and independent of any planner:

| # | Constraint |
|---|---|
| R1 | 1–3 obstacles per cluster, spread over at most 6 m; clusters 12–25 m apart. |
| R2 | Every obstacle is on the road: \|d\| ≤ 1.8 m. |
| R3/R4 | Obstacles closer than 1.5 m along the road form a row. Each row must leave a gap inside the lane (\|d\| ≤ 2.2 m) at least 0.9 m (robot radius 0.6 + 0.3 m margin) from every obstacle. |
| R5 | Between consecutive rows the car can shift sideways by at most 0.4 × the distance between them. |

A dynamic programme propagates the reachable lateral intervals from row to
row. If the set becomes empty, the layout is infeasible. Frenet is not used
as the filter because, with the robot's settings, it cannot pass even a
single centred obstacle.

Three design choices fix failures seen in earlier models:

| Problem | Evidence | Fix |
|---|---|---|
| A continuous policy has to pass through `d_target ≈ 0` when it switches from "avoid right" to "avoid left", so there is always a band of obstacle positions near the centreline where it drives straight into the obstacle. More training only narrows or moves the band. | Previous model: 18/87 collisions with obstacles within ±0.05 m of the car's line. | **Mirrored frame + side latch.** The policy always sees the obstacle on its left; if it is really on the right, the state is mirrored going in and the action coming out. Picking a side becomes a sign test, and the side is latched per obstacle so sensor noise cannot flip it mid-manoeuvre. |
| "No obstacle" was encoded exactly like "obstacle on the centreline at the edge of vision". | Erratic lane keeping, no stable point at the lane centre. | Explicit obstacle-in-view flag. |
| The critic valued being off-centre, so the policy drifted ~0.7 m. A +200 goal bonus at the end of the road made value depend on distance travelled, which the policy cannot observe. Offset also grew with distance (every episode starts near the centre). The policy mistook "offset" for "near the goal". | Remaining return rose from −7.6 at the start to +164 near the goal. Forcing the centre on an empty road scored 90 vs 37 for the policy. | Per-step bonus instead of a goal bonus. The end of the road is a time-limit truncation, not a terminal state. |
| The gate used the 1.2 m planning clearance. Layouts with a gap between 0.9 and 1.2 m wide always reached the policy, which then learned to swerve wide everywhere. | Previous model: some layouts the classical planner passes were failed by SAC. | Gate at 0.9 m, the same gap rule R3/R4 uses. |
| A linear offset output left a small constant bias on an empty road. | Previous model: lane offset 0.06 m. | Signed-square action mapping (`action_power: 2`). |

The observation/action encoding (including the mirrored frame) lives in
[`planner_motion/rl_policy.py`](planner_motion/rl_policy.py) and is shared
by training and the live node. The constants used at training time are saved
next to the model (`.meta.json`), so decoding on the car matches training
even if the robot's `plan_*` parameters differ. Older models without the new
meta fields still load and run as before.

Model in `models/`: `sac_frenet_straight_parity_s1_150k.zip` (+ `.meta.json`),
the one the config points to. Robot settings, clearance 1.2 m, 1–3 obstacle
clusters, 3-obstacle observation, gate at 0.9 m, `action_power: 2`, seed 1,
150k steps. Earlier models were removed from the repo; they are still in the
git history.

```bash
pip install -r requirements.txt
python3 train_frenet_rl.py            # trains, writes models/sac_frenet_straight.zip + .meta.json
python3 test_frenet_rl.py [model] [--set single|multi|hard|all|report] [--video DIR] [--horizon T]   # animated, SAC and classical side by side
python3 compare_rl_frenet.py [model] [--multi]                      # tables: left / right / centre / none, or multi-obstacle
```

**Results in simulation (robot settings, clearance 1.2 m).** Model
`sac_frenet_straight_parity_s1_150k`. Both planners replan every tick, and
pure pursuit runs at 40 Hz. "SAC" means the logic used on the car: the
gate, the policy, the hard checks, and a fallback to the classical planner
on any tick where the RL path is rejected.

200 generated layouts with 1–3-obstacle clusters (feasible by the constraints above):

| | Classical planner | SAC |
|---|---|---|
| Failures | **122/200**, all "no feasible path" | **2/200** |
| Min distance to obstacles (5th percentile) | — | 0.93 m |
| Mean \|d\| | — | 0.15 m |
| Planning time per tick | ~15 ms | 0.8 ms |
| Ticks that fell back to classical | — | 0.06 % |

The classical planner also fails both SAC failures. Over every layout the
classical planner passes (78 generated + 15 hand-built), SAC fails **none**.
Without the fallback, the policy alone fails 10/200.

Single obstacle:
- Policy alone: 0/87 collisions for obstacles near the car's line, and
  0/400 over random episodes with ±0.05 m obstacle measurement noise.
- Empty road: lane offset 0.00 m (same as classical).

Why the classical planner fails: each tick it picks the cheapest path on a
discrete grid, and `plan_center_weight` 20 pulls it strongly to the centre.
It stays centred until the obstacle is too close, then finds no feasible
path. The policy learned to start avoiding earlier.

Caveats:
- Single seed, simulation only.
- Evaluation layouts come from the same generator as training (different seeds).
- The classical baseline uses the robot's current tuning (horizon 2–3 s).
  A classical planner with a longer horizon, able to use the 8 m vision
  range, has not been compared yet.

The released model is the 150k checkpoint: the first one with no layout
that the classical planner passes and SAC fails.

**Representative scenarios** (`test_frenet_rl.py --set report --video DIR`).
In the test script both planners pick `Ti` from the same range, 2.0–3.0 s:
the classical planner samples 2.0, 2.2, …, 3.0 s (`np.arange` on the car
stops at 2.8 s), and the policy picks any value in between. The panel shows
5 m ahead. `--video DIR` saves an H.264 side-by-side video per scenario
(`<name>_compare.mp4`).

| # | Scenario | Shows | SAC | Min dist (m) | Mean \|d\| (m) | Classical | Min dist (m) | Mean \|d\| (m) |
|---|---|---|---|---|---|---|---|---|
| 1 | no_obstacle | Lane keeping, empty road | pass | — | 0.000 | pass | — | 0.000 |
| 2 | heading_offset_left | Recovery from a 20° heading error | pass | — | 0.009 | pass | — | 0.013 |
| 3 | obstacle_left0.5 | One off-centre obstacle | pass | 1.13 | 0.052 | pass | 0.81 | 0.023 |
| 4 | obstacle_center | One centred obstacle | pass | 1.01 | 0.102 | **no path** | 1.02 | 0.000 |
| 5 | zigzag trái→phải | Two staggered obstacles, 10 m apart | pass | 1.13 | 0.106 | pass | 0.81 | 0.045 |
| 6 | zigzag sát (4 m) | Two staggered obstacles, 4 m apart | pass | 1.02 | 0.190 | pass | 0.81 | 0.041 |
| 7 | giữa + chặn trái | Centred obstacle, left side blocked | pass | 1.19 | 0.164 | **no path** | 1.10 | 0.001 |
| 8 | gen_ep10_Frenet_fail | Generated cluster, only SAC passes | pass | 0.98 | 0.159 | **no path** | 1.11 | 0.009 |
| 9 | gen_ep39_both_ok | Generated cluster, both pass | pass | 1.04 | 0.418 | pass | 0.82 | 0.055 |
| 10 | gen_ep187_both_fail | Limitation: both fail | **no path** | 0.99 | 0.350 | **no path** | 0.77 | 0.036 |

SAC fails 1/10 and the classical planner 4/10. There are no collisions:
every failure is "no feasible path". SAC swerves wider (higher mean |d|) and
keeps more distance from obstacles (1.0–1.2 m vs 0.81 m). On all 26
scenarios (`--set all`) SAC fails 2/26 and the classical planner 7/26.

Fairness checks:
- **Path length.** With `--horizon 3.0` both planners use the same fixed
  `Ti`, so their paths have the same length (2.8 m). The results do not
  change (2/26 vs 7/26). Adding `Ti` = 3.0 s to the classical grid does not
  change them either. The classical failures are not caused by shorter paths.
- **Obstacle information.** Both planners receive the same obstacles
  (everything within 8 m). The classical cost only penalises obstacles near
  its 2–3 m path, while the SAC gate checks the straight path up to 8 m. Part
  of SAC's earlier reaction comes from this hand-written gate, not from
  learning.

## Known limitations

- Encoder odometry is unfiltered dead reckoning — it accumulates drift,
  especially wheel slip on sharp turns. This is the reason both EKFs exist.
- GPS is only trusted to correct lateral position once, on entry to a
  pre-mapped curve zone — treating it as a continuous correction source was
  found to let a single bad reading teleport the pose by several meters.
- `frenet_optimal_trajectory.py` at the package root is an offline reference
  implementation (curved cubic-spline planner), kept for experimentation —
  it is not wired into the live ROS pipeline.
- The RL lateral planner has only been validated in simulation.
- The simulation integrates `FrenetEKF.predict` unchanged and flips the sign
  of pure pursuit's `angular_z`. The sign convention (+d right, −d left) is
  unchanged. With the robot's settings, this is the only sign combination
  under which the tuned classical planner holds the lane in simulation.
  The change is confined to `sim_predict` in `train_frenet_rl.py`;
  `control/ekf.py` and `control/pure_pursuit.py` are unchanged.
- `plan_clearance` was lowered from 3.2 m to 1.2 m. This also changes the
  classical planner on the car: it now passes closer to obstacles.
- The obstacle vision range used in training is 8 m. The perception config
  has no explicit range limit, so check that obstacles are actually
  detected that far on the car.

## License

MIT — see [LICENSE](LICENSE).
