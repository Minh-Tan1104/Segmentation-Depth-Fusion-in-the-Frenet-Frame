# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the `RL_CAR` ament_python ROS2 package. When built as part of a ROS2
workspace, this directory is normally checked out under `<workspace>/src/`
(e.g. `~/ros2_ws/src/RL_CAR`) — adjust the workspace-root paths in the
commands below to match wherever this repo is cloned. A full architecture
writeup already exists in [README.md](README.md) (Vietnamese) — read it
before making non-trivial changes; this file only adds what the README
doesn't (commands + a terse orientation map).

## What this is

Full self-driving stack for a differential-drive car (ROS2 Humble / Jetson
Orin). Despite the "RL_CAR" name, the pipeline is **rule-based + classical
planning** (YOLO detection/segmentation + Frenet Optimal Trajectory + EKF) —
there is no RL component (no gym env, policy network, or training loop).

## Commands

Build (always required after editing `.py` or `config/rl_car_params.yaml` —
`install/` is a static copy, not a symlink). Run from the workspace root
(the parent of `src/`):

```bash
colcon build --packages-select RL_CAR
```

Run full stack (camera + perception + control + encoder + joystick + GPS).
These scripts wipe `build/`/`install/`/`log/` and rebuild before launching:

```bash
scripts/run_full_stack.sh            # launch_gps:=true by default
scripts/Run_RL.sh                    # camera pipeline only: perception + planner_motion + visualization
```

Launch directly once already built (no rebuild):

```bash
source /opt/ros/humble/setup.bash
source <workspace>/install/setup.bash
ros2 launch RL_CAR full_stack.launch.py launch_gps:=true
ros2 launch RL_CAR rgb_dual_model.launch.py launch_realsense:=false   # e.g. against a rosbag
```

Check perception GPU deps:

```bash
python3 -c "import torch, ultralytics, cv2, scipy; print(torch.__version__, torch.cuda.is_available())"
```

There is no test suite in this repo.

## Configuration

**All node parameters live in one file:** [config/rl_car_params.yaml](config/rl_car_params.yaml).
To change speed, EKF, planner, model paths, topics, GPS, etc., edit this file
— not `launch/*.py` or `node.py`. Both launch files load it for every node.
The `/**` wildcard block holds params shared across `control_node`,
`planner_motion_node`, and `visualization_node` (`plan_*`, `curve_zone_*`) —
edit once there rather than duplicating per-node, since ROS2 YAML has no
anchor/alias support. Rebuild after any edit (see Commands).

## Architecture

7 executables across 6 Python subpackages of one ament_python package. Each
node splits **`logic.py`** (pure Python/numpy/cv2, no `rclpy` import, testable
offline) from **`node.py`** (the `rclpy.Node` — subscribe/publish/param only).

**Core rule: only `control_node` publishes `/cmd_vel`.** Perception and GPS
only supply data; no other node drives the car.

```
joy_pygame_node ──/joy──┐
encoder_node ──/odom──┤
perception_node ──/perception/frenet_state──┤──► control_node ──/cmd_vel──► encoder_node → hoverboard
gps_node ──/gps/route_state──┘                    │
                                                    ├──► visualization_node (read-only debug grid)
planner_motion_node (parallel Frenet planner, viz/rosbag only — NOT in the drive path)
```

- **perception_node** (GPU/CUDA): two YOLO models on RealSense RGB+Depth —
  segmentation for lane lines → Frenet `(d, s, heading)`; detection for
  obstacles → projected into Frenet `(s, x)` with EMA+TTL tracking.
- **control_node**: the brain. Two timers on a `MultiThreadedExecutor` with
  separate callback groups: `_control_tick` (50Hz, FrenetEKF predict from
  `/odom`, publishes mode/ekf_state, drives `/cmd_vel` directly in MANUAL) and
  `_planner_tick` (15Hz, MutuallyExclusive group — runs its own Frenet Optimal
  Planner + pure pursuit + GPS curvature feed-forward, drives `/cmd_vel` in
  AUTO).
- **Two independent EKFs**: [control/ekf.py](control/ekf.py)
  (`FrenetEKF`, state `[s,d,psi,v]`, corrects from camera) is lane-tracking;
  [Gps/route_ekf.py](Gps/route_ekf.py) (`RouteEKF`, state
  `[x,y,psi]`) is route-level, corrects from GPS + anchors laterally from
  `/control/ekf_state`. `control_node` blends GPS `d` in via `gps_d_gain`
  only when vision is stale and GPS confidence is high; `psi` and `s` always
  come from FrenetEKF, never GPS.
- **Frenet Optimal Planner** ([planner_motion/frenet_planner.py](planner_motion/frenet_planner.py)):
  generates candidate trajectories (quintic lateral × quartic longitudinal),
  picks by cost (jerk + time + center offset `k_d` + obstacle avoidance
  `k_obs`). Reference path is **straight** during normal lane following;
  inside a curve zone (`curve_frenet_enable`) `control_node` switches to a
  **curved reference** (`ReferenceCourse`, a scipy CubicSpline over the CSV
  route polyline): state `(s, d, psi_err)` comes from RouteEKF instead of
  FrenetEKF, candidates are converted to map-local coordinates, obstacles are
  transformed from the ego frame. Since RouteEKF measures `(s,d,psi_err)`
  against the raw polyline (piecewise-linear, ~7° heading error vs. the
  spline at sharp curvature) while the planner runs on the spline,
  `_planner_tick_curve` reconstructs the world pose from the polyline
  convention and **re-projects it onto `ReferenceCourse`** so state and
  reference stay in the same frame. Pure pursuit uses a **separate**
  `curve_lookahead_distance` (not the straight-mode `lookahead_distance` —
  the straight value's gain is too aggressive for the larger tracking errors
  seen mid-curve and causes oscillation). Zone exit hands back to vision via
  `FrenetEKF.reset_lateral()` **only once both** the position-based zone exit
  and vision reacquisition (`_vision_fresh()`) are true — exiting on position
  alone can strand the car with neither GPS nor vision authority.
  `control_node` owns its own planner instance (fed by EKF dead-reckoning)
  separate from `planner_motion_node`'s (fed only by live camera Frenet), so
  control can still replan with vision lost. A bench-test mode
  (`curve_frenet_force_test`) drives this whole pipeline from a second
  `RouteEKF` seeded on a fixed CSV curve zone and integrated from real
  `/odom`, for exercising the curve logic without GPS hardware — the
  `_test_ekf` must never anchor from vision (that path is only for gps_node's
  real RouteEKF; see the gotcha below).
- **gps_node**: off by default (`launch_gps:=false`), reads UBX NAV-PVT,
  map-matches against [map/gps_path_2m.csv](map/gps_path_2m.csv)
  via [Gps/map_matcher.py](Gps/map_matcher.py) (`project_xy`,
  `heading_at`, `curvature_at`, `detect_curve_zones`).
- **encoder_node**: owns the hoverboard serial link, decodes wheel RPM,
  integrates differential-drive dead-reckoning odometry
  ([Encoder/logic.py](Encoder/logic.py)) → `/odom` + TF. No
  filtering — accumulates error, especially wheel slip on sharp turns; this is
  exactly why both EKFs exist.
- **visualization_node**: read-only 2×2 debug grid (camera+seg, Frenet/EKF
  planner, encoder trail, GPS route) → `/perception/overlay`. No GPU, no
  control authority.

## Known gotchas

- QoS: RGB/depth subscriptions (`perception_node`) and `visual_frame`
  (`visualization_node`) must stay `RELIABLE` — `BEST_EFFORT` silently drops
  large image frames under load with no error. Small JSON topics
  (`frenet_state`, `visual_frenet`) stay `BEST_EFFORT`.
- `frenet_optimal_trajectory.py` at the package root (curved cubic-spline
  reference planner) is an **offline reference tool**, not part of the live
  ROS pipeline.
- `Encoder/encoder_odom.py`, `Encoder/speed_control.py`,
  `Encoder/wireless.py`, `Gps/Gps.py` are standalone
  debug/offline scripts, not run as part of the stack —
  `encoder_odom.py` imports a nonexistent `control_speed` module and will
  error if run.
- `k_d` (`plan_center_weight`) is shared between the lane-centering and
  speed-tracking cost terms in the planner — you cannot tighten lane
  centering without also tightening speed tracking without editing
  `frenet_planner.py` itself.
- `Gps/node.py`'s `anchor_lateral()` call must stay gated on
  `not self.route_ekf.in_curve_zone`. Inside a curve zone, `control_node`
  blocks vision from correcting `FrenetEKF` (curve mode owns steering), so
  `/control/ekf_state` is just FrenetEKF's straight-frame dead-reckoning
  drifting with the curve's actual curvature — anchoring RouteEKF to that
  every tick feeds it garbage and corrupts `/gps/route_state` mid-curve
  (small drift near zone entry, growing drift near the middle — this was a
  real bug, confirmed by the bench-test mode tracking cleanly while the real
  GPS path didn't).
