# RL_CAR

Stack tự hành hoàn chỉnh cho xe **differential-drive** (2 bánh chủ động, lái bằng
vi sai tốc độ), chạy trên ROS2 Humble / Jetson Orin. Hệ thống hợp nhất 4 nhóm
chức năng — **nhận thức** (camera: YOLO detection + segmentation), **định vị**
(GPS + bản đồ tuyến), **điều khiển** (EKF fusion + Frenet Optimal Planner + pure
pursuit) và **chấp hành** (encoder/hoverboard) — với `control_node` là bộ não
duy nhất sinh lệnh lái `/cmd_vel`.

> **Tên gọi:** dù project tên là RL_CAR, pipeline hiện tại là **rule-based +
> classical planning** (YOLO + Frenet Optimal Trajectory + EKF), **không có
> thành phần reinforcement learning** (không gym env / policy network / training
> loop).

```
                    ┌─────────── Cảm biến ───────────┐
   Tay cầm PS4    Hoverboard     RealSense D4xx    u-blox GPS   map/gps_path_2m.csv
        │             │(encoder)      │(RGB+Depth)      │(UBX)        │(tuyến CSV)
        ▼             ▼               ▼                 ▼             ▼
  joy_pygame     encoder_node   perception_node     gps_node ◄───────┘
     _node       (/odom + TF)   (YOLO seg+detect,   (RouteEKF: encoder
        │             │          Frenet, obstacle)   + GPS + map-match)
        │(/joy)       │              │(/perception/       │(/gps/route_state:
        │             │              │  frenet_state)     │ s,d,psi_err,sigma_d,kappa_ff)
        ▼             ▼              ▼                     ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  control_node  —  BỘ NÃO LÁI XE (nơi DUY NHẤT sinh /cmd_vel)  │
  │  • MANUAL: trục tay cầm  ──►  /cmd_vel trực tiếp              │
  │  • AUTO:  FrenetEKF (encoder⊕camera)  ►  Frenet Optimal        │
  │           Planner  ►  pure pursuit (+GPS curvature feed-fwd)   │
  └──────────────────────────────────────────────────────────────┘
        │(/cmd_vel)                    │(/control/ekf_state, /planner_viz, /mode)
        ▼                              ▼
   encoder_node ──► hoverboard   visualization_node (lưới 2×2 debug, chỉ đọc)
                                       │(/perception/overlay)
                        planner_motion_node ─┘ (Frenet planner song song,
                                                CHỈ phục vụ viz/rosbag)
```

---

## Mục lục

- [Danh sách node](#danh-sách-node)
- [Cấu trúc package](#cấu-trúc-package)
- [Cấu hình tập trung (rl_car_params.yaml)](#cấu-hình-tập-trung-rl_car_paramsyaml)
- [Các tầng xử lý chi tiết](#các-tầng-xử-lý-chi-tiết)
  - [Nhận thức (perception)](#1-nhận-thức-perception)
  - [Điều khiển (control) — 2 tầng EKF + planner + pure pursuit](#2-điều-khiển-control)
  - [Định vị GPS (RouteEKF + map-matching + feed-forward cua)](#3-định-vị-gps)
  - [Odometry encoder](#4-odometry-encoder)
  - [Visualization](#5-visualization)
- [Topic chính](#topic-chính)
- [Launch files](#launch-files)
- [Chạy nhanh](#chạy-nhanh)
- [Tham số đáng chú ý](#tham-số-đáng-chú-ý)
- [QoS — bài học rớt frame](#qos--bài-học-rớt-frame)
- [Lưu ý / hạn chế đã biết](#lưu-ý--hạn-chế-đã-biết)

---

## Danh sách node

7 executable ROS2, chia trong 6 Python package của cùng 1 ament_python package
`RL_CAR`. Mỗi node tách **`logic.py`** (thuần Python/numpy/cv2, không import
rclpy — chứa toàn bộ tính toán, test offline được) và **`node.py`** (lớp `Node`,
chỉ lo subscribe/publish/param).

| Node | Executable | Vai trò | GPU? |
|---|---|---|---|
| **perception_node** | `perception_node` | YOLO detection (vật cản) + segmentation (vạch làn) → Frenet (d, s, heading) → chiếu vật cản vào Frenet. Blend segmentation lên RGB. | ✅ torch/CUDA |
| **planner_motion_node** | `planner_motion_node` | Frenet Optimal Planner chạy **song song** từ `frenet_state` — CHỈ phục vụ visualization/rosbag, KHÔNG nằm trong đường lái xe thật. | ❌ |
| **visualization_node** | `visualization_node` | Ghép lưới 2×2 debug (camera / Frenet-EKF / encoder trail / GPS route) → `/perception/overlay` (+ cửa sổ cv2). Chỉ đọc, không điều khiển. | ❌ |
| **encoder_node** | `encoder_node` | Sở hữu serial hoverboard. Đọc RPM 2 bánh → tích phân differential-drive odometry → `/odom` + TF. Nhận `/cmd_vel` → lệnh raw. | ❌ |
| **control_node** | `control_node` | **Bộ não lái xe.** Toggle manual/auto (nút R1), FrenetEKF, Frenet Optimal Planner nội bộ, pure pursuit, GPS assist. **Nơi duy nhất publish `/cmd_vel`.** | ❌ |
| **joy_pygame_node** | `joy_pygame_node` | Đọc tay cầm PS4 DualShock qua pygame (ổn định hơn joy_node/evdev) → `/joy`. | ❌ |
| **gps_node** | `gps_node` | Đọc UBX NAV-PVT. RouteEKF (encoder predict + GPS correct + neo camera) map-match lên tuyến CSV → `/gps/route_state`. KHÔNG lái xe trực tiếp. | ❌ |

**Nguyên tắc cốt lõi:** chỉ `control_node` được sinh `/cmd_vel`. GPS và
perception chỉ cung cấp dữ liệu; không node nào khác có quyền lái xe.

---

## Cấu trúc package

```
RL_CAR/
├── config/
│   └── rl_car_params.yaml       # ★ NGUỒN CẤU HÌNH DUY NHẤT cho MỌI node
├── perception/
│   ├── model_paths.py           # đường dẫn model YOLO detect/seg (fallback theo máy)
│   ├── logic.py                 # PerceptionLogic: seg + Frenet fit + detection + obstacle tracking
│   └── node.py                  # PerceptionNode
├── control/
│   ├── ekf.py                   # FrenetEKF: fusion odom + quan sát làn (tầng 1)
│   ├── pure_pursuit.py          # compute_cmd_vel + curvature feed-forward
│   ├── joy_pygame_node.py       # đọc PS4 qua pygame (executable joy_pygame_node)
│   │                            # (planner: dùng chung planner_motion.logic.PlannerLogic)
│   └── node.py                  # ControlNode (2 timer: _control_tick + _planner_tick)
├── planner_motion/
│   ├── frenet_planner.py        # Frenet Optimal Trajectory (quintic/quartic polynomial, reference thẳng)
│   ├── logic.py                 # PlannerLogic: wrap FrenetOptimalPlanner (dùng CHUNG cho control + planner_motion)
│   └── node.py                  # PlannerMotionNode (viz/rosbag)
├── Gps/
│   ├── gps_reader.py            # parse UBX NAV-PVT -> GpsSample (pyserial + pyubx2)
│   ├── map_matcher.py           # RouteMapMatcher: nạp CSV, project_xy, heading_at, curvature_at, detect_curve_zones
│   ├── route_ekf.py             # RouteEKF: encoder predict + GPS correct + anchor_lateral (tầng 2)
│   ├── node.py                  # GpsNode
│   ├── Gps.py                   # script xem/thu tuyến offline (matplotlib+contextily), KHÔNG phải node ROS
│   └── sim_route_ekf.py         # sim offline tune RouteEKF
├── Encoder/
│   ├── protocol.py              # wire protocol hoverboard (struct pack/unpack, SPEED_DIVISOR=16)
│   ├── logic.py                 # DifferentialOdometry + cmd_vel_to_raw (không rclpy)
│   ├── node.py                  # EncoderNode
│   └── speed_control.py / wireless.py / encoder_odom.py  # script debug độc lập (KHÔNG chạy trong stack)
├── visualization/
│   ├── logic.py                 # OverlayRenderer: draw_frenet_panel / draw_gps_panel / draw_encoder_panel
│   └── node.py                  # VisualizationNode
├── launch/
│   ├── realsense.launch.py      # driver camera RealSense
│   ├── rgb_dual_model.launch.py # perception + planner_motion + visualization (+ camera)
│   └── full_stack.launch.py     # rgb_dual + joy + encoder + control + gps
├── map/
│   └── gps_path_2m.csv          # tuyến ghi sẵn (lat,lon), ~2m/điểm — dùng cho GPS + panel
├── scripts/
│   ├── Run_RL.sh                # build sạch + launch rgb_dual_model
│   ├── run_full_stack.sh        # build sạch + launch full_stack (launch_gps:=true)
│   ├── export_rgb_frames.py     # trích frame RGB từ rosbag để gán nhãn
│   └── record_rgb_video.py / prepare_rosbag_for_humble.py
├── model/detection/ , model/seg/   # trọng số YOLO (.pt)
├── frenet_optimal_trajectory.py    # bản Frenet reference CONG (cubic spline) — công cụ offline, KHÔNG dùng trong ROS
├── package.xml / setup.py / setup.cfg
└── resource/RL_CAR
```

---

## Cấu hình tập trung (rl_car_params.yaml)

**Toàn bộ tham số của MỌI node nằm trong 1 file:**
[config/rl_car_params.yaml](config/rl_car_params.yaml). Muốn chỉnh tốc độ, EKF,
planner, model, topic, GPS… thì **sửa file này**, không cần đụng `launch/*.py`
hay `node.py`. Cả 2 launch file đều load thẳng file này cho mọi node.

**Khối `/**` (wildcard node name, hỗ trợ từ Foxy):** các tham số `plan_*` +
`curve_zone_*` được `control_node`, `planner_motion_node` và `visualization_node`
dùng CHUNG. ROS2 không hỗ trợ YAML anchor/alias để tham chiếu giá trị, nên khai
báo 1 lần ở khối `/**` (áp cho mọi node) thay vì chép tay từng node — sửa 1 chỗ,
mọi node cùng đổi. Ví dụ khối `/**`:

```yaml
/**:
  ros__parameters:
    plan_speed: 2.0
    plan_max_curvature: 5.0
    plan_center_offset: 0.0
    plan_center_weight: 4.0        # phạt lệch tâm làn (k_d)
    plan_min_horizon_s: 5.0        # horizon polynomial ngang (giảm cong đoạn đầu)
    plan_max_horizon_s: 6.0
    curve_zone_curvature_thresh: 0.05   # ngưỡng |kappa| coi là "trong cua"
    curve_zone_dilate_m: 4.0
```

Sau khi sửa yaml **phải build lại** để copy sang `install/`:

```bash
cd /home/rl/ros2_ws && colcon build --packages-select RL_CAR
```

---

## Các tầng xử lý chi tiết

### 1. Nhận thức (perception)

`perception_node` chạy 2 model YOLO trên RealSense RGB+Depth:

- **Segmentation** tìm vạch làn (`seg_target_class_id`: 0=curb, 1=dashed_yellow_line)
  → fit spline → tính trạng thái **Frenet** `(d, s, heading)` trong frame xe.
- **Detection** tìm vật cản → lấy depth theo percentile trong bbox → chiếu vào
  Frenet `(s_m, x_m)`. Có **obstacle tracking** (EMA + TTL + matching theo
  pixel/Frenet cost) để ổn định vị trí vật cản qua các frame.
- Blend segmentation lên RGB → publish `visual_frame` (ảnh) + `frenet_state`
  (JSON: detections + `frenet_viz`, chưa có quỹ đạo), cùng `header.stamp`.

### 2. Điều khiển (control)

`control_node` là trung tâm. 2 timer chạy trên `MultiThreadedExecutor` với
callback group riêng để không chặn nhau:

- **`_control_tick`** (`control_rate_hz`=50Hz): FrenetEKF `predict()` từ `/odom`
  mỗi tick, publish `/control/mode` + `/control/ekf_state`. Ở mode MANUAL thì
  publish `/cmd_vel` từ trục tay cầm.
- **`_planner_tick`** (`planner_rate_hz`=15Hz, callback group MutuallyExclusive):
  chạy Frenet Optimal Planner + pure pursuit → `/cmd_vel` ở mode AUTO.

**Hai tầng EKF (độc lập, không đụng nhau):**

| | Tầng 1 — FrenetEKF ([control/ekf.py](control/ekf.py)) | Tầng 2 — RouteEKF ([Gps/route_ekf.py](Gps/route_ekf.py)) |
|---|---|---|
| State | `[s, d, psi, v]` (frame làn) | `[x, y, psi]` (frame mét cục bộ của tuyến CSV) |
| Predict | `/odom` (encoder) mỗi tick | `/odom` (encoder) mỗi tick |
| Correct | camera `/perception/frenet_state` (d, heading) | GPS fix (R = hAcc² trung thực) + neo camera qua `anchor_lateral()` |
| Vai trò | bám làn khi CÓ vision | dẫn đường toàn tuyến, dự phòng khi mất vision ở cua |

**Frenet Optimal Planner** ([planner_motion/frenet_planner.py](planner_motion/frenet_planner.py)):
sinh hàng trăm quỹ đạo ứng viên (quintic ngang × quartic dọc), chọn cái tối ưu
theo cost (jerk + thời gian + lệch tâm `k_d` + né vật cản `k_obs`). **Reference
THẲNG** — không nhận hình học cong, chỉ nhận `(c_d, c_d_d)` vô hướng tại thời
điểm hiện tại. `control_node` sở hữu 1 bản planner riêng (chạy từ EKF dead-reckon)
để **vẫn replan được khi mất line**, khác với `planner_motion_node` chỉ replan
khi có frenet đo trực tiếp từ camera.

**Pure pursuit** ([control/pure_pursuit.py](control/pure_pursuit.py)): bám path,
tính `angular_z` từ lateral_error + heading, có **curvature feed-forward** cộng
thẳng vào curvature trước khi clamp (xem GPS bên dưới), rồi giới hạn theo
`max_angular_speed`/`max_wheel_speed`.

### 3. Định vị GPS

`gps_node` chỉ chạy khi bật `launch_gps:=true` (mặc định tắt để test trong
nhà/rosbag không cần module GPS).

- **Đọc UBX NAV-PVT** qua serial trên thread riêng, parse bằng
  [Gps/gps_reader.py](Gps/gps_reader.py).
- **RouteMapMatcher** ([Gps/map_matcher.py](Gps/map_matcher.py)): nạp tuyến
  `map/gps_path_2m.csv` (chỉ đọc cột `lat,lon`), reproject sang frame mét
  azimuthal-equidistant, làm mượt, dựng polyline + arc-length. Cung cấp
  `project_xy`, `heading_at`, **`curvature_at`** (độ cong có dấu), và
  **`detect_curve_zones`** (tìm các đoạn `[s_start, s_end]` là khúc cua).
- **RouteEKF**: predict `/odom`, correct GPS (qua gate fix-type/hAcc/Mahalanobis),
  `anchor_lateral()` liên tục từ `/control/ekf_state` khi vision còn tươi → lúc
  vào cua mất line vẫn có sẵn `d/psi_err` khớp thực tế.
- Publish `/gps/route_state = [s_m, d_m, psi_err, sigma_d, kappa_ff]`.

**Curvature feed-forward (bám cua khi mất vision):** phần tử thứ 5 `kappa_ff` là
độ cong CSV tại `s + curve_ff_preview_m`, **chỉ khác 0 khi vị trí rơi vào 1 curve
zone**. `control_node` cộng `gps_ff_gain × kappa_ff` vào curvature của pure
pursuit → xe đánh lái *theo* cua thay vì *đuổi* cua. Panel Frenet trong
visualization cũng **uốn cong theo `kappa_ff`** để hiển thị khớp hình dạng thật.

**control_node dùng GPS thế nào** ([control/node.py](control/node.py) `_use_gps_route`):
bật `gps_assist_enable` + mất vision quá `gps_vision_stale_s` + `sigma_d <
gps_max_sigma_d`. Khi đó:
- **`d`** trộn tuyến tính giữa FrenetEKF và GPS theo `gps_d_gain`
  (`d = state.d + gps_d_gain × (gps_d − state.d)`; 0.0 = bỏ hẳn GPS, 1.0 = tin
  GPS hoàn toàn).
- **`psi` (heading) LUÔN lấy từ FrenetEKF**, GPS không bao giờ ghi đè — tránh 2
  nguồn heading độc lập trôi lệch nhau.
- **`s`** luôn từ FrenetEKF (GPS không có tương đương).

### 4. Odometry encoder

`encoder_node` đọc feedback hoverboard 18-byte (start `0xCD 0xAB`), giải mã
`speedL/speedR` (chia `SPEED_DIVISOR=16` → RPM), rồi tích phân differential-drive
([Encoder/logic.py](Encoder/logic.py)):

```
rpm_to_mps = 2π·wheel_radius / 60
v     = (vL + vR) / 2
omega = (vR − vL) / track_width
theta += omega·dt ;  x += v·cos(theta)·dt ;  y += v·sin(theta)·dt
```

Dead-reckoning thuần, không lọc — sai số cộng dồn theo quãng đường (đặc biệt
**trượt bánh khi cua gấp** làm phóng đại quãng đường/góc), đây là lý do cần 2
tầng EKF sửa lại bằng camera/GPS. Publish `/odom` (Odometry) + TF `odom→base_link`
theo `odom_publish_rate_hz`=50Hz, tách khỏi vòng đọc serial 150Hz.

### 5. Visualization

`visualization_node` render lưới 2×2 debug (không chạy model, không cần GPU):

```
┌────────────────────┬────────────────────┐
│ Camera + seg overlay│ Frenet / EKF planner│  ← uốn cong theo kappa_ff khi GPS-cua
├────────────────────┼────────────────────┤
│ Encoder trail (odom)│ GPS route + curve   │  ← tô cam đoạn cua (curve zone)
└────────────────────┴────────────────────┘
```

Tự nạp riêng 1 bản `map/gps_path_2m.csv` (RouteMapMatcher độc lập với gps_node)
để vẽ panel GPS. Publish `/perception/overlay` (+ cửa sổ cv2 nếu `use_window=true`).

---

## Topic chính

| Topic | Kiểu | Publisher | Nội dung |
|---|---|---|---|
| `/camera/camera/color/image_raw` | Image | (RealSense) | RGB gốc → perception_node |
| `/camera/camera/aligned_depth_to_color/image_raw` | Image | (RealSense) | depth căn theo RGB |
| `/perception/visual_frame` | Image | perception_node | RGB đã blend segmentation |
| `/perception/frenet_state` | String(JSON) | perception_node | detections + `frenet_viz` (chưa có quỹ đạo) |
| `/perception/frenet/d`, `/heading`, `/coeffs` | Float64(MultiArray) | perception_node | trạng thái Frenet đã EMA |
| `/perception/visual_frenet` | String(JSON) | planner_motion_node | như `frenet_state` + `optimal_path`/`candidate_paths` |
| `/perception/frenet/optimal_path`, `/target_d` | Float64MultiArray/Float64 | planner_motion_node | quỹ đạo tối ưu (viz) |
| `/perception/overlay` | Image | visualization_node | ảnh overlay debug 2×2 |
| `/odom` | Odometry | encoder_node | odometry encoder (+ TF) |
| `/cmd_vel` | Twist | **control_node** | **lệnh lái xe (nguồn duy nhất)** → encoder_node |
| `/joy` | Joy | joy_pygame_node | trạng thái tay cầm PS4 |
| `/control/mode` | String | control_node | "manual" / "auto" |
| `/control/ekf_state` | Float64MultiArray | control_node | `[s, d, psi, v]` FrenetEKF |
| `/control/planner_viz` | String(JSON) | control_node | path planner nội bộ (+ kappa_ff) cho viz |
| `/gps/route_state` | Float64MultiArray | gps_node | `[s_m, d_m, psi_err, sigma_d, kappa_ff]` |

---

## Launch files

| Launch | Node khởi động | Arg |
|---|---|---|
| **rgb_dual_model.launch.py** | camera + perception + planner_motion + visualization | `launch_realsense:=true` (tắt khi chạy rosbag) |
| **full_stack.launch.py** | *include rgb_dual_model* + joy + encoder + control + gps | `launch_gps:=false` (bật GPS khi có module) |

`full_stack.launch.py` include thẳng `rgb_dual_model.launch.py` rồi thêm 4 node
lái xe. `gps_node` tắt mặc định (`launch_gps`) — `control_node` vẫn chạy bình
thường vì `gps_assist_enable` mặc định `false`, không phụ thuộc node này.

---

## Chạy nhanh

```bash
# Full stack (camera + control + encoder + tay cầm), có GPS:
./scripts/run_full_stack.sh                    # build sạch + launch full_stack, launch_gps:=true

# Chỉ pipeline camera (perception + planner + visualization):
./scripts/Run_RL.sh                            # build sạch + launch rgb_dual_model
./scripts/Run_RL.sh use_window:=false          # override tham số launch

# Hoặc launch trực tiếp nếu workspace đã build:
source /opt/ros/humble/setup.bash
source /home/rl/ros2_ws/install/setup.bash
ros2 launch RL_CAR full_stack.launch.py launch_gps:=true
```

Kiểm tra nhanh phụ thuộc GPU cho perception_node:

```bash
python3 -c "import torch, ultralytics, cv2, scipy; print(torch.__version__, torch.cuda.is_available())"
```

---

## Tham số đáng chú ý

Xem đầy đủ trong [config/rl_car_params.yaml](config/rl_car_params.yaml) (có
comment từng dòng). Nhóm quan trọng:

**Frenet planner** (khối `/**`, dùng chung):
- `plan_speed`, `plan_max_curvature`, `plan_road_width`, `plan_clearance`,
  `plan_robot_radius` (né vật cản), `plan_obstacle_weight` (k_obs).
- `plan_center_offset` — +d lệch tâm làn sang phải.
- `plan_center_weight` (k_d) — tăng = bám giữa làn chặt hơn (đánh đổi: bám tốc
  độ chặt hơn, vì k_d dùng chung 2 cụm cost).
- `plan_min_horizon_s` / `plan_max_horizon_s` — horizon polynomial ngang; **tăng
  để giảm độ cong đoạn đầu** khi lệch ngang lớn + center_weight cao (đánh đổi:
  path dài hơn).

**GPS assist** (`control_node`):
- `gps_assist_enable` (mặc định false), `gps_vision_stale_s`, `gps_max_sigma_d`.
- `gps_d_gain` — mức tin GPS d (0 = bỏ, 1 = tin hoàn toàn).
- `gps_ff_gain` — hệ số curvature feed-forward trong cua.

**RouteEKF** (`gps_node`): `q_xy`, `q_psi`, `min_fix_type`, `max_h_acc_m`,
`gps_gate_chi2`, `anchor_sigma_d`, `curve_gps_r_scale` (bớt tin GPS trong cua),
`curve_ff_preview_m`.

**Perception**: `yolo_model_path`/`seg_model_path`, `seg_target_class_id`
(0=curb, 1=dashed_yellow_line), `yolo_conf`/`seg_conf`, `frenet_alpha` (EMA),
`lane_width_m`, obstacle tracking (`obstacle_*`).

**Encoder**: `wheel_radius`, `track_width`, `cmd_speed_scale`/`cmd_steer_scale`
(hiệu chỉnh thực nghiệm), `right_wheel_sign`.

**Visualization**: `use_window`, `display_width`/`height`, `window_scale`.

---

## QoS — bài học rớt frame

Ảnh RGB (1280×720) là message lớn. Nếu subscriber khai `BEST_EFFORT`, gói UDP rớt
do hệ thống bận sẽ **mất vĩnh viễn, không log/error gì** — nhìn như "rớt frame
không rõ lý do". Vì vậy: subscription RGB/depth ở `perception_node` và
`visual_frame` ở `visualization_node` dùng QoS **`RELIABLE`** (DDS tự
retransmit). Các topic JSON nhỏ (`frenet_state`, `visual_frenet`) giữ
`BEST_EFFORT` cho nhẹ.

---

## Lưu ý / hạn chế đã biết

- **Encoder phóng đại quãng đường khi cua gấp** (trượt bánh skid-steer) — hạn chế
  vật lý cố hữu của dead-reckoning, không phải bug; cần EKF camera/GPS sửa lại.
- **Frenet reference thẳng**: planner không "nhìn thấy" khúc cua phía trước từ
  bản đồ — chỉ nhận `(d, psi)` tức thời. GPS curvature feed-forward là cách bù
  gián tiếp (bơm độ cong CSV vào tay lái), không phải reference cong thật.
- **k_d dùng chung 2 cụm cost**: tăng `plan_center_weight` để bám giữa làn cũng
  làm xe bám `plan_speed` chặt hơn — không tách riêng được nếu không sửa sâu
  `frenet_planner.py`.
- `frenet_optimal_trajectory.py` ở thư mục gốc (bản reference CONG dùng cubic
  spline) là **công cụ offline**, KHÔNG nằm trong pipeline ROS đang chạy.
- Các file `Encoder/encoder_odom.py`, `Encoder/speed_control.py`,
  `Encoder/wireless.py`, `Gps/Gps.py` là **script debug/offline độc lập**, không
  chạy trong stack (encoder_odom.py còn import module `control_speed` không tồn
  tại — sẽ lỗi nếu chạy).
- Sau khi sửa `rl_car_params.yaml` hoặc bất kỳ `.py` nào, **phải `colcon build
  --packages-select RL_CAR`** vì `install/` là bản copy tĩnh (không symlink).

