import os
import sys

import  numpy as  np
import math
try:
    from OPTIMAL_TRAJECTORY.CUBIC_PLANNER import cubic_spline_planner
except ModuleNotFoundError:
    from CUBIC_PLANNER import cubic_spline_planner


def import_matplotlib_pyplot(prefer_interactive=True):
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib chưa được cài. Cài bằng `pip install matplotlib` để chạy mô phỏng 2D."
        ) from exc

    attempted = []
    backend_candidates = []
    env_backend = os.environ.get("MPLBACKEND")
    if env_backend:
        backend_candidates.append(env_backend)
    if prefer_interactive:
        backend_candidates.extend(["TkAgg", "Agg"])
    else:
        backend_candidates.append("Agg")

    seen = set()
    for backend in backend_candidates:
        if not backend or backend in seen:
            continue
        seen.add(backend)
        try:
            sys.modules.pop("matplotlib.pyplot", None)
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt

            if prefer_interactive and backend != "Agg":
                plt.ion()
            elif prefer_interactive and backend == "Agg":
                print("GUI backend không khả dụng, dùng matplotlib Agg headless; sẽ không mở cửa sổ animation.")

            return plt
        except Exception as exc:
            attempted.append(f"{backend}: {exc}")

    raise RuntimeError(
        "Không khởi tạo được matplotlib backend an toàn. "
        + " | ".join(attempted)
    )

# Parameter 
MAX_SPEED = 20.0 / 3.6    # maximum speed [m/s]
MAX_ACCEL = 3             # maximum acceleration [m/ss]
MAX_CURVATURE = 10        # maximum curvature (độ cong) [1/m]
MAX_ROAD_WIDTH = 6        # maximum road width [m]
D_ROAD_W = 0.6           # road width sampling length [m]
DT = 0.3                  # time tick [s]
MAXT = 5.0               # max prediction time [m]
MINT = 1.7                # min prediction time [m]
TARGET_SPEED = 18 / 3.6   # target speed [m/s]
D_T_S = 0.1 / 3.6         # target speed sampling length [m/s]
N_S_SAMPLE = 1            # sampling number of target speed
ROBOT_RADIUS = 2.2      # robot radius [m]

# cost weights Trajectory 
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 2.0
KLON = 1.0
KPATH_CHANGE = 2.5              # penalty for changing planned path too much
MIN_PATH_LENGTH = 4.5           # minimum spatial path length [m]
MIN_PATH_POINTS = 4             # minimum number of points in global path

# 2D simulation parameters
WB = 2.7                    # wheel base [m]
MAX_STEER = np.deg2rad(30)  # maximum steering angle [rad]
LOOKAHEAD_DISTANCE = 2.8    # pure pursuit lookahead [m]
SIM_TIME = 45.0             # simulation horizon [s]
GOAL_TOLERANCE = 2.0        # goal threshold [m]
VEHICLE_LENGTH = 4.5        # vehicle body length [m]
VEHICLE_WIDTH = 2.0         # vehicle body width [m]
WHEEL_LENGTH = 0.9          # wheel length [m]
WHEEL_WIDTH = 0.35          # wheel width [m]
WHEEL_TRACK = 1.45          # left-right wheel distance [m]
REAR_AXLE_TO_CENTER = 0.9   # body center offset from rear axle [m]
OBSTACLE_LENGTH = 2.8       # obstacle rectangle length [m]
OBSTACLE_WIDTH = 1.8        # obstacle rectangle width [m]
LANE_COUNT = 2                                 # total visible lanes
LANE_WIDTH = MAX_ROAD_WIDTH / LANE_COUNT       # lane width [m]
ROAD_HALF_WIDTH = (LANE_COUNT * LANE_WIDTH) / 2.0  # half road width [m]
FRENET_DI_MARGIN = D_ROAD_W / 2.0
FRENET_DI_MIN = float(-ROAD_HALF_WIDTH + FRENET_DI_MARGIN)
FRENET_DI_MAX = float(ROAD_HALF_WIDTH - FRENET_DI_MARGIN)
FRENET_DI_VALUES = np.arange(FRENET_DI_MIN, FRENET_DI_MAX + 1e-9, D_ROAD_W)
DEMO_START_D = -LANE_WIDTH / 2.0
DEMO_START_SPEED = TARGET_SPEED * 0.35

# Base reference for conversion
base_latitude  = 10.8532570333 # Latitude of the first waypoint
base_longitude = 106.7715131967  # Longitude of the first waypoint
scaling_factor = 100000

class quintic_polynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):

        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5
        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4
        return xt
    
    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3
        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2
        return xt


class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):

        # calc coefficient of quintic polynomial
        self.xs = xs   # Vị trí ban đầu.
        self.vxs = vxs # Vận tốc ban đầu.
        self.axs = axs # Gia tốc ban đầu.
        self.vxe = vxe # Vận tốc tại thời điểm kết thúc.
        self.axe = axe # Gia tốc tại thời điểm kết thúc.
                       # T: Thời gian di chuyển từ điểm đầu đến điểm cuối.
        self.a0 = xs        # Giá trị tại thời điểm ban đầu.
        self.a1 = vxs       # Hệ số của vận tốc ban đầu.
        self.a2 = axs / 2.0 # Hệ số gia tốc ban đầu (gia tốc chia 2 để phù hợp với công thức đa thức).

        A = np.array([[3 * T ** 2,  4 * T ** 3],
                      [  6 * T   , 12 * T ** 2]])       # Ma trận A là ma trận hệ số của phương trình
        
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                             axe - 2 * self.a2        ]) # Vector b chứa các giá trị đích
        
        x = np.linalg.solve(A, b) # np.linalg.solve(A, b) giải hệ phương trình để tìm giá trị của a3 và a4.

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        """Tính giá trị vị trí x(t) của đa thức tại thời điểm t."""
        
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        """Tính vận tốc v(t), là đạo hàm bậc nhất của x(t)"""
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3
        return xt

    def calc_second_derivative(self, t):
        """Tính gia tốc a(t), là đạo hàm bậc hai của x(t)"""
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2
        return xt

    def calc_third_derivative(self, t):
        """Tính giật j(t) (jerk), là đạo hàm bậc ba của x(t)"""
        xt = 6 * self.a3 + 24 * self.a4 * t
        return xt

class Frenet_path:
    def __init__(self):
        self.t = []     # Danh sách thời gian tương ứng với các điểm trên quỹ đạo.
        self.d = []     # Danh sách giá trị độ lệch ngang (lateral offset) tại mỗi thời điểm t. 
                        # Giá trị này đại diện cho khoảng cách từ quỹ đạo tham chiếu đến quỹ đạo thực tế.
        self.d_d = []   # Vận tốc lệch ngang (lateral velocity) tại mỗi điểm trên quỹ đạo.
        self.d_dd = []  # Gia tốc lệch ngang (lateral acceleration).
        self.d_ddd = [] # Giật lệch ngang (lateral jerk), tức là đạo hàm bậc ba của d theo thời gian.
        
        self.s = []     # Danh sách giá trị dọc theo quỹ đạo tham chiếu tại mỗi thời điểm t. Đây là khoảng cách tích lũy từ điểm đầu quỹ đạo.
        self.s_d = []   # Vận tốc dọc theo quỹ đạo (longitudinal velocity)
        self.s_dd = []  # Gia tốc dọc (longitudinal acceleration)
        self.s_ddd = [] # Giật dọc (longitudinal jerk)
        
        #Cost
        self.cd = 0.0 # Cost liên quan đến chuyển động ngang (lateral motion cost)
        self.cv = 0.0 # Cost liên quan đến chuyển động dọc (longitudinal motion cost)
        self.cf = 0.0 # Tổng chi phí (total cost), được tính dựa trên cd và cv. 
                      # Giá trị này thường được sử dụng để so sánh và chọn quỹ đạo tối ưu

        # Các thông số trong không gian Cartesian (kết quả chuyển đổi từ Frenet)
        self.x = []   # Danh sách tọa độ x trong không gian Cartesian
        self.y = []   # Danh sách tọa độ y trong không gian Cartesian
        self.yaw = [] # Góc phương vị (yaw angle) tại mỗi điểm trong không gian Cartesian
        self.ds = []  # Khoảng cách giữa các điểm liên tiếp trên quỹ đạo trong không gian Cartesian
        self.c = []   # Độ cong (curvature) tại mỗi điểm trên quỹ đạo trong không gian Cartesian


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # Sample đối xứng theo bề rộng mặt đường nhìn thấy.
    # Trước đây range `-1 .. +4.0` bị lệch sang một phía dù road geometry là đối xứng
    # quanh reference path (`ROAD_HALF_WIDTH = 3m`). Điều đó bias classical planner
    # sang các quỹ đạo lệch phải. Giữ margin nửa bước sampling để không chạm mép road.
    di_values = FRENET_DI_VALUES
    Ti_values = np.arange(MINT, MAXT, DT)
    
    # Loop through each lateral offset (di) and time duration (Ti)
    for di in di_values:
        for Ti in Ti_values:
            # Create lateral trajectory using quintic polynomial (for all time steps)
            lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            t_values = np.arange(0.0, Ti, DT)

            # Calculate lateral parameters (fp.d, fp.d_d, fp.d_dd, fp.d_ddd) using vectorized operations
            fp_d = lat_qp.calc_point(t_values)
            fp_d_d = lat_qp.calc_first_derivative(t_values)
            fp_d_dd = lat_qp.calc_second_derivative(t_values)
            fp_d_ddd = lat_qp.calc_third_derivative(t_values)

            # Create longitudinal trajectories for each target speed tv
            tv_values = np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S)

            # Use broadcasting to calculate the longitudinal trajectories in parallel
            for tv in tv_values:
                tfp = Frenet_path()

                # Generate longitudinal trajectory using quartic polynomial
                lon_qp = quartic_polynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                # Calculate longitudinal parameters (tfp.s, tfp.s_d, tfp.s_dd, tfp.s_ddd) using vectorized operations
                tfp_s = lon_qp.calc_point(t_values)
                tfp_s_d = lon_qp.calc_first_derivative(t_values)
                tfp_s_dd = lon_qp.calc_second_derivative(t_values)
                tfp_s_ddd = lon_qp.calc_third_derivative(t_values)

                # Compute costs (Jp, Js, ds)
                Jp = np.sum(np.power(fp_d_ddd, 2))  # Use fp_d_ddd for lateral jerk
                Js = np.sum(np.power(tfp_s_ddd, 2))  # Use tfp_s_ddd for longitudinal jerk
                ds = np.square(TARGET_SPEED - tfp_s_d[-1])

                # Calculate total costs (tfp.cd, tfp.cv, tfp.cf) using vectorized operations
                tfp_cd = KJ * Jp + KT * Ti + KD * np.square(fp_d[-1])  # Cost do độ lệch ngang cuối cùng.
                tfp_cv = KJ * Js + KT * Ti + KD * ds
                tfp_cf = KLAT * tfp_cd + KLON * tfp_cv

                # Assign the calculated values to tfp
                tfp.t = t_values
                tfp.d = fp_d
                tfp.d_d = fp_d_d
                tfp.d_dd = fp_d_dd
                tfp.d_ddd = fp_d_ddd
                tfp.s = tfp_s
                tfp.s_d = tfp_s_d
                tfp.s_dd = tfp_s_dd
                tfp.s_ddd = tfp_s_ddd
                tfp.cd = tfp_cd
                tfp.cv = tfp_cv
                tfp.cf = tfp_cf

                # Append the trajectory to the list
                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    
    for fp in fplist:
        # calc global positions
        # Xử lý từng quỹ đạo Frenet để chuyển đổi sang hệ tọa độ toàn cục.
        for i in range(len(fp.s)):
            
            # Tính vị trí toàn cục (ix,iy) tại khoảng cách s[i] dọc theo đường tham chiếu.
            ix, iy = csp.calc_position(fp.s[i])
            
            if ix is None:
                # Nếu không tính được vị trí (ví dụ: ngoài phạm vi đường tham chiếu), thoát khỏi vòng lặp.
                break
            
            # Tính góc hướng (yaw) tại khoảng cách s[i]
            iyaw = csp.calc_yaw(fp.s[i])
            
            # Sử dụng độ lệch ngang di để tính vị trí toàn cục (fx,fy)
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            
            # Lưu kết quả vào
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        # Tính góc hướng và khoảng cách
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            
            # Sử dụng atan2 để tính góc hướng giữa hai điểm liên tiếp
            # Sử dụng công thức Pythagoras để tính độ dài đoạn thẳng ds giữa hai điểm liên tiếp.
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.sqrt(dx**2 + dy**2))

        if len(fp.x) < 2 or len(fp.yaw) == 0 or len(fp.ds) == 0:
            continue

        # Lưu kết quả vào
        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        # Tính độ cong (curvature)
        for i in range(len(fp.yaw) - 1):
            
            # Độ cong: Tính độ thay đổi của góc hướng (yaw) giữa hai điểm liên tiếp chia cho khoảng cách ds
            # Độ cong này biểu diễn mức độ uốn cong của quỹ đạo tại mỗi điểm
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
    # - Danh sách các quỹ đạo với thông tin toàn cục đầy đủ bao gồm:
    #    + Vị trí toàn cục (x,y).
    #    + Góc hướng yaw.
    #    + Khoảng cách ds.
    #    + Độ cong c.
    return fplist



def check_collision(fp, ob):
    """
    Kiểm tra va chạm nhưng chỉ xét 3/4 quỹ đạo phía xa nhất của xe.

    Parameters:
        fp: Đối tượng Frenet_path chứa danh sách tọa độ (x, y).
        ob: Mảng NumPy chứa danh sách tọa độ (x, y) của các chướng ngại vật.

    Returns:
        True nếu 3/4 quỹ đạo phía xa an toàn, False nếu có va chạm.
    """
    path_length = len(fp.x)
    three_fourths_index = path_length // 4  # Lấy 1/4 đầu bỏ đi, lấy 3/4 sau
    
    # Chỉ lấy 3/4 quỹ đạo phía xa
    x_far = fp.x[three_fourths_index:]  
    y_far = fp.y[three_fourths_index:]

    for i in range(ob.shape[0]):  # Duyệt qua từng chướng ngại vật
        x_ob, y_ob = ob[i]  # Tọa độ của vật cản

        # Kiểm tra va chạm chỉ với 3/4 quỹ đạo phía xa
        collision = any((ix - x_ob) ** 2 + (iy - y_ob) ** 2 <= ROBOT_RADIUS**2 
                        for ix, iy in zip(x_far, y_far))

        if collision:
            return False  # Có va chạm, quỹ đạo không an toàn

    return True  # Không có va chạm, quỹ đạo an toàn


def calc_path_length(fp):
    if len(fp.ds) == 0:
        return 0.0
    return float(np.sum(fp.ds))


def reaches_goal_region(fp, csp, goal_margin=1.0):
    if len(fp.s) == 0:
        return False
    return fp.s[-1] >= csp.s[-1] - goal_margin


def allow_short_path_near_goal(fp, csp):
    if csp is None or len(fp.s) == 0:
        return False
    remaining_distance = csp.s[-1] - fp.s[0]
    return remaining_distance <= max(MIN_PATH_LENGTH, LOOKAHEAD_DISTANCE * 2.0)


def calc_path_change_cost(fp, prev_path, shift_steps=1):
    if prev_path is None or len(prev_path.x) <= shift_steps or len(fp.x) < 2:
        return 0.0

    overlap = min(len(fp.x), len(prev_path.x) - shift_steps)
    if overlap < 2:
        return 0.0

    current_xy = np.column_stack((fp.x[:overlap], fp.y[:overlap]))
    prev_xy = np.column_stack((
        prev_path.x[shift_steps:shift_steps + overlap],
        prev_path.y[shift_steps:shift_steps + overlap],
    ))

    position_change = np.mean(np.linalg.norm(current_xy - prev_xy, axis=1))
    lateral_change = abs(fp.d[min(overlap - 1, len(fp.d) - 1)] - prev_path.d[min(shift_steps + overlap - 1, len(prev_path.d) - 1)])
    return float(position_change + 0.5 * lateral_change)


def check_paths(fplist, ob, csp=None):
    """Hàm check_paths được sử dụng để kiểm tra các quỹ đạo (paths)
    trong danh sách fplist dựa trên các ràng buộc động học và tránh va chạm. 
    Chỉ những quỹ đạo hợp lệ (thỏa mãn các điều kiện) mới được giữ lại."""

    # Danh sách các đối tượng thuộc lớp Frenet_path, 
    # mỗi đối tượng đại diện cho một quỹ đạo với các thuộc tính như vận tốc, gia tốc, độ cong, v.v.
    # ob: Mảng NumPy chứa tọa độ của các chướng ngại vật trong môi trường.

    # Mảng này lưu trữ chỉ số của các quỹ đạo hợp lệ trong fplist
    okind = []
    
    # Kiểm tra từng quỹ đạo
    for i in range(len(fplist)):
        path_length = calc_path_length(fplist[i])
        goal_reachable = csp is not None and reaches_goal_region(fplist[i], csp)
        short_path_allowed = csp is not None and allow_short_path_near_goal(fplist[i], csp)
        
        # Kiểm tra điều kiện động học

        if len(fplist[i].x) < MIN_PATH_POINTS and not short_path_allowed:
            continue
        elif path_length < MIN_PATH_LENGTH and not goal_reachable and not short_path_allowed:
            continue
        elif any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            # Giới hạn vận tốc tối đa
            # print("Max speed check")
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            # Giới hạn độ cong tối đa
            # print("Max accel check")
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            # Giới hạn độ cong tối đa
            # print("Max curvature check")
            continue
        elif not check_collision(fplist[i], ob):
            # Nếu quỹ đạo va chạm với bất kỳ chướng ngại vật nào, bỏ qua quỹ đạo.
            # print("check_collision")
            continue
        # Nếu bất kì điều kiện nào không thỏa thì bỏ qua path đó
        # Lưu các quỹ đạo hợp lệ thỏa mãn tất cả điều kiện trên, thêm chỉ số i vào danh sách okind.
        okind.append(i)
    # Trích xuất các quỹ đạo hợp lệ từ danh sách fplist dựa trên chỉ số trong okind
    return [fplist[i] for i in okind]

def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, prev_path=None):
    """ Hàm frenet_optimal_planning thực hiện quá trình lập kế hoạch chuyển động tối ưu trong hệ tọa độ Frenet, 
        dựa trên các thông tin đầu vào về trạng thái xe, đường cong tham chiếu, và các chướng ngại vật. 
        Hàm trả về quỹ đạo tối ưu dựa trên tiêu chí chi phí tối thiểu. """

    # - Các tham số đầu vào
    #     + csp: Đường cong tham chiếu (Cubic Spline Path) trong hệ tọa độ toàn cục.
    #     + Dùng để tính toán vị trí và hướng toàn cục từ tọa độ Frenet.

    #     + s0:Vị trí dọc (longitudinal) ban đầu của xe trên đường tham chiếu.
    #     + c_speed:Vận tốc dọc ban đầu của xe.

    #     + c_d, c_d_d, c_d_dd:
    #     + Các trạng thái ngang (lateral state):
    #     + c_d: vị trí ngang ban đầu.
    #     + c_d_d: vận tốc ngang ban đầu.
    #     + c_d_dd: gia tốc ngang ban đầu.

    #     ob:Mảng NumPy chứa tọa độ của các chướng ngại vật.

    # Tính toán các quỹ đạo trong hệ tọa độ Frenet
    # Hàm calc_frenet_paths tạo ra một danh sách các quỹ đạo ứng viên trong hệ tọa độ Frenet, 
    # bao gồm cả chuyển động ngang (lateral) và dọc (longitudinal).
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    
    # Chuyển đổi quỹ đạo sang hệ tọa độ toàn cục
    # Sử dụng đường cong tham chiếu csp, hàm calc_global_paths tính toán vị trí toàn cục (x,y), góc lái (yaw), 
    # và độ cong cho từng quỹ đạo trong danh sách.
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob, csp=csp)

    # find minimum cost path
    mincost = float("inf")
    bestpath = None
    for fp in fplist:
        continuity_cost = KPATH_CHANGE * calc_path_change_cost(fp, prev_path)
        total_cost = fp.cf + continuity_cost
        fp.path_change_cost = continuity_cost
        fp.total_cost = total_cost

        if mincost >= total_cost:
            mincost = total_cost
            bestpath = fp
    # print(f"num path: {len(fplist)}")
    return bestpath, fplist

def generate_target_course(x, y):
    
    """Hàm generate_target_course tạo ra đường dẫn tham chiếu (target course) từ một tập hợp các điểm (x,y). 
        Đường dẫn này được biểu diễn dưới dạng spline 2D và bao gồm:
        các thuộc tính như tọa độ (x,y), góc định hướng (yaw), và độ cong (curvature)."""
    
    # cubic_spline_planner.Spline2D tạo spline hai chiều dựa trên các điểm x và y
    # Spline 2D cung cấp các hàm tính toán vị trí, góc yaw, và độ cong cho bất kỳ giá trị dọc s.
    csp = cubic_spline_planner.Spline2D(x, y)
    
    # Tạo danh sách các giá trị dọc s
    # Mỗi giá trị s đại diện cho một vị trí dọc (longitudinal position) dọc theo spline
    # Khoảng cách giữa các giá trị s là 0.1, đảm bảo độ phân giải cao.
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        # Tính tọa độ (x,y) tại vị trí s.
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        # Tính góc định hướng (yaw) tại s
        ryaw.append(csp.calc_yaw(i_s))
        # Tính độ cong (curvature) tại s
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp

def cartesian_to_frenet(x, y, yaw, csp):
    """
    Chuyển đổi tọa độ Cartesian sang Frenet dựa trên đường spline tham chiếu.

    Args:
        x (float): Tọa độ x của xe.
        y (float): Tọa độ y của xe.
        yaw (float): Góc phương vị của xe (radians).
        csp (CubicSpline2D): Đường spline tham chiếu.

    Returns:
        s (float): Vị trí dọc theo spline.
        d (float): Khoảng cách vuông góc đến spline.
        d_d (float): Tốc độ vuông góc.
        d_dd (float): Gia tốc vuông góc.
    """
    # Tạo danh sách giá trị s
    s_min = 0.0
    s_max = csp.s[-1]
    ds = 0.1  # Bước tìm kiếm
    s_values = np.arange(s_min, s_max, ds)

    # Tính toán vị trí (x, y) trên spline cho tất cả giá trị s
    positions = np.array([csp.calc_position(s) for s in s_values])
    x_spline, y_spline = positions[:, 0], positions[:, 1]

    # Tính khoảng cách từ điểm (x, y) đến tất cả các điểm trên spline
    distances = np.hypot(x_spline - x, y_spline - y)

    # Tìm giá trị s tương ứng với khoảng cách nhỏ nhất
    min_idx = np.argmin(distances)
    s_best = s_values[min_idx]

    # Tính toán các giá trị dựa trên s_best
    s = s_best
    x_ref, y_ref = csp.calc_position(s)
    yaw_ref = csp.calc_yaw(s)
    d = np.hypot(x - x_ref, y - y_ref)

    # Xác định hướng của d (trái hay phải spline)
    cross_product = (x - x_ref) * -np.sin(yaw_ref) + (y - y_ref) * np.cos(yaw_ref)
    if cross_product < 0:
        d *= -1

    # Tính tốc độ và gia tốc vuông góc
    dx = x - x_ref
    dy = y - y_ref
    v_ref = np.array([np.cos(yaw_ref), np.sin(yaw_ref)])
    v_cartesian = np.array([dx, dy])

    # Tốc độ vuông góc
    d_d = np.dot(v_cartesian, [-v_ref[1], v_ref[0]])

    # Gia tốc vuông góc (giả sử không có dữ liệu gia tốc thêm)
    d_dd = 0.0

    return s, d, d_d, d_dd

def project_onto_path(x, y, tx, ty):
    """
    Project the point (x, y) onto the closest segment of the target course (tx, ty).
    Returns the projected point (px, py).
    """
    # Chuyển dữ liệu thành mảng NumPy
    segments_start = np.stack([tx[:-1], ty[:-1]], axis=1)
    segments_end = np.stack([tx[1:], ty[1:]], axis=1)
    segment_vecs = segments_end - segments_start

    # Tính vector từ mỗi điểm bắt đầu đoạn thẳng đến điểm cần chiếu
    point_vecs = np.array([x, y]) - segments_start

    # Tính độ dài của các đoạn thẳng
    segment_lengths = np.linalg.norm(segment_vecs, axis=1)
    nonzero_mask = segment_lengths > 0  # Tránh chia cho 0

    # Tính tỉ lệ chiếu của điểm lên các đoạn thẳng
    projections = np.einsum('ij,ij->i', point_vecs, segment_vecs)  # Dot product
    projections = np.divide(projections, segment_lengths, where=nonzero_mask)
    projections = np.clip(projections, 0, segment_lengths)  # Clip vào đoạn thẳng

    # Tính các điểm chiếu
    projected_points = segments_start + (projections[:, None] / segment_lengths[:, None]) * segment_vecs

    # Tính khoảng cách từ điểm gốc đến các điểm chiếu
    distances = np.linalg.norm(projected_points - np.array([x, y]), axis=1)

    # Lấy điểm chiếu gần nhất
    min_idx = np.argmin(distances)
    closest_point = projected_points[min_idx]

    return closest_point

def calculate_heading_from_gps(lat1, lon1, lat2, lon2):
    """
    Tính heading giữa 2 điểm GPS trong hệ tọa độ XY.
    Trả về heading với độ chính xác cao.
    """
    # Bán kính Trái Đất (m)
    R = 6371e3  

    # Chuyển lat/lon sang radians
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    # Tính khoảng cách x, y trong hệ tọa độ phẳng (equirectangular projection)
    dx = R * (lon2 - lon1) * math.cos(lat1)
    dy = R * (lat2 - lat1)

    # Tính heading trong hệ XY (theo góc arctan2)
    heading_xy = math.atan2(dy, dx)  # Trả về radian
    heading_xy_deg = math.degrees(heading_xy)

    # Đảm bảo góc trong khoảng [0, 360]
    if heading_xy_deg < 0:
        heading_xy_deg += 180

    return heading_xy_deg

def course_to_waypoint(current_lat, current_long, target_lat, target_long):
    """Calculate the heading to the target waypoint."""
    dlon = math.radians(float(target_long) - float(current_long))
    c_lat = math.radians(float(current_lat))
    t_lat = math.radians(float(target_lat))
    a1 = math.sin(dlon) * math.cos(t_lat)
    a2 = math.sin(c_lat) * math.cos(t_lat) * math.cos(dlon)
    a2 = math.cos(c_lat) * math.sin(t_lat) - a2
    a2 = math.atan2(a1, a2)
    if a2 < 0.0:
        a2 += math.pi * 2
    return math.degrees(a2)


def find_lookahead_point(x, y, lookahead_distance, path_x, path_y):
    """
    Tìm giao điểm giữa đường tròn (tâm xe, bán kính lookahead_distance) 
    và đoạn thẳng nối hai waypoint gần nhất.
    """
    for i in range(len(path_x) - 1):
        x1, y1 = path_x[i], path_y[i]
        x2, y2 = path_x[i + 1], path_y[i + 1]

        # Hệ số của phương trình đường thẳng y = ax + b
        if x2 - x1 == 0:
            continue  # Bỏ qua nếu đường thẳng thẳng đứng

        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        # Phương trình đường tròn: (x - x0)^2 + (y - y0)^2 = r^2
        # Kết hợp với y = ax + b, ta có phương trình bậc 2 Ax^2 + Bx + C = 0
        A = 1 + a**2
        B = 2 * (a * (b - y) - x)
        C = x**2 + (b - y)**2 - lookahead_distance**2

        # Giải phương trình bậc 2
        delta = B**2 - 4 * A * C
        if delta < 0:
            continue  # Không có giao điểm thực

        sqrt_delta = np.sqrt(delta)
        x_sol1 = (-B + sqrt_delta) / (2 * A)
        x_sol2 = (-B - sqrt_delta) / (2 * A)
        y_sol1 = a * x_sol1 + b
        y_sol2 = a * x_sol2 + b

        # Chọn điểm phía trước xe trên đường đi
        if x1 <= x_sol1 <= x2 or x1 >= x_sol1 >= x2:
            return np.array([x_sol1, y_sol1])
        if x1 <= x_sol2 <= x2 or x1 >= x_sol2 >= x2:
            return np.array([x_sol2, y_sol2])

    return np.array([path_x[-1], path_y[-1]])  # Nếu không tìm thấy, chọn điểm cuối

def pure_pursuit_control_frenet(lat, lon, optimal_path, x, y, yaw, lookahead_distance, WB):
    path_x, path_y = optimal_path.x, optimal_path.y

    # Tìm điểm lookahead dựa vào giao điểm giữa đường tròn và đường thẳng
    lookahead_point = find_lookahead_point(x, y, lookahead_distance, path_x, path_y)

    # Tính góc alpha giữa hướng xe và lookahead point: radian
    alpha = np.arctan2(lookahead_point[1] - y, lookahead_point[0] - x) - yaw

    # Tính góc lái bằng công thức Pure Pursuit
    steering_angle = -np.arctan2(2.0 * WB * np.sin(alpha), lookahead_distance) * 180/np.pi
    steering_angle = np.clip(steering_angle, -30, 30)

    return steering_angle, alpha 

def calculate_speed_at_projected_point(projected_point, nearest_idx, csp):
    """
    Calculate the speed at the projected point on the target course.
    """
    # Speed at the nearest point in the course (csp is the speed profile)
    speed_at_point = csp[nearest_idx]
    return speed_at_point

def lat_lon_to_xy(lat, lon, lat0=10.8532570333, lon0=106.7715131967):
    """
    Convert latitude and longitude to x, y using an equirectangular projection.

    lat0 and lon0 represent the center point (origin) for the projection.
    """
    R = 6371e3  # Bán kính Trái Đất (mét)
    x = R * math.radians(lon0 - lon) * math.cos(math.radians(lat0))
    y = R * math.radians(lat0 - lat)
    return x, y

def xy_to_lat_lon(x, y):
    """Convert Cartesian coordinates (X, Y) back to latitude and longitude."""
    lon = (x / scaling_factor) + base_longitude
    lat = (y / scaling_factor) + base_latitude
    return lat, lon

def convert_yaw(yaw_deg_system1, yaw_offset = 92):
    """
    Chuyển đổi góc yaw từ hệ tọa độ 1 sang hệ tọa độ 2.
    
    Args:
        yaw_deg_system1 (float): Góc yaw trong hệ tọa độ 1 (độ).
        yaw_offset (float): Độ lệch góc yaw giữa hệ tọa độ 1 và 2 (độ, mặc định là 100).
        
    Returns:
        float: Góc yaw trong hệ tọa độ 2 (độ).
    """
    # Chuyển góc từ độ sang radian
    yaw_rad_system1 = np.deg2rad(yaw_deg_system1)
    yaw_offset_rad  = np.deg2rad(yaw_offset)
    
    # Tính góc yaw trong hệ tọa độ 2
    yaw_rad_system2 = yaw_rad_system1 - yaw_offset_rad
    
    # Đảm bảo góc nằm trong khoảng [0, 360) độ
    yaw_deg_system2 = (90 + (90 - np.rad2deg(yaw_rad_system2))) % 360
    
    return yaw_deg_system2


def calculate_heading_from_gps(lat1, lon1, lat2, lon2):
    """
    Tính heading giữa 2 điểm GPS trong hệ tọa độ XY.
    Trả về heading với độ chính xác cao.
    """
    # Bán kính Trái Đất (m)
    R = 6371e3  


    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)


    dx = R * (lon2 - lon1) * math.cos(lat1)
    dy = R * (lat2 - lat1)

 
    heading_xy = math.atan2(dy, dx)  
    heading_xy_cvt = convert_yaw(heading_xy, yaw_offset=90)
    
    heading_xy_deg = (heading_xy_cvt + 360) % 360 
    
    return heading_xy_deg

def transform_obstacle_to_global(x_vehicle, y_vehicle, heading, x_obstacle, y_obstacle):
    # Ma trận chuyển đổi
    T = np.array([
        [np.cos(heading), -np.sin(heading), x_vehicle],
        [np.sin(heading),  np.cos(heading), y_vehicle],
        [     0,                0,              1    ]
    ])
    

    obstacle_local = np.array([x_obstacle, -y_obstacle , 1])
    

    obstacle_global = np.dot(T, obstacle_local)
    

    return obstacle_global[0], obstacle_global[1]


class SimulationState:
    def __init__(self, x, y, yaw, v):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.steer = 0.0

    def update(self, accel, steer):
        steer = np.clip(steer, -MAX_STEER, MAX_STEER)
        self.steer = steer
        self.x += self.v * math.cos(self.yaw) * DT
        self.y += self.v * math.sin(self.yaw) * DT
        self.yaw += self.v * math.tan(steer) * DT / WB
        self.yaw = normalize_angle(self.yaw)
        self.v = np.clip(self.v + accel * DT, 0.0, MAX_SPEED)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def state_to_frenet(state, csp, s_step=0.2):
    s_values = np.arange(0.0, csp.s[-1], s_step)
    positions = np.array([csp.calc_position(s) for s in s_values])

    dx = positions[:, 0] - state.x
    dy = positions[:, 1] - state.y
    distances = np.hypot(dx, dy)
    min_idx = int(np.argmin(distances))

    s = float(s_values[min_idx])
    ref_x, ref_y = positions[min_idx]
    ref_yaw = csp.calc_yaw(s)

    tangent = np.array([math.cos(ref_yaw), math.sin(ref_yaw)])
    normal = np.array([-math.sin(ref_yaw), math.cos(ref_yaw)])
    relative_pos = np.array([state.x - ref_x, state.y - ref_y])

    d = float(np.dot(relative_pos, normal))
    heading_error = normalize_angle(state.yaw - ref_yaw)
    s_d = max(0.1, state.v * math.cos(heading_error))
    d_d = state.v * math.sin(heading_error)
    d_dd = 0.0

    return s, s_d, d, d_d, d_dd


def frenet_point_to_global(csp, s, d):
    x_ref, y_ref = csp.calc_position(s)
    yaw_ref = csp.calc_yaw(s)
    x = x_ref + d * math.cos(yaw_ref + math.pi / 2.0)
    y = y_ref + d * math.sin(yaw_ref + math.pi / 2.0)
    return float(x), float(y)


def pure_pursuit_control_2d(state, optimal_path, lookahead_distance=LOOKAHEAD_DISTANCE):
    lookahead_point = find_lookahead_point(
        state.x, state.y, lookahead_distance, optimal_path.x, optimal_path.y
    )
    alpha = normalize_angle(
        math.atan2(lookahead_point[1] - state.y, lookahead_point[0] - state.x) - state.yaw
    )
    steer = math.atan2(2.0 * WB * math.sin(alpha), lookahead_distance)
    steer = np.clip(steer, -MAX_STEER, MAX_STEER)
    return steer, lookahead_point


def proportional_speed_control(target_speed, current_speed, gain=1.0):
    return np.clip(gain * (target_speed - current_speed), -MAX_ACCEL, MAX_ACCEL)


def rotate_points(points, yaw):
    rotation = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw),  math.cos(yaw)],
    ])
    return points @ rotation.T


def rectangle_outline(center_x, center_y, length, width, yaw):
    local_corners = np.array([
        [ length / 2.0,  width / 2.0],
        [ length / 2.0, -width / 2.0],
        [-length / 2.0, -width / 2.0],
        [-length / 2.0,  width / 2.0],
        [ length / 2.0,  width / 2.0],
    ])
    rotated = rotate_points(local_corners, yaw)
    rotated[:, 0] += center_x
    rotated[:, 1] += center_y
    return rotated


def draw_rectangle(ax, center_x, center_y, length, width, yaw, edgecolor="k", facecolor="none", linewidth=1.5, alpha=1.0):
    outline = rectangle_outline(center_x, center_y, length, width, yaw)
    ax.fill(outline[:, 0], outline[:, 1], facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)


def offset_path(path_x, path_y, path_yaw, offset):
    path_x = np.asarray(path_x)
    path_y = np.asarray(path_y)
    path_yaw = np.asarray(path_yaw)

    offset_x = path_x - offset * np.sin(path_yaw)
    offset_y = path_y + offset * np.cos(path_yaw)
    return offset_x, offset_y


def fill_lane_band(ax, path_x, path_y, path_yaw, inner_offset, outer_offset, facecolor, alpha):
    inner_x, inner_y = offset_path(path_x, path_y, path_yaw, inner_offset)
    outer_x, outer_y = offset_path(path_x, path_y, path_yaw, outer_offset)
    band_x = np.concatenate([inner_x, outer_x[::-1]])
    band_y = np.concatenate([inner_y, outer_y[::-1]])
    ax.fill(band_x, band_y, facecolor=facecolor, edgecolor="none", alpha=alpha, zorder=0.5)


def draw_lane_markings(ax, path_x, path_y, path_yaw):
    left_edge_x, left_edge_y = offset_path(path_x, path_y, path_yaw, ROAD_HALF_WIDTH)
    right_edge_x, right_edge_y = offset_path(path_x, path_y, path_yaw, -ROAD_HALF_WIDTH)

    road_x = np.concatenate([left_edge_x, right_edge_x[::-1]])
    road_y = np.concatenate([left_edge_y, right_edge_y[::-1]])

    ax.fill(road_x, road_y, facecolor="white", edgecolor="none", alpha=1.0, zorder=0)
    ax.plot(left_edge_x, left_edge_y, color="black", linewidth=2.4, zorder=1)
    ax.plot(right_edge_x, right_edge_y, color="black", linewidth=2.4, zorder=1)

    center_x, center_y = offset_path(path_x, path_y, path_yaw, 0.0)
    ax.plot(
        center_x,
        center_y,
        linestyle=(0, (10, 10)),
        color="black",
        linewidth=2.2,
        zorder=1,
    )


def draw_vehicle(ax, state):
    body_center_x = state.x + REAR_AXLE_TO_CENTER * math.cos(state.yaw)
    body_center_y = state.y + REAR_AXLE_TO_CENTER * math.sin(state.yaw)

    draw_rectangle(
        ax,
        body_center_x,
        body_center_y,
        VEHICLE_LENGTH,
        VEHICLE_WIDTH,
        state.yaw,
        edgecolor="k",
        facecolor="white",
        linewidth=2.0,
        alpha=1.0,
    )

    wheel_centers_local = [
        (-0.15,  WHEEL_TRACK / 2.0),
        (-0.15, -WHEEL_TRACK / 2.0),
        (WB,     WHEEL_TRACK / 2.0),
        (WB,    -WHEEL_TRACK / 2.0),
    ]

    rear_wheels = wheel_centers_local[:2]
    front_wheels = wheel_centers_local[2:]

    for local_x, local_y in rear_wheels:
        wheel_center = rotate_points(np.array([[local_x, local_y]]), state.yaw)[0] + np.array([state.x, state.y])
        draw_rectangle(
            ax,
            wheel_center[0],
            wheel_center[1],
            WHEEL_LENGTH,
            WHEEL_WIDTH,
            state.yaw,
            edgecolor="k",
            facecolor="black",
            linewidth=1.2,
        )

    for local_x, local_y in front_wheels:
        wheel_center = rotate_points(np.array([[local_x, local_y]]), state.yaw)[0] + np.array([state.x, state.y])
        draw_rectangle(
            ax,
            wheel_center[0],
            wheel_center[1],
            WHEEL_LENGTH,
            WHEEL_WIDTH,
            state.yaw + state.steer,
            edgecolor="k",
            facecolor="black",
            linewidth=1.2,
        )

    ax.plot(state.x, state.y, "ko", markersize=4)
    heading_tip_x = body_center_x + (VEHICLE_LENGTH / 2.0) * math.cos(state.yaw)
    heading_tip_y = body_center_y + (VEHICLE_LENGTH / 2.0) * math.sin(state.yaw)
    ax.plot([state.x, heading_tip_x], [state.y, heading_tip_y], color="k", linewidth=1.5)


def draw_obstacles(ax, obstacles):
    for ox, oy in obstacles:
        draw_rectangle(
            ax,
            ox,
            oy,
            OBSTACLE_LENGTH,
            OBSTACLE_WIDTH,
            0.0,
            edgecolor="black",
            facecolor="black",
            linewidth=1.8,
            alpha=1.0,
        )


def build_demo_course():
    # Đồng bộ với RL_frenet.py: bump lên +5 rồi xuống -5, tổng 185m
    base_wx = [0.0, 10.0, 20.0, 35.0, 50.0, 65.0, 80.0, 95.0,
               110.0, 125.0, 140.0, 155.0, 170.0, 185.0]
    base_wy = [0.0, 0.0, 2.0, 5.0, 5.0, 2.0, 0.0, 0.0,
               -2.0, -5.0, -5.0, -2.0, 0.0, 0.0]
    segment_length = base_wx[-1] - base_wx[0]
    wx = base_wx + [x + segment_length for x in base_wx[1:]]
    wy = base_wy + base_wy[1:]
    return generate_target_course(wx, wy)


def build_straight_course(length=200.0, num_points=11):
    """Reference thẳng (y=0) khớp với perception thật: s = đoạn thẳng phía trước,
    d = lệch ngang, heading ≈ 0. Dùng nhiều điểm thẳng hàng để Spline2D ổn định."""
    wx = list(np.linspace(0.0, float(length), int(num_points)))
    wy = [0.0] * int(num_points)
    return generate_target_course(wx, wy)


def build_demo_obstacles():
    try:
        from RL_frenet import build_fixed_demo_obstacles
        return build_fixed_demo_obstacles()
    except ImportError:
        return np.array([
            [18.0, 3.0],
            [50.0, 2.0],
            [60.0, 5.5],
            [80.0, 2.5],
            [100.0, -2.5],
        ], dtype=np.float32)


def obstacles_from_frenet(csp, frenet_obstacles, s_offset=0.0, flip_d=False):
    """Chuyển danh sách obstacle ở dạng Frenet (s, d) -> global (x, y) bám theo csp.

    Dùng để đưa obstacle THẬT (từ perception, đo theo s = khoảng cách phía trước,
    d = lệch ngang) vào mô phỏng 2D vốn chỉ nhận obstacle global (x, y).

    Args:
        csp: Spline2D tham chiếu của sim (lấy từ build_demo_course()).
        frenet_obstacles: iterable các cặp (s, d) [mét].
        s_offset: cộng thêm vào s (vd: vị trí s hiện tại của xe khi snapshot).
                  Perception đo s tương đối từ xe, sim bắt đầu ở s≈0 nên mặc định 0.
        flip_d: True nếu quy ước dấu d của perception ngược với sim (sim: +d = trái).

    Returns:
        np.ndarray shape (N, 2) toạ độ global (x, y), đã clamp s trong phạm vi course.
    """
    s_max = csp.s[-1]
    points = []
    for s, d in frenet_obstacles:
        s_clamped = float(np.clip(s + s_offset, 0.0, s_max))
        d_val = -float(d) if flip_d else float(d)
        x, y = frenet_point_to_global(csp, s_clamped, d_val)
        points.append([x, y])
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def load_perception_obstacles(json_path, use_filtered=True):
    """Đọc obstacle (s, d) từ JSON do rgb_dual_model_node publish trên topic `detections`.

    Payload có dạng {"detections": [{"frenet": {"s_m":..., "d_m":...,
    "s_m_filtered":..., "d_m_filtered":...}}, ...]}.
    Có thể là 1 message (dict) hoặc nhiều message (list các dict) đã ghi ra file.

    Returns: list các cặp (s, d) [mét].
    """
    import json

    with open(json_path, "r") as f:
        data = json.load(f)

    messages = data if isinstance(data, list) else [data]
    sd_list = []
    for msg in messages:
        for det in msg.get("detections", []):
            fr = det.get("frenet")
            if not fr:
                continue
            if use_filtered:
                s = fr.get("s_m_filtered", fr.get("s_m"))
                d = fr.get("d_m_filtered", fr.get("d_m"))
            else:
                s, d = fr.get("s_m"), fr.get("d_m")
            if s is None or d is None:
                continue
            sd_list.append((float(s), float(d)))
    return sd_list


def run_frenet_2d_simulation(show_animation=True, custom_obstacles=None, course="classical"):
    """đư
    custom_obstacles: optional np.ndarray (N, 2) — override default obstacles
    course: "classical" (default, original demo), "rl" (match RL_frenet course),
            or "straight" (reference thẳng khớp perception thật: s thẳng, d/heading từ depth)
    """
    if course == "rl":
        # Import RL course for fair comparison
        try:
            from RL_frenet import build_demo_course as build_rl_course
            tx, ty, tyaw, _, csp = build_rl_course()
        except ImportError:
            tx, ty, tyaw, _, csp = build_demo_course()
    elif course == "straight":
        tx, ty, tyaw, _, csp = build_straight_course()
    else:
        tx, ty, tyaw, _, csp = build_demo_course()
    ob = custom_obstacles if custom_obstacles is not None else build_demo_obstacles()
    start_x, start_y = frenet_point_to_global(csp, 0.0, DEMO_START_D)
    state = SimulationState(start_x, start_y, tyaw[0], DEMO_START_SPEED)

    history_x = [state.x]
    history_y = [state.y]
    plt = None

    if show_animation:
        plt = import_matplotlib_pyplot(prefer_interactive=True)
        fig, ax = plt.subplots(figsize=(11, 6))
    else:
        fig, ax = None, None

    last_best_path = None
    for step in range(int(SIM_TIME / DT)):
        s0, c_speed, c_d, c_d_d, c_d_dd = state_to_frenet(state, csp)
        best_path, candidate_paths = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, prev_path=last_best_path
        )

        if best_path is None:
            print("No feasible Frenet path found. Stop simulation.")
            break

        steer, lookahead_point = pure_pursuit_control_2d(state, best_path)
        accel = proportional_speed_control(TARGET_SPEED, state.v)
        state.update(accel, steer)
        last_best_path = best_path

        history_x.append(state.x)
        history_y.append(state.y)

        if show_animation:
            ax.cla()
            ax.set_facecolor("white")
            draw_lane_markings(ax, tx, ty, tyaw)
            ax.plot(tx, ty, "--", color="black", linewidth=1.1, alpha=0.55, label="reference")

            for fp in candidate_paths:
                ax.plot(fp.x, fp.y, color="0.82", linewidth=0.9)

            ax.plot(best_path.x, best_path.y, color="black", linewidth=2.5, label="best path")
            ax.plot(history_x, history_y, color="black", linewidth=2, linestyle=":", label="trajectory")
            ax.plot(lookahead_point[0], lookahead_point[1], "kx", markersize=10, label="lookahead")
            draw_obstacles(ax, ob)
            ax.plot([], [], color="black", linewidth=8, label="obstacle")

            draw_vehicle(ax, state)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(state.x - 15.0, state.x + 35.0)
            ax.set_ylim(min(ty) - 8.0, max(ty) + 8.0)
            ax.grid(True)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(
                f"Frenet 2D simulation | step={step} | speed={state.v * 3.6:.1f} km/h"
            )
            ax.legend(loc="upper left")
            plt.pause(0.001)

        goal_distance = math.hypot(state.x - tx[-1], state.y - ty[-1])
        if s0 >= csp.s[-1] - 1.0 or goal_distance <= GOAL_TOLERANCE:
            print("Goal reached.")
            break

    if show_animation:
        if last_best_path is not None:
            ax.plot(last_best_path.x, last_best_path.y, color="black", linewidth=2.5)
        plt.ioff()
        plt.show()

    return history_x, history_y, last_best_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Classical Frenet 2D simulation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed cho obstacle generation (giống RL_frenet). Mặc định: dùng obstacle cố định.")
    parser.add_argument("--obstacle-count", type=int, default=5)
    parser.add_argument("--obstacle-pattern", type=str, default="standard",
                        choices=["standard", "cluster", "opposite", "same_lane", "dense", "zigzag_cluster"])
    parser.add_argument("--obstacle-min-gap", type=float, default=8.0)
    parser.add_argument("--obstacle-lateral-jitter", type=float, default=1.0)
    parser.add_argument("--obstacle-start-s", type=float, default=15.0)
    parser.add_argument("--obstacles-json", type=str, default=None,
                        help="Đường dẫn JSON 'detections' từ perception (rgb_dual_model_node) để lấy obstacle (s,d) thật.")
    parser.add_argument("--obstacles-sd", type=str, default=None,
                        help='Nhập tay obstacle dạng Frenet, vd: "30,1.0 55,-1.2 80,0.5" (s,d cách nhau bởi dấu phẩy, mỗi obstacle cách nhau bởi space).')
    parser.add_argument("--obstacles-s-offset", type=float, default=0.0,
                        help="Cộng vào s của obstacle thật (mặc định 0: đặt ngay phía trước vạch xuất phát).")
    parser.add_argument("--obstacles-flip-d", action="store_true",
                        help="Đảo dấu d nếu quy ước perception ngược với sim (sim: +d = trái).")
    parser.add_argument("--course-length", type=float, default=200.0,
                        help="Chiều dài reference thẳng [m] khi dùng obstacle thật (khớp perception: s thẳng).")
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    custom_obs = None
    course = "classical"

    # Đưa obstacle THẬT (Frenet s,d) vào sim: ưu tiên hơn obstacle giả/seed.
    frenet_sd = None
    if args.obstacles_json is not None:
        frenet_sd = load_perception_obstacles(args.obstacles_json)
        print(f"Loaded {len(frenet_sd)} obstacle(s) từ {args.obstacles_json}")
    elif args.obstacles_sd is not None:
        frenet_sd = [tuple(float(v) for v in pair.split(",")) for pair in args.obstacles_sd.split()]
        print(f"Parsed {len(frenet_sd)} obstacle(s) từ --obstacles-sd")

    if frenet_sd is not None:
        # Môi trường khớp perception: reference THẲNG (s = đoạn thẳng, d/heading từ depth).
        # Dùng đúng csp mà sim sẽ chạy để chuyển (s,d) -> (x,y).
        _, _, _, _, csp_sim = build_straight_course(length=args.course_length)
        custom_obs = obstacles_from_frenet(
            csp_sim, frenet_sd,
            s_offset=args.obstacles_s_offset,
            flip_d=args.obstacles_flip_d,
        )
        run_frenet_2d_simulation(show_animation=not args.no_animation,
                                 custom_obstacles=custom_obs, course="straight")
        return

    if args.seed is not None:
        # Dùng RL_frenet obstacle generator để so sánh fair
        try:
            from RL_frenet import build_rollout_obstacles, build_demo_course as build_rl_course
            _, _, _, _, csp_rl = build_rl_course()
            custom_obs = build_rollout_obstacles(
                csp_rl,
                obstacle_count=args.obstacle_count,
                seed=args.seed,
                obstacle_start_s=args.obstacle_start_s,
                obstacle_lateral_jitter=args.obstacle_lateral_jitter,
                obstacle_min_gap=args.obstacle_min_gap,
                pattern=args.obstacle_pattern,
            )
            course = "rl"
            print(f"Using RL course + seed={args.seed}, pattern={args.obstacle_pattern}, count={args.obstacle_count}")
            print(f"Generated {len(custom_obs)} obstacles")
        except ImportError as e:
            print(f"Warning: Cannot import RL_frenet ({e}). Falling back to classical demo.")
    run_frenet_2d_simulation(show_animation=not args.no_animation, custom_obstacles=custom_obs, course=course)


if __name__ == "__main__":
    main()

