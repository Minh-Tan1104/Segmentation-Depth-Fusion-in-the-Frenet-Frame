#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Tìm workspace root bằng cách đi lên tới thư mục "src" gần nhất rồi lấy
# parent — không giả định độ sâu lồng cố định, vì package có thể nằm trực
# tiếp ở src/RL_CAR (máy khác) hoặc lồng sâu hơn như ở máy này
# (src/RL_CAR_Cotrol_perception_visualation/RL_CAR).
WS_SRC_DIR="${PKG_ROOT}"
while [[ "$(basename "${WS_SRC_DIR}")" != "src" ]]; do
  WS_SRC_DIR="$(dirname "${WS_SRC_DIR}")"
  if [[ "${WS_SRC_DIR}" == "/" ]]; then
    echo "Không tìm thấy thư mục 'src' tổ tiên của ${PKG_ROOT}" >&2
    exit 1
  fi
done
WS_ROOT="$(dirname "${WS_SRC_DIR}")"
PKG_NAME="RL_CAR"
LAUNCH_FILE="full_stack.launch.py"

LOCAL_BUILD_DIR="${PKG_ROOT}/build"
LOCAL_INSTALL_DIR="${PKG_ROOT}/install"
LOCAL_LOG_DIR="${PKG_ROOT}/log"

WS_BUILD_DIR="${WS_ROOT}/build/${PKG_NAME}"
WS_INSTALL_DIR="${WS_ROOT}/install/${PKG_NAME}"
WS_LOG_DIR="${WS_ROOT}/log"

# use_window/window_scale/seg_min_area/seg_x_min/seg_x_max KHÔNG còn là
# launch arg từ khi gom hết tham số node vào config/rl_car_params.yaml —
# `ros2 launch --show-args` giờ chỉ còn launch_gps/launch_realsense +
# arg riêng của realsense2_camera. Muốn chỉnh mấy cái đó thì sửa thẳng YAML.
DEFAULT_ARGS=(
  "launch_gps:=true"
)

source_setup() {
  local setup_file="$1"
  set +u
  # shellcheck disable=SC1090
  source "${setup_file}"
  set -u
}

if [[ $# -gt 0 ]]; then
  LAUNCH_ARGS=("$@")
else
  LAUNCH_ARGS=("${DEFAULT_ARGS[@]}")
fi

echo "[1/5] Kill leftover node processes từ lần chạy trước (tránh chiếm serial/camera)"
# Dùng pattern có "/" đứng trước (khớp đường dẫn executable thật, vd
# .../lib/RL_CAR/encoder_node) — KHÔNG dùng tên trần (vd "encoder_node")
# vì pkill -f so khớp cả vào cmdline của chính shell đang chạy script này,
# có thể tự kill nhầm tiến trình hiện tại.
for pattern in "lib/${PKG_NAME}/" "lib/realsense2_camera/" "lib/joy/"; do
  pkill -f "${pattern}" 2>/dev/null || true
done
sleep 1

echo "[2/5] Remove stale package-local artifacts"
rm -rf "${LOCAL_BUILD_DIR}" "${LOCAL_INSTALL_DIR}" "${LOCAL_LOG_DIR}"

echo "[3/5] Remove workspace artifacts for ${PKG_NAME}"
rm -rf "${WS_BUILD_DIR}" "${WS_INSTALL_DIR}"
rm -rf "${WS_LOG_DIR}/latest" "${WS_LOG_DIR}/latest_build"

echo "[4/5] Build ${PKG_NAME} from workspace root: ${WS_ROOT}"
source_setup /opt/ros/humble/setup.bash
cd "${WS_ROOT}"
colcon build --packages-select "${PKG_NAME}"

echo "[5/5] Source workspace overlay and launch ${LAUNCH_FILE} (realsense + perception_node + planner_motion_node + visualization_node + joy_pygame_node + encoder_node + control_node)"
source_setup "${WS_ROOT}/install/setup.bash"
echo "ros2 launch ${PKG_NAME} ${LAUNCH_FILE} ${LAUNCH_ARGS[*]}"
# rs_launch.py (realsense2_camera) cảnh báo "not supported" cho MỌI tham số
# của RL_CAR vì nó soi toàn bộ launch context chung — vô hại, chỉ lọc bớt
# noise chứ không ẩn lỗi thật.
ros2 launch "${PKG_NAME}" "${LAUNCH_FILE}" "${LAUNCH_ARGS[@]}" 2>&1 \
  | grep -vE "is not supported\. Supported parameters are:|^\['accel_fps'"
