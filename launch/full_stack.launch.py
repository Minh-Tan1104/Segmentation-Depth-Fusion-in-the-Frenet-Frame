import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    rgb_dual_model_launch = os.path.join(
        get_package_share_directory("RL_CAR"), "launch", "rgb_dual_model.launch.py"
    )
    # Tất cả tham số của joy_pygame_node/encoder_node/control_node/gps_node
    # (serial, scale động cơ, EKF, FOT planner nội bộ, RouteEKF...) nằm trong
    # đây. Muốn chỉnh thì sửa file YAML này, không cần đụng launch file.
    params_file = os.path.join(
        get_package_share_directory("RL_CAR"), "config", "rl_car_params.yaml"
    )

    return LaunchDescription(
        [
            # Tắt khi không có module GPS gắn vào (test trong nhà, rosbag...)
            # — control_node vẫn chạy bình thường vì gps_assist_enable mặc
            # định false, không phụ thuộc node này có chạy hay không.
            DeclareLaunchArgument("launch_gps", default_value="false"),
            # ── Stack camera: perception -> planner_motion -> visualization ──
            # (giữ nguyên rgb_dual_model.launch.py để vẫn chạy độc lập được
            # khi không có serial/tay cầm gắn vào, ví dụ test với rosbag)
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rgb_dual_model_launch)
            ),
            # ── Tay cầm PS4 DualShock qua Bluetooth (đọc bằng pygame, ổn định
            # hơn joy_node/evdev trên tay cầm "Wireless Controller" này) ──
            Node(
                package="RL_CAR",
                executable="joy_pygame_node",
                name="joy_pygame_node",
                output="screen",
                parameters=[params_file],
            ),
            # ── Encoder: sở hữu serial hoverboard, publish /odom, nhận /cmd_vel ──
            Node(
                package="RL_CAR",
                executable="encoder_node",
                name="encoder_node",
                output="screen",
                parameters=[params_file],
            ),
            # ── Control: chuyển mode manual/auto (R1), EKF fusion, pure pursuit ──
            Node(
                package="RL_CAR",
                executable="control_node",
                name="control_node",
                output="screen",
                parameters=[params_file],
            ),
            # ── GPS: dẫn đường (s/d/psi_err theo tuyến CSV), KHÔNG lái xe ──
            # (xem Gps/node.py — chỉ publish /gps/route_state, control_node tự
            # quyết có dùng hay không qua gps_assist_enable)
            Node(
                package="RL_CAR",
                executable="gps_node",
                name="gps_node",
                output="screen",
                parameters=[params_file],
                condition=IfCondition(LaunchConfiguration("launch_gps")),
            ),
        ]
    )
