import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    realsense_launch = os.path.join(
        get_package_share_directory("RL_CAR"), "launch", "realsense.launch.py"
    )
    # Tất cả tham số của perception_node/planner_motion_node/visualization_node
    # (topic, model, ngưỡng detect, FOT planner...) nằm trong đây. Muốn chỉnh
    # thì sửa file YAML này, không cần đụng launch file.
    params_file = os.path.join(
        get_package_share_directory("RL_CAR"), "config", "rl_car_params.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                # Tắt khi chạy lại từ rosbag (không có camera thật gắn vào).
                "launch_realsense",
                default_value="true",
            ),
            # ── Camera RealSense: publish RGB + depth (aligned) cho perception_node ──
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch),
                condition=IfCondition(LaunchConfiguration("launch_realsense")),
            ),
            # ── Perception: camera RGB+Depth -> seg làn -> Frenet -> detection vật cản ──
            # Publish visual_frame (ảnh đã blend segmentation) + frenet_state
            # (JSON detections/frenet, chưa có planner) cùng header.stamp.
            Node(
                package="RL_CAR",
                executable="perception_node",
                name="perception_node",
                output="screen",
                parameters=[params_file],
            ),
            # ── Planner: Frenet Optimal Planner chạy process riêng, không chặn YOLO ──
            # Nhận frenet_state từ perception, gắn thêm optimal_path/candidate_paths
            # rồi publish ra đúng visual_frenet_topic mà visualization_node đang đợi.
            Node(
                package="RL_CAR",
                executable="planner_motion_node",
                name="planner_motion_node",
                output="screen",
                parameters=[params_file],
            ),
            # ── Visualization: vẽ overlay từ visual_frame + visual_frenet ──
            Node(
                package="RL_CAR",
                executable="visualization_node",
                name="visualization_node",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
