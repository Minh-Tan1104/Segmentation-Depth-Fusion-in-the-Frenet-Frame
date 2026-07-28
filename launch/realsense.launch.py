import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description() -> LaunchDescription:
    rs_launch = os.path.join(
        get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"
    )

    return LaunchDescription(
        [
            # Bật align_depth để topic aligned_depth_to_color khớp với
            # depth_topic mặc định mà perception_node đang subscribe.
            # camera_name/camera_namespace mặc định "camera" -> topic
            # /camera/camera/color/image_raw, đúng với rgb_topic mặc định.
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch),
                launch_arguments={
                    "align_depth.enable": "true",
                    "enable_sync": "true",
                }.items(),
            ),
        ]
    )
