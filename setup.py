from glob import glob

from setuptools import setup

package_name = "RL_CAR"

setup(
    name=package_name,
    version="0.1.0",
    packages=["perception", "planner_motion", "visualization", "Encoder", "control", "Gps"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/map", glob("map/*.csv")),
    ],
    install_requires=["setuptools", "scipy", "pyserial", "pyubx2", "pyproj"],
    zip_safe=True,
    maintainer="minh_tan",
    maintainer_email="khoa16042003@gmail.com",
    description="Perception (seg làn + Frenet + detection) -> planner (Frenet Optimal) -> visualization cho xe tự hành.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = perception.node:main",
            "planner_motion_node = planner_motion.node:main",
            "visualization_node = visualization.node:main",
            "encoder_node = Encoder.node:main",
            "control_node = control.node:main",
            "joy_pygame_node = control.joy_pygame_node:main",
            "gps_node = Gps.node:main",
        ],
    },
)
