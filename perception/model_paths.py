from __future__ import annotations

import os

# Model không được colcon copy vào install/ (data_files không khai báo) nên
# không thể dùng đường dẫn tính theo __file__ (sau install, __file__ nằm
# trong install/RL_CAR/lib/..., không phải source tree). Trỏ thẳng tới
# source tree cho chắc chắn — override bằng biến môi trường RL_CAR_SOURCE_ROOT
# nếu workspace nằm ở đường dẫn khác (máy khác, user khác).
_SOURCE_ROOT = os.environ.get(
    "RL_CAR_SOURCE_ROOT",
    "/home/rl/ros2_ws/src/RL_CAR_Cotrol_perception_visualation/RL_CAR",
)


def _first_existing(*candidates: str) -> str:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0]


def default_yolo_model_path() -> str:
    primary = os.path.join(_SOURCE_ROOT, "model", "detection", "yolo11n.pt")
    fallback = os.path.join(_SOURCE_ROOT, "yolo11n.pt")
    return _first_existing(primary, fallback)


def default_seg_model_path() -> str:
    primary = os.path.join(_SOURCE_ROOT, "model", "seg", "best.pt")
    fallback = os.path.join(_SOURCE_ROOT, "yolov8n-seg.pt")
    return _first_existing(primary, fallback)
