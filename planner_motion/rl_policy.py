"""Policy RL (SAC, train bằng train_frenet_rl.py) thay phần CHỌN (d_target, Ti)
của Frenet planner straight mode — bật bằng plan_use_rl (control_node).

Thuần numpy: stable_baselines3/torch chỉ import khi thật sự load model
(RLPolicy), nên control_node không cần torch khi plan_use_rl=false.

Observation/action PHẢI mã hoá y hệt lúc train: build_observation và
decode_action dùng chung cho env train (train_frenet_rl.py) lẫn lúc chạy
thật, với hằng số đọc từ file meta lưu cạnh model (RLPolicyMeta) — KHÔNG lấy
từ plan_* runtime, vì policy học ngữ nghĩa action theo đúng khoảng lúc train
(config robot có thể khác, vd plan_min/max_horizon_s).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RLPolicyMeta:
    center_offset: float
    d_min: float
    d_max: float
    min_t: float
    max_t: float
    vision_range_m: float

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RLPolicyMeta":
        with open(path) as f:
            return cls(**json.load(f))


def meta_path(model_path: str) -> str:
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    return base + ".meta.json"


def obs_high(meta: RLPolicyMeta) -> np.ndarray:
    return np.array(
        [meta.d_max * 1.25, math.pi, meta.d_max * 1.1, meta.d_max * 1.1],
        dtype=np.float32,
    )


def nearest_obstacle_ahead(
    obstacles: list[tuple[float, float]], vision_range_m: float
) -> tuple[float, float] | None:
    """(ds, d) của obstacle gần nhất PHÍA TRƯỚC trong tầm nhìn, None nếu không
    có. obstacles: (ds, d) — ds tương đối vị trí xe (s0=0), d theo khung làn."""
    best = None
    for ds, d in obstacles:
        if 0.0 <= ds <= vision_range_m and (best is None or ds < best[0]):
            best = (ds, d)
    return best


def build_observation(
    d: float, psi: float, obstacle: tuple[float, float] | None, meta: RLPolicyMeta
) -> np.ndarray:
    """[d, psi, ds_obs scale về ~[0, d_max], d_obs]; không có obstacle trong
    tầm nhìn -> (ds=vision_range_m, d_obs=0), đúng mặc định lúc train."""
    ds_obs, d_obs = obstacle if obstacle is not None else (meta.vision_range_m, 0.0)
    obs = np.array(
        [d, psi, ds_obs / meta.vision_range_m * meta.d_max, d_obs], dtype=np.float32
    )
    high = obs_high(meta)
    return np.clip(obs, -high, high)


def decode_action(action: np.ndarray, meta: RLPolicyMeta) -> tuple[float, float]:
    """action∈[-1,1]^2 -> (d_target TUYỆT ĐỐI, Ti)."""
    d_target = float(
        np.clip(meta.center_offset + float(action[0]) * meta.d_max, meta.d_min, meta.d_max)
    )
    Ti = meta.min_t + (float(action[1]) + 1.0) / 2.0 * (meta.max_t - meta.min_t)
    return d_target, max(Ti, 1e-2)


class RLPolicy:
    def __init__(self, model_path: str) -> None:
        from stable_baselines3 import SAC

        path = os.path.expanduser(model_path)
        self.meta = RLPolicyMeta.load(meta_path(path))
        self.model = SAC.load(path, device="cpu")

    def select(
        self, d: float, psi: float, obstacles: list[tuple[float, float]]
    ) -> tuple[float, float]:
        obstacle = nearest_obstacle_ahead(obstacles, self.meta.vision_range_m)
        obs = build_observation(d, psi, obstacle, self.meta)
        action, _ = self.model.predict(obs, deterministic=True)
        return decode_action(action, self.meta)
