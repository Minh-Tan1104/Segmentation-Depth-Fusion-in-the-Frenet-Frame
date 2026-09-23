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
    # 1: obs 4 chiều [d, psi, ds, d_obs] — "không obstacle" mã hoá (ds=tầm
    #    nhìn, d_obs=0), TRÙNG với obstacle giữa làn vừa vào tầm nhìn.
    # 2: thêm cờ has_obstacle (obs 5 chiều) để tách 2 trường hợp đó.
    # Meta cũ không có field -> 1, model cũ vẫn chạy đúng.
    obs_version: int = 1
    # True: policy train trong KHUNG GƯƠNG (SideLatch) — vật cản luôn ở phía
    # trái-hoặc-thẳng xe trong input, action lật lại khi xuất. Meta cũ -> False.
    canonical: bool = False
    side_switch_m: float = 0.3

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
    high = [meta.d_max * 1.25, math.pi, meta.d_max * 1.1, meta.d_max * 1.1]
    if meta.obs_version >= 2:
        high.append(1.0)
    return np.array(high, dtype=np.float32)


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
    """[d, psi, ds_obs scale về ~[0, d_max], d_obs(, has_obstacle nếu
    obs_version>=2)]; không có obstacle trong tầm nhìn -> (ds=vision_range_m,
    d_obs=0, has_obstacle=0), đúng mặc định lúc train."""
    ds_obs, d_obs = obstacle if obstacle is not None else (meta.vision_range_m, 0.0)
    feats = [d, psi, ds_obs / meta.vision_range_m * meta.d_max, d_obs]
    if meta.obs_version >= 2:
        feats.append(0.0 if obstacle is None else 1.0)
    obs = np.array(feats, dtype=np.float32)
    high = obs_high(meta)
    return np.clip(obs, -high, high)


def decode_action(action: np.ndarray, meta: RLPolicyMeta) -> tuple[float, float]:
    """action∈[-1,1]^2 -> (d_target TUYỆT ĐỐI, Ti)."""
    d_target = float(
        np.clip(meta.center_offset + float(action[0]) * meta.d_max, meta.d_min, meta.d_max)
    )
    Ti = meta.min_t + (float(action[1]) + 1.0) / 2.0 * (meta.max_t - meta.min_t)
    return d_target, max(Ti, 1e-2)


class SideLatch:
    """Chọn + CHỐT phía né cho khung gương (side=+1 giữ nguyên, -1 lật gương
    d/psi/d_obs và action[0]).

    Vì sao cần: policy là hàm LIÊN TỤC của d_obs, nên chuyển từ "né phải"
    sang "né trái" bắt buộc đi qua 1 dải d_target≈0 quanh d_obs≈d — đúng dải
    đâm thẳng vào vật cản giữa làn (đo: model train thường va ở d_obs 0..+0.05
    dù train thêm/oversample chỉ làm dải hẹp lại hoặc dời chỗ). Trong khung
    gương vật cản luôn ở trái-hoặc-thẳng xe -> policy chỉ học 1 phía, việc chọn
    phía là 1 phép so dấu (công tắc cứng), không còn dải nội suy.

    Chốt: giữ side cho tới khi hết vật cản trong tầm nhìn / đổi vật cản mới;
    chỉ đổi phía khi vật cản nằm rõ ràng ở phía kia quá switch_m — nhiễu
    perception quanh d_obs≈d không làm xe đổi phía giữa chừng."""

    def __init__(self, switch_m: float) -> None:
        self.switch_m = switch_m
        self.side: int | None = None
        self._last_ds: float | None = None

    def reset(self) -> None:
        self.side = None
        self._last_ds = None

    def update(self, d: float, obstacle: tuple[float, float] | None) -> int:
        if obstacle is None:
            self.reset()
            return 1
        ds, d_obs = obstacle
        if self._last_ds is not None and ds > self._last_ds + 1.0:
            self.side = None  # vật cản gần nhất đã đổi sang vật cản khác
        self._last_ds = ds
        rel = d_obs - d
        if self.side is None:
            # vật cản bên phải (hoặc thẳng mà xe đang ở nửa trái) -> lật gương
            self.side = -1 if rel > 0.0 or (rel == 0.0 and d < 0.0) else 1
        elif self.side * rel > self.switch_m:
            self.side = -self.side
        return self.side


def canonical_observation(
    d: float, psi: float, obstacle: tuple[float, float] | None, side: int, meta: RLPolicyMeta
) -> np.ndarray:
    if obstacle is not None:
        obstacle = (obstacle[0], side * obstacle[1])
    return build_observation(side * d, side * psi, obstacle, meta)


class RLPolicy:
    def __init__(self, model_path: str) -> None:
        from stable_baselines3 import SAC

        path = os.path.expanduser(model_path)
        self.meta = RLPolicyMeta.load(meta_path(path))
        self.model = SAC.load(path, device="cpu")
        self.latch = SideLatch(self.meta.side_switch_m)

    def action(
        self, d: float, psi: float, obstacles: list[tuple[float, float]], latch: bool = True
    ) -> np.ndarray:
        """Action thô ∈[-1,1]^2 ở khung THẬT. latch=False: truy vấn phụ (vd
        hỏi "nếu không có vật cản"), không đụng trạng thái SideLatch."""
        obstacle = nearest_obstacle_ahead(obstacles, self.meta.vision_range_m)
        if not self.meta.canonical:
            obs = build_observation(d, psi, obstacle, self.meta)
            return self.model.predict(obs, deterministic=True)[0]
        if latch:
            side = self.latch.update(d, obstacle)
        else:
            side = 1 if obstacle is None else (self.latch.side or 1)
        obs = canonical_observation(d, psi, obstacle, side, self.meta)
        action = self.model.predict(obs, deterministic=True)[0].copy()
        action[0] *= side  # center_offset=0 (assert lúc train) -> lật quanh tâm làn
        return action

    def select(
        self, d: float, psi: float, obstacles: list[tuple[float, float]], latch: bool = True
    ) -> tuple[float, float]:
        return decode_action(self.action(d, psi, obstacles, latch), self.meta)
