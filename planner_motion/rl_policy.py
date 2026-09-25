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
    # 3: tối đa max_obstacles vật gần nhất, mỗi vật (ds, d_obs, present):
    #    [d, psi] + max_obstacles*3 — cần khi 2-3 vật cùng trong tầm nhìn.
    # Meta cũ không có field -> 1, model cũ vẫn chạy đúng.
    obs_version: int = 1
    max_obstacles: int = 1
    # True: policy train trong KHUNG GƯƠNG (SideLatch) — vật cản luôn ở phía
    # trái-hoặc-thẳng xe trong input, action lật lại khi xuất. Meta cũ -> False.
    canonical: bool = False
    side_switch_m: float = 0.3
    # d_target = center + d_max * sign(a)*|a|^action_power. 1 = tuyến tính
    # (model cũ). 2: gần tâm mịn hơn — sai số 0.025 của mạng chỉ còn ~1.5 mm
    # thay vì 6 cm (đo: model tuyến tính đứng lệch +0.06 m khi đường trống vì
    # chi phí 6 cm quá nhỏ so với nhiễu critic), vẫn tới được ±d_max.
    action_power: float = 1.0

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
    if meta.obs_version >= 3:
        high = [meta.d_max * 1.25, math.pi] + [meta.d_max * 1.1, meta.d_max * 1.1, 1.0] * meta.max_obstacles
        return np.array(high, dtype=np.float32)
    high = [meta.d_max * 1.25, math.pi, meta.d_max * 1.1, meta.d_max * 1.1]
    if meta.obs_version >= 2:
        high.append(1.0)
    return np.array(high, dtype=np.float32)


def obstacles_ahead(
    obstacles: list[tuple[float, float]], vision_range_m: float, k: int
) -> list[tuple[float, float]]:
    """k vật cản (ds, d) gần nhất PHÍA TRƯỚC trong tầm nhìn, gần nhất trước."""
    ahead = sorted(o for o in obstacles if 0.0 <= o[0] <= vision_range_m)
    return ahead[:k]


def build_observation(
    d: float, psi: float, obstacles: list[tuple[float, float]], meta: RLPolicyMeta
) -> np.ndarray:
    """obstacles: vật cản trong tầm nhìn (ds, d), GẦN NHẤT TRƯỚC (xem
    obstacles_ahead). obs_version 1/2: chỉ dùng vật đầu — [d, psi, ds scale
    về ~[0, d_max], d_obs(, has_obstacle)]. obs_version 3: [d, psi] + mỗi
    slot (ds, d_obs, present) cho max_obstacles vật. Slot trống / không có
    vật: (ds=vision_range_m, d_obs=0, present=0), đúng mặc định lúc train."""
    scale = meta.d_max / meta.vision_range_m
    if meta.obs_version >= 3:
        feats = [d, psi]
        for i in range(meta.max_obstacles):
            if i < len(obstacles):
                feats += [obstacles[i][0] * scale, obstacles[i][1], 1.0]
            else:
                feats += [meta.d_max, 0.0, 0.0]
        obs = np.array(feats, dtype=np.float32)
        high = obs_high(meta)
        return np.clip(obs, -high, high)
    obstacle = obstacles[0] if obstacles else None
    ds_obs, d_obs = obstacle if obstacle is not None else (meta.vision_range_m, 0.0)
    feats = [d, psi, ds_obs * scale, d_obs]
    if meta.obs_version >= 2:
        feats.append(0.0 if obstacle is None else 1.0)
    obs = np.array(feats, dtype=np.float32)
    high = obs_high(meta)
    return np.clip(obs, -high, high)


# Gate cho đi thẳng nếu đường bám làn cách mọi vật >= robot_radius + margin
# (không phải plan_clearance): Frenet cũng đi qua khe giữa 2 vật khi cách ~1 m
# (chỉ bị phạt mềm), còn gate 1.2 m đẩy các khe đó sang policy và policy né
# quá tay vào vật phía sau (đo: 2/93 kịch bản Frenet qua mà RL không qua ->
# 0/93 với margin 0.3 m). Cùng biên với ràng buộc khả thi R4 lúc train.
GATE_MARGIN_M = 0.3


def gate_distance(robot_radius: float) -> float:
    return robot_radius + GATE_MARGIN_M


def lane_keeping_is_clear(path_s: np.ndarray, path_d: np.ndarray,
                          obstacles: list[tuple[float, float]], min_dist: float,
                          horizon_m: float) -> bool:
    """Gate "chỉ đưa vật cản cho policy khi cần": đường bám làn (path_s,
    path_d; s tương đối xe) KÉO DÀI tới horizon_m, giữ nguyên d cuối, có
    cách mọi vật cản (ds, d) ít nhất min_dist (xem gate_distance) không.

    Kéo dài vì path chỉ dài Ti*v (robot: 2-3 m) trong khi vật cản thấy tới
    tầm nhìn (8 m): so với path ngắn, vật ở ds 3-6 m luôn "đủ xa" -> gate cho
    đi thẳng tới khi quá muộn để né (đo: 9/20 thất bại cụm vật cản do đây)."""
    s_ext = np.append(path_s, max(horizon_m, float(path_s[-1])))
    d_ext = np.append(path_d, path_d[-1])
    for ds, d_obs in obstacles:
        if abs(d_obs - float(np.interp(ds, s_ext, d_ext))) < min_dist:
            return False
        if float(np.min(np.hypot(path_s - ds, path_d - d_obs))) < min_dist:
            return False
    return True


def decode_action(action: np.ndarray, meta: RLPolicyMeta) -> tuple[float, float]:
    """action∈[-1,1]^2 -> (d_target TUYỆT ĐỐI, Ti)."""
    a0 = float(action[0])
    shaped = math.copysign(abs(a0) ** meta.action_power, a0)
    d_target = float(
        np.clip(meta.center_offset + shaped * meta.d_max, meta.d_min, meta.d_max)
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
    d: float, psi: float, obstacles: list[tuple[float, float]], side: int, meta: RLPolicyMeta
) -> np.ndarray:
    """Lật gương d, psi và d_obs của MỌI vật theo side (chốt theo vật gần nhất)."""
    mirrored = [(ds, side * d_obs) for ds, d_obs in obstacles]
    return build_observation(side * d, side * psi, mirrored, meta)


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
        visible = obstacles_ahead(obstacles, self.meta.vision_range_m, self.meta.max_obstacles)
        nearest = visible[0] if visible else None
        if not self.meta.canonical:
            obs = build_observation(d, psi, visible, self.meta)
            return self.model.predict(obs, deterministic=True)[0]
        if latch:
            side = self.latch.update(d, nearest)
        else:
            side = 1 if nearest is None else (self.latch.side or 1)
        obs = canonical_observation(d, psi, visible, side, self.meta)
        action = self.model.predict(obs, deterministic=True)[0].copy()
        action[0] *= side  # center_offset=0 (assert lúc train) -> lật quanh tâm làn
        return action

    def select(
        self, d: float, psi: float, obstacles: list[tuple[float, float]], latch: bool = True
    ) -> tuple[float, float]:
        return decode_action(self.action(d, psi, obstacles, latch), self.meta)
