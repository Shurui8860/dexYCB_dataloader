from __future__ import annotations
import numpy as np
from typing import Dict, List, Sequence, Tuple, Iterable
import torch


class JointConvention:
    """A simple joint indexing convention: finger -> ordered joint indices."""
    def __init__(self, name: str, layout: Dict[str, Sequence[int]]):
        self.name = name
        self.layout = {k: list(v) for k, v in layout.items()}
        # Precompute semantic <-> index maps
        self._idx_to_sem: Dict[int, Tuple[str, int]] = {
            idx: (finger, k) for finger, ids in self.layout.items() for k, idx in enumerate(ids)
        }
        self._sem_to_idx: Dict[Tuple[str, int], int] = {
            (finger, k): idx for finger, ids in self.layout.items() for k, idx in enumerate(ids)
        }
        self.size = max(self._idx_to_sem) + 1

    def idx_to_sem(self) -> Dict[int, Tuple[str, int]]:
        return self._idx_to_sem

    def sem_to_idx(self) -> Dict[Tuple[str, int], int]:
        return self._sem_to_idx

    @property
    def get_name(self):
        return self.name

    @property
    def get_layout(self):
        return self.layout


class JointReindexer:
    """Reindex joints from `src` convention to `dst` via a permutation."""
    def __init__(self, src: JointConvention, dst: JointConvention):
        self.src = src
        self.dst = dst
        N = dst.size
        self.perm = np.fromiter(
            (src.sem_to_idx()[dst.idx_to_sem()[i]] for i in range(N)),
            dtype=int, count=N
        )

    def apply(self, joints: np.ndarray) -> np.ndarray:
        """Reorder joints: joints shape (..., N, D)."""
        if joints.shape[-2] != self.perm.size:
            raise ValueError(f"Expected joints.shape[-2]=={self.perm.size}, got {joints.shape[-2]}")
        return joints[..., self.perm, :]

    def inverse(self) -> "JointReindexer":
        """Return the inverse mapper (dst -> src)."""
        inv = JointReindexer.__new__(JointReindexer)
        inv.src, inv.dst = self.dst, self.src
        inv.perm = np.argsort(self.perm)
        return inv

    def __repr__(self):
        return f"JointReindexer({self.src.name} -> {self.dst.name}, N={self.perm.size})"


# ---- Conventions ----
MANO21 = JointConvention(
    "MANO21/OpenPose",
    {
        "wrist":  [0],
        "thumb":  [1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    },
)

HO3D = JointConvention(
    "HO3D",
    {
        "wrist":  [0],
        "index":  [1, 2, 3, 17],
        "middle": [4, 5, 6, 18],
        "ring":   [10, 11, 12, 19],
        "pinky":  [7, 8, 9, 20],
        "thumb":  [13, 14, 15, 16],
    },
)

# ---- Ready-to-use mappers ----
MANO_TO_HO3D = JointReindexer(MANO21, HO3D)
HO3D_TO_MANO = MANO_TO_HO3D.inverse()

# Optional convenience wrappers
def mano_to_ho3d(x: np.ndarray) -> np.ndarray:
    return MANO_TO_HO3D.apply(x)

def ho3d_to_mano(x: np.ndarray) -> np.ndarray:
    return HO3D_TO_MANO.apply(x)


class YCBRegistry:
    """Minimal OOP wrapper around the YCB id<->name mapping."""
    __slots__ = ("_id_to_name", "_name_to_id")

    def __init__(self, id_to_name: Dict[int, str]):
        self._id_to_name: Dict[int, str] = {int(k): str(v) for k, v in id_to_name.items()}
        self._name_to_id: Dict[str, int] = {v: k for k, v in self._id_to_name.items()}

    # --- Lookups ---
    def id_to_name(self, ycb_id: int) -> str:
        try:
            return self._id_to_name[int(ycb_id)]
        except (KeyError, ValueError):
            raise ValueError(f"Unknown YCB id: {ycb_id}")

    def name_to_id(self, name: str) -> int:
        try:
            return self._name_to_id[name]
        except KeyError:
            raise ValueError(f"Unknown YCB name: {name}")

    # --- Introspection ---
    def ids(self) -> Iterable[int]:
        return self._id_to_name.keys()

    def names(self) -> Iterable[str]:
        return self._id_to_name.values()

    def items(self) -> Iterable[tuple[int, str]]:
        return self._id_to_name.items()

    def __contains__(self, key) -> bool:
        if isinstance(key, int):
            return key in self._id_to_name
        if isinstance(key, str):
            return key in self._name_to_id
        return False

    def __len__(self) -> int:
        return len(self._id_to_name)

    def __repr__(self) -> str:
        return f"YCBRegistry(n={len(self)})"


# ---- Default registry ----
YCB = YCBRegistry(
    {
        1: "002_master_chef_can",
        2: "003_cracker_box",
        3: "004_sugar_box",
        4: "005_tomato_soup_can",
        5: "006_mustard_bottle",
        6: "007_tuna_fish_can",
        7: "008_pudding_box",
        8: "009_gelatin_box",
        9: "010_potted_meat_can",
        10: "011_banana",
        11: "019_pitcher_base",
        12: "021_bleach_cleanser",
        13: "024_bowl",
        14: "025_mug",
        15: "035_power_drill",
        16: "036_wood_block",
        17: "037_scissors",
        18: "040_large_marker",
        19: "051_large_clamp",
        20: "052_extra_large_clamp",
        21: "061_foam_brick",
    }
)

# --- Optional functional shims (keep old API working) ---
def ycb_id_to_name(ycb_id: int) -> str:
    return YCB.id_to_name(ycb_id)

def ycb_name_to_id(name: str) -> int:
    return YCB.name_to_id(name)


def quaternionToAxisAngle(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized quaternion -> (axis, angle).

    Parameters
    P : (..., 4) array-like
        Quaternions in (e0, e1, e2, e3) with scalar first.

    Returns
    axes : (..., 3) np.ndarray
    angles : (...) np.ndarray
    """
    P = np.asarray(P, dtype=float)

    assert P.shape[-1] == 4, "last dim must be 4"
    e0 = P[..., 0]
    e = P[..., 1:4]
    n = np.linalg.norm(e, axis=-1)

    axes = np.zeros(P.shape[:-1] + (3,), dtype=float)
    nz = n != 0
    axes[nz] = e[nz] / n[nz, None]
    axes[~nz] = np.array([1.0, 0.0, 0.0])

    angles = np.zeros_like(n)
    w_zero = e0 == 0
    angles[w_zero] = np.pi
    mask = ~w_zero
    angles[mask] = 2.0 * np.arctan(n[mask] / e0[mask])

    return axes, angles


def axisAngleToRotvec(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Convert (axis, angle) -> rotation vector r = axis * angle.
    axes:  (..., 3)
    angles: (...)   (radians)
    returns: (..., 3) with ||r|| = angle
    """
    axes = np.asarray(axes, dtype=float)
    angles = np.asarray(angles, dtype=float)
    return axes * angles[..., None]


# ---- Optional tiny check ----
if __name__ == "__main__":
    N, D = 21, 3
    x = np.arange(N * D).reshape(1, N, D)
    y = MANO_TO_HO3D.apply(x)
    x_back = HO3D_TO_MANO.apply(y)
    print(MANO_TO_HO3D)
    print("perm:", MANO_TO_HO3D.perm.tolist())
    print("roundtrip ok:", np.array_equal(x, x_back))
