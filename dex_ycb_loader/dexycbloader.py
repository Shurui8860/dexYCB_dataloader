# ycb_loader.py
from __future__ import annotations
import os
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from loader_utils import ycb_id_to_name, quaternionToAxisAngle, axisAngleToRotvec, mano_to_ho3d
from dex_ycb_toolkit.layers.mano_layer import MANOLayer


class DexYCBLoader:
    """
    Minimal reader for a DexYCB sequence.

    - __init__(sequence_name): only resolves and stores the sequence path.
    - read_meta(): reads meta.yml, stores self.beta,
    - read_pose(): reads pose.npz, stores
    """

    def __init__(self, sequence_name: str, order="mano"):
        # Resolve the sequence directory only (no I/O here)

        assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
        _DEX_YCB_DIR = os.environ['DEX_YCB_DIR']

        self.root = Path(_DEX_YCB_DIR)
        self.seq_dir = (self.root / sequence_name).resolve()
        self.seq_name = sequence_name
        self.order = order

        # Placeholders
        self.handBeta: Optional[np.ndarray] = None

        self.objName = None
        self.handPose = None
        self.handTrans = None
        self.handJoints3D = None
        self.objTrans = None
        self.objRot = None
        self.side = None

        self.grasp_idx: Optional[int] = None
        self.ycb_ids: Optional[int] = None
        self.ycb_names = None
        self.grasp_ind = None

        # Perform reads
        self.read_meta()
        self.read_pose()

    def read_meta(self) -> Dict[str, Any]:
        """
        Load meta.yml, derive MANO beta file paths, read picked object id/name.
        Returns a minimal dict with the parsed fields.
        """
        meta_path = self.seq_dir / "meta.yml"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.yml not found: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        self.set_mano_beta(meta)
        self.set_ycb(meta)
        self.set_side(meta)

        return {
            "grasp_idx": self.grasp_idx,
            "ycb_id": self.ycb_ids,
            "ycb_name": self.objName,
            "side": self.side,
        }


    def set_ycb(self, meta):
        """Set YCB ids/names and the grasped object from meta.yml."""
        # Read candidate YCB ids and the chosen grasp index
        ycb_ids = list(meta.get("ycb_ids", []))
        grasp_idx = int(meta.get("ycb_grasp_ind", -1))

        if not ycb_ids:
            raise ValueError("`ycb_ids` is empty or missing in meta.yml")
        if grasp_idx < 0 or grasp_idx >= len(ycb_ids):
            raise ValueError(f"`ycb_grasp_ind` ({grasp_idx}) out of range for ycb_ids (len={len(ycb_ids)})")

        # Cache all ids, names, and the selected grasped object
        self.ycb_ids = ycb_ids         # index into ycb_ids
        self.grasp_ind = grasp_idx     # actual YCB id at that index
        self.grasp_idx = self.ycb_ids[grasp_idx]
        self.ycb_names = [ycb_id_to_name(id) for id in self.ycb_ids]
        self.objName = ycb_id_to_name(self.grasp_idx)     # picked object name


    def set_mano_beta(self, meta) -> List[np.ndarray]:
        """
        Read MANO beta values from each mano.yml discovered by read_meta().
        Returns a list of np.ndarray with shape (10,) dtype float32.
        """
        # Build MANO beta absolute paths from meta['mano_calib']
        calib_ids = list(meta.get("mano_calib", []))
        mano_beta_paths = [
            (self.root / "calibration" / f"mano_{cid}" / "mano.yml").resolve()
            for cid in calib_ids
        ]
        mano_beta_path = Path(mano_beta_paths[0])
        with mano_beta_path.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        betas = np.asarray(y["betas"], dtype=np.float32).reshape(-1)
        self.handBeta = betas

        return betas

    def set_side(self, meta) -> str:
        """Set hand side ('right' or 'left') from meta.yml and return it."""
        # meta['mano_sides'] is a list like ['right'] or ['left']
        side = list(meta.get("mano_sides", []))
        self.side = side[0]
        self.num_frames = meta.get("num_frames", 1)
        return side

    def read_pose(self):
        """
        Load pose.npz (expects 'pose_m' and 'pose_y').
        Returns a dict with the two arrays.
        """
        pose_path = self.seq_dir / "pose.npz"
        if not pose_path.exists():
            raise FileNotFoundError(f"pose.npz not found: {pose_path}")

        with np.load(pose_path, allow_pickle=True) as z:
            if "pose_m" not in z or "pose_y" not in z:
                raise KeyError(f"'pose_m' or 'pose_y' missing in {pose_path}. Keys={list(z.keys())}")
            pose_m = z["pose_m"].astype(np.float32)  # (T, H, 51)
            pose_y = z["pose_y"].astype(np.float32)  # (T, O, 7)
            self.set_mano(pose_m)
            self.set_joints()
            self.set_ycb_pos(pose_y)
        return {"pose_m": pose_m, "pose_y":pose_y}


    def set_mano(self, pose_m: np.ndarray):
        """
        Given pose_m of shape [T, 1, 51], split into:
          - handPose  (..., 48): MANO pose coefficients (as provided)
          - handTrans (..., 3) : translation
        If an image has no visible hand / no annotation, that frame's 51-dim vector is all zeros.
        """
        p = np.asarray(pose_m, dtype=np.float32)
        p = p[:, 0, :]  # (T, 51)
        self.handPose = np.ascontiguousarray(p[:, :48], dtype=np.float32)  # (T, 48)
        self.handTrans = np.ascontiguousarray(p[:, 48:51], dtype=np.float32)  # (T, 3)

    def set_joints(self, device: str = "cpu"):
        """Run MANO and cache joints for all frames (meters)."""
        # Convert to tensors on the target device
        p = torch.from_numpy(self.handPose).to(device)
        t = torch.from_numpy(self.handTrans).to(device)

        # Build MANO wrapper and evaluate
        layer = MANOLayer(side=self.side, betas=self.handBeta)
        vertices, joints = layer.forward(p, t)

        # Store as numpy for downstream use
        self.handJoints3D = joints.detach().cpu().numpy().astype(np.float64, copy=False)
        if self.order == "ho3d":
            self.handJoints3D = mano_to_ho3d(self.handJoints3D)
        return self.handJoints3D

    def set_ycb_pos(self, pose_y):
        """Extract the grasped object's rotation and translation from pose_y."""
        # Choose the grasped object index determined in set_ycb()
        gi = self.grasp_ind
        if gi is None:
            raise AttributeError("grasp_ind (or grasp_idx) is not set on this instance.")
        # Split pose_y -> rotation/translation (per-frame) for that object
        dict = self.split_pose_y(pose_y, gi)
        self.objRot = dict["objRot"]
        self.objTrans = dict["objTrans"]

    @staticmethod
    def split_pose_y(pose_y: np.ndarray, obj_idx: int = 0):
        """
        Split pose_y into translation (T,3) and quaternion (T,4).

        Parameters
        pose_y : Shape (T, N, 7)  Layout: [w, x, y, z, tx, ty, tz].
        obj_idx : int  Object index to select when pose_y is (T, N, 7).

        Returns
        t : (T, 3) float64  Translation per frame.
        q : (T, 4) float64  Quaternion per frame in wxyz order.
        """
        pose_y = np.asarray(pose_y)

        if pose_y.ndim == 3 and pose_y.shape[-1] == 7:
            X = pose_y[:, obj_idx, :]  # pick one object → (T, 7)
        else:
            raise ValueError("pose_y must be (T, N, 7) or (T, 7) with last dim = 7")

        objQuat = X[:, :4].astype(np.float64)  # w, x, y, z
        objTrans = X[:, 4:7].astype(np.float64)  # tx, ty, tz
        axes, angles = quaternionToAxisAngle(objQuat)
        objRot = axisAngleToRotvec(axes, angles)

        return {"objRot": objRot, "objTrans": objTrans, "objQuat": objQuat}

    @property
    def get_handPose(self, frame=None):
        return self.handPose if frame is None else self.handPose[frame]   # (T,48) or (48,)

    @property
    def get_handTrans(self, frame=None):
        return self.handTrans if frame is None else self.handTrans[frame]   # (T,3) or (3,)

    @property
    def getHandJoint3D(self, frame=None):
        return self.handJoints3D if frame is None else self.handJoints3D[frame]  # (T,21,3) or (21,3)

    @property
    def get_objTrans(self, frame=None):
        return self.objTrans if frame is None else self.objTrans[frame]   # (T,3) or (3,)

    @property
    def get_objRot(self, frame=None):
        return self.objRot if frame is None else self.objRot[frame]  # (T,3,3) or (3,3)

    @property
    def get_side(self):
        return self.side

    @property
    def get_ycb_name(self):
        return self.objName

    @property
    def get_handBeta(self):
        return self.handBeta    # (10,)

    @property
    def get_num_frames(self):
        return self.num_frames

    @property
    def get_seq_path(self):
        return self.seq_name

    @property
    def get_joint_order(self):
        return self.order

    def as_dict(self, frame=None):
        """
        Return a simple dict of core fields for this sequence.
        Uses the existing property getters (full sequences).
        """
        if frame is None:
            return {
                "seqName": self.seq_name,
                "handPose": self.get_handPose,  # (T, 48)
                "handTrans": self.get_handTrans,  # (T, 3)
                "handBeta": self.get_handBeta,  # (10,)
                "objRot": self.get_objRot,  # (T, 3, 3) or (T, 3) depending on your split
                "objTrans": self.get_objTrans,  # (T, 3)
                "objName": self.get_ycb_name,  # str
                "handJoints3D": self.getHandJoint3D,  # (T, 21, 3)
                "side": self.get_side,
                "num_frames": self.get_num_frames, # T
                "order": self.get_joint_order,
            }
        else:
            return {
                "seqName": self.seq_name,
                "handPose": self.get_handPose[frame],  # (T, 48)
                "handTrans": self.get_handTrans[frame],  # (3,)
                "handBeta": self.get_handBeta,  # (10,)
                "objRot": self.get_objRot[frame],  # (3,)
                "objTrans": self.get_objTrans[frame],  # (T, 3)
                "objName": self.get_ycb_name,  # str
                "handJoints3D": self.getHandJoint3D[frame],  # (T, 21, 3)
                "side": self.get_side,
                "frame": frame,
                "order": self.get_joint_order,
            }



def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True,
                        help="Sequence name, e.g., 20200709-subject-01/20200709_141754")
    args = parser.parse_args()

    print("[env] DEX_YCB_DIR =", os.environ.get("DEX_YCB_DIR", "<unset>"))

    loader = DexYCBLoader(args.seq)
    d = loader.as_dict  # use the property

    print(f"[seq] path        : {loader.seq_dir}")
    print(f"[mano] side       : {loader.get_side}")
    print(f"[mano] betas      : shape={d['handBeta'].shape}")
    print(f"[mano] handPose   : {d['handPose'].shape}")
    print(f"[mano] handTrans  : {d['handTrans'].shape}")
    print(f"[mano] joints3D   : {d['handJoints3D'].shape}")
    print(f"[ycb ] name       : {d['objName']}")
    print(f"[ycb ] objTrans   : {d['objTrans'].shape}")
    print(f"[ycb ] objRot     : {d['objRot'].shape}")

if __name__ == "__main__":
    main()

