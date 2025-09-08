#!/usr/bin/env python3
import os
import yaml
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from loader_utils import JointConvention
from dexycbloader import DexYCBLoader
from type_split import HandSplitIndex


class DexYCBPickleExporter:
    """
    Export per-frame dictionaries from DexYCB sequences into pickles.

    - Preserves `subject/sequence` path by default:
        out_root / side / subject / sequence / meta / 0000.pkl
    """

    def __init__(self, out_root: Union[str, Path] = "dexYCB_dataset", side: str = "left", order="ho3d",
        yml: str = None, cfg: Optional[Union[str, Path]] = None,  # optional YAML path
    ):
        # Defaults from args
        self.out_root = Path(out_root)
        self.side: str = side
        self.order = order
        self.project_root = Path(__file__).resolve().parents[1]

        if yml is None:
            self.yml =  self.project_root / "dexYCB_dataset" / "config" / "hand_splits.yaml"
            print(self.yml)
        else:
            self.yml: Optional[Path] = self.project_root / Path(yml)  # path to hand_splits.yaml (optional)

        # If a YAML config path exists, load and override from it
        if cfg is not None:
            self.from_yaml(cfg)

    def from_yaml(self, cfg: Union[str, Path]):
        """
        Read exporter settings from YAML and set attributes in-place.
        Accepts keys:
          - out_root: str|path
          - side: "right"|"left"
          - order: "mano"|"ho3d"|{name:..., joints:{...}} | (alias) JointConvention: {...}
          - yml | hand_splits | hand_splits_yaml: path to hand_splits.yaml
        """
        cfg = Path(cfg)
        print("Reading settings from {}".format(cfg))

        # If it's not absolute, resolve it relative to *this* file's directory
        if not cfg.is_absolute():
            here = Path(__file__).resolve().parent
            cfg = (here / cfg).resolve()
        if not Path(cfg).exists():
            raise FileNotFoundError(cfg)

        cfg = yaml.safe_load(cfg.read_text()) or {}

        # out_root (resolve relative to YAML)
        out_root_val = cfg.get("out_root", self.out_root)
        out_root_path = Path(out_root_val)
        self.out_root = out_root_path if out_root_path.is_absolute() else (self.project_root / out_root_path).resolve()

        # side
        self.side = str(cfg.get("side", self.side))

        # order: None | "mano"/"ho3d" | mapping -> JointConvention
        node = cfg.get("order")
        if isinstance(node, dict):
            self.order = JointConvention(name=node.get("name"),
                                         layout=node.get("joints", {}))
        elif node is not None:
            self.order = node  # keep string ("mano"/"ho3d") or None

        # yml / hand_splits manifest (resolve relative to YAML)
        yml_val = cfg.get("hand_splits")
        if yml_val:
            yml_path = Path(yml_val)
            self.yml = yml_path if yml_path.is_absolute() else (self.project_root / yml_path).resolve()

        return self

    # ------------------------ path helpers ------------------------
    def out_dir(self, seq_ref: Union[str, Path], side: str) -> Path:
        """
        Build (and create) the output directory for a given sequence.
        """
        p = Path(seq_ref)
        parts = p.parts  # tuple of path components

        if len(parts) >= 2:
            # Preserve the canonical DexYCB structure: "<subject>/<sequence>"
            key = Path(parts[-2]) / parts[-1]
        else:
            # Only a single component was provided; use it as the sequence name
            key = Path(p.name)

        # Compose: out_root/<side>/<subject>/<sequence>/meta/
        # `self.side` typically "left" or "right"
        out_dir = self.out_root / side / key / "meta"

        # Ensure the directory exists (create parents as needed)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # ------------------------ core work ------------------------
    def process(self, seq_ref: Union[str, Path], side: str) -> Path:
        """
        Process one sequence:
        - Build a loader key with forward slashes (DexYCBLoader requirement).
        - Iterate all frames.
        - Serialize each frame's dict to `<out_dir>/<frame_idx>.pkl`.
        """
        # Normalize the sequence reference into your canonical path/key
        loader_key = str(seq_ref).replace(os.sep, "/")

        # Construct the per-sequence data loader (order comes from your config)
        loader = DexYCBLoader(loader_key, order=self.order)

        # Total number of frames in this sequence (property on the loader)
        num_frames = loader.get_num_frames

        # Destination directory for this sequence’s per-frame pickles
        out_dir = self.out_dir(seq_ref, side)

        for frame_idx in range(num_frames):
            # Build a per-frame dictionary (must contain only pickle-able objects)
            frame_dict = loader.as_dict(frame_idx)

            # Ensure output directory exists (can be hoisted outside loop for micro-perf)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Write as zero-padded file names: 0000.pkl, 0001.pkl, ...
            out_file = out_dir / f"{frame_idx:04d}.pkl"
            with out_file.open("wb") as f:
                pickle.dump(frame_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Simple progress/logging line
        print(f"[done] {loader_key}: {num_frames} frames -> {out_dir}")

    def process_all(self):
        """
        Load the split file at `self.yml` and process sequences for the configured side(s).
        Notes
        `self.side` can be "left", "right", or "both".
        `self.yml` should be a YAML produced by HandSplitIndex, with top-level keys
    """
        # Normalize to a list of sides to iterate over.
        # If "both", handle left then right; otherwise just the requested side.
        sides = ["left", "right"] if self.side == "both" else [self.side]

        for side in sides:
            # Pull the sequence list for this side from the YAML.
            # `files` is a list of absolute paths to sequences (or sequence roots).
            files: List[str] = HandSplitIndex.read_side_paths(
                self.yml, side=side, absolute=True
            )

            # Process each sequence one-by-one using the class's per-sequence routine.
            for seq_ref in files:
                # `seq_ref` is typically the sequence directory (e.g., .../<SEQ_NAME>/)
                # and will be consumed by `process`.
                self.process(seq_ref, side)


if __name__ == "__main__":
    exporter = DexYCBPickleExporter(cfg="config.yaml")
    exporter.process_all()
