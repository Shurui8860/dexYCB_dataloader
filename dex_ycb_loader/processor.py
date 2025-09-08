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
    @staticmethod
    def _seq_key(seq_ref: Union[str, Path]) -> Path:
        """
        Turn an arbitrary seq_ref (absolute or relative) into the key used by:
          - DexYCBLoader (expects 'subject/sequence')
          - Output directory layout
        """
        p = Path(seq_ref)
        parts = p.parts

        if len(parts) >= 2:
            # Keep `subject/sequence`
            key = Path(parts[-2]) / parts[-1]
        else:
            # Only the final sequence name
            key = Path(p.name)

        return key

    def out_dir(self, seq_ref: Union[str, Path]) -> Path:
        """
        Ensure and return: out_root/<side>/<subject>/<sequence>/meta/ (or sequence-only)
        """
        key = self._seq_key(seq_ref)
        out_dir = self.out_root / self.side / key / "meta"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # ------------------------ core work ------------------------
    def process_sequence(self, seq_ref: Union[str, Path]):
        """
        Process a single sequence: iterate frames and dump pickles.
        `DexYCBLoader` expects 'subject/sequence' with forward slashes.
        """
        key = self._seq_key(seq_ref)
        loader_key = str(key).replace(os.sep, "/")
        loader = DexYCBLoader(loader_key, order=self.order)

        num_frames = loader.get_num_frames  # property in your loader
        out_dir = self.out_dir(seq_ref)

        for frame_idx in range(num_frames):
            frame_dict = loader.as_dict(frame_idx)  # per-frame dictionary

            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{frame_idx:04d}.pkl"
            with out_file.open("wb") as f:
                pickle.dump(frame_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[done] {loader_key}: {num_frames} frames -> {out_dir}")

    def process_from_yaml(self):
        """
        Read the YAML split file and process all sequences for `self.side`.
        """
        print(self.yml)
        files: List[str] = HandSplitIndex.read_side_paths(self.yml, side=self.side, absolute=True)
        for seq_ref in files:
            self.process_sequence(seq_ref)


if __name__ == "__main__":
    exporter = DexYCBPickleExporter(cfg="config.yaml")
    exporter.process_from_yaml()
