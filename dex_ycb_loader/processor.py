#!/usr/bin/env python3
import os
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

    def __init__(self, out_root: Union[str, Path] = "dexYCB_dataset", side: str = "right",order=None,
        cfg: Optional[Union[str, Path]] = None,  # optional YAML path
    ):
        # Defaults from args
        self.out_root = Path(out_root)
        self.side: str = side
        self.order = order
        self.yml: Optional[Path] = None  # path to hand_splits.yaml (optional)

        # If a YAML config path exists, load and override from it
        if cfg is not None and Path(cfg).exists():
            self.from_yaml(cfg)

    def from_yaml(self, cfg_path: Union[str, Path]):
        """
        Read exporter settings from YAML and set attributes in-place.
        Accepts keys:
          - out_root: str|path
          - side: "right"|"left"
          - order: "mano"|"ho3d"|{name:..., joints:{...}} | (alias) JointConvention: {...}
          - yml | hand_splits | hand_splits_yaml: path to hand_splits.yaml
        """
        cfg_path = Path(cfg_path)
        base = cfg_path.parent
        cfg = yaml.safe_load(cfg_path.read_text()) or {}

        # out_root (resolve relative to YAML)
        out_root_val = cfg.get("out_root", self.out_root)
        out_root_path = Path(out_root_val)
        self.out_root = out_root_path if out_root_path.is_absolute() else (base / out_root_path).resolve()

        # side
        self.side = str(cfg.get("side", self.side))

        # order: None | "mano"/"ho3d" | mapping -> JointConvention
        node = cfg.get("order") or cfg.get("JointConvention")
        print(node)
        if isinstance(node, dict):
            self.order = JointConvention(name=node.get("name"),
                                         layout=node.get("joints", {}))
        elif node is not None:
            self.order = node  # keep string ("mano"/"ho3d") or None

        # yml / hand_splits manifest (resolve relative to YAML)
        yml_val = cfg.get("yml") or cfg.get("hand_splits") or cfg.get("hand_splits_yaml")
        if yml_val:
            yml_path = Path(yml_val)
            self.yml = yml_path if yml_path.is_absolute() else (base / yml_path).resolve()

        return self

    # ------------------------ path helpers ------------------------
    def _seq_key(self, seq_ref: Union[str, Path]) -> Path:
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

    def out_dir_for_sequence(self, seq_ref: Union[str, Path]) -> Path:
        """
        Ensure and return: out_root/<side>/<subject>/<sequence>/meta/ (or sequence-only)
        """
        key = self._seq_key(seq_ref)
        out_dir = self.out_root / self.side / key / "meta"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # ------------------------ IO helpers ------------------------
    @staticmethod
    def _save_frame_pickle(frame_dict: Dict[str, Any], out_dir: Path, frame_idx: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{frame_idx:04d}.pkl"
        with out_file.open("wb") as f:
            pickle.dump(frame_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        out_dir = self.out_dir_for_sequence(seq_ref)

        for frame in range(num_frames):
            frame_data = loader.as_dict(frame)  # per-frame dictionary
            self._save_frame_pickle(frame_data, out_dir, frame)

        print(f"[done] {loader_key}: {num_frames} frames -> {out_dir}")

    def process_from_yaml(self, yml: Union[str, Path]):
        """
        Read the YAML split file and process all sequences for `self.side`.
        """
        files: List[str] = HandSplitIndex.read_side_paths(yml, side=self.side, absolute=True)
        for seq_ref in files:
            self.process_sequence(seq_ref)

    # ------------------------ convenience ------------------------

    @staticmethod
    def default_hand_splits_yaml() -> Path:
        """
        Return project_root/dexYCB_dataset/config/hand_splits.yaml
        (project_root = parent of this script's folder)
        """
        project_root = Path(__file__).resolve().parents[1]
        return project_root / "dexYCB_dataset" / "config" / "hand_splits.yaml"


# ---------------------------- CLI ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DexYCB sequences to per-frame pickles.")
    p.add_argument("--yml", type=Path, help="Path to hand_splits.yaml.")
    p.add_argument("--out_root", type=Path, default=Path("dexYCB_dataset"))
    p.add_argument("--side", choices=["right", "left"], default="right")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    yml_path = args.yml or DexYCBPickleExporter.default_hand_splits_yaml()
    # exporter = DexYCBPickleExporter(
    #     out_root=args.out_root,
    #     side=args.side,
    #     order="ho3d"
    # )
    exporter = DexYCBPickleExporter(cfg="config.yaml")
    exporter.process_from_yaml(yml_path)


if __name__ == "__main__":
    main()
