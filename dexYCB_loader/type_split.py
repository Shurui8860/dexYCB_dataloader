from __future__ import annotations

import os
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Iterable, Union, Optional


class HandSplitIndex:
    """
    Build and read left/right sequence indices for DexYCB by scanning `meta.yml`
    files and reading the `mano_sides` field.

    Typical usage:
        idx = HandSplitIndex()  # uses $DEX_YCB_DIR
        idx.build(relative=True, out_dir="dexYCB_dataset/config")
        right_abs = idx.read_side_paths(idx.default_yaml_path(), side="right", absolute=True)
    """

    # ------------------------- construction -------------------------
    def __init__(self, data_root: Optional[Union[str, Path]] = None, out_dir: Union[str, Path] = ".",
                 encoding: str = "utf-8") -> None:
        if data_root is None:
            assert "DEX_YCB_DIR" in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
            data_root = os.environ["DEX_YCB_DIR"]

        self.root: Path = Path(data_root).resolve()
        self.out_dir: Path = Path(out_dir)
        self.encoding = encoding

    # ------------------------- helpers -------------------------
    def default_yaml_path(self) -> Path:
        """Default manifest path: <out_dir>/hand_splits.yaml."""
        return (self.out_dir / "hand_splits.yaml").resolve()

    # ------------------------- core split -------------------------
    def split(self, *, relative: bool = True) -> Dict[str, List[str]]:
        """
        Scan `self.root` for all `meta.yml` files, read `mano_sides`,
        and split sequences into LEFT / RIGHT.

        Returns a dict with 'left' and 'right' lists of paths (strings),
        relative to `self.root` if `relative=True`, else absolute strings.
        """
        right: List[str] = []
        left: List[str] = []

        meta_files = sorted(self.root.rglob("meta.yml"))
        print(f"[split] found {len(meta_files)} meta.yml files")

        for meta_p in meta_files:
            seq_dir = meta_p.parent

            with meta_p.open("r", encoding=self.encoding) as f:
                data = yaml.safe_load(f) or {}
            sides = data.get("mano_sides", [])

            # Normalize path once
            if relative:
                seq_str = str(seq_dir.relative_to(self.root))
            else:
                seq_str = str(seq_dir.resolve())

            # Allow both if present (robustness)
            if "right" in sides:
                right.append(seq_str)
            if "left" in sides:
                left.append(seq_str)

        right, left = sorted(set(right)), sorted(set(left))
        print(f"[split] totals: right={len(right)}  left={len(left)}.")
        return {"right": right, "left": left}

    # ------------------------- writers -------------------------
    def write_csvs(self, splits: Dict[str, Iterable[Union[str, Path]]],
                   out_dir: Optional[Union[str, Path]] = None) -> tuple[Path, Path]:
        """
        Write two CSV files (left_side.csv, right_side.csv) listing sequence paths.
        Each row contains one path (as given, typically relative to data_root).
        """
        out = Path(out_dir) if out_dir is not None else self.out_dir
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}
        for side in ("left", "right"):
            rows = [str(p) for p in splits.get(side, [])]
            csv_path = out / f"{side}_side.csv"
            with csv_path.open("w", newline="", encoding=self.encoding) as f:
                writer = csv.writer(f)
                for r in rows:
                    writer.writerow([r])
            paths[side] = csv_path

        return paths["left"].resolve(), paths["right"].resolve()

    def write_yaml(self, yaml_path: Optional[Union[str, Path]] = None,
                   left_csv: Optional[Union[str, Path]] = None,
                   right_csv: Optional[Union[str, Path]] = None) -> Path:
        """
        Create a YAML manifest with keys: data_root, left, right.
        CSV paths are stored relative to the YAML file's directory if provided as relative paths.
        """
        ypath = Path(yaml_path) if yaml_path is not None else self.default_yaml_path()
        ypath.parent.mkdir(parents=True, exist_ok=True)

        # Default CSV filenames if not given
        left_rel = Path(left_csv) if left_csv is not None else Path("left_side.csv")
        right_rel = Path(right_csv) if right_csv is not None else Path("right_side.csv")

        manifest = {
            "data_root": str(self.root),
            "left": str(left_rel),
            "right": str(right_rel),
        }

        with ypath.open("w", encoding=self.encoding) as f:
            yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)

        return ypath.resolve()

    def build(self, *, relative: bool = True, out_dir: Optional[Union[str, Path]] = None,
              yaml_path: Optional[Union[str, Path]] = None) -> Path:
        """
        End-to-end:
          1) Split sequences by hand from `self.root`
          2) Write left/right CSVs into `out_dir` (or self.out_dir)
          3) Write YAML manifest and return its path
        """
        if out_dir is None:
            out_dir = self.out_dir

        # 1) split
        splits = self.split(relative=relative)

        # 2) csvs
        left_csv, right_csv = self.write_csvs(splits, out_dir=out_dir)

        # 3) yaml (store CSV names relative to YAML dir)
        ypath = Path(yaml_path) if yaml_path is not None else Path(out_dir) / "hand_splits.yaml"
        return self.write_yaml(
            yaml_path=ypath,
            left_csv=Path(left_csv).name,
            right_csv=Path(right_csv).name,
        )

    # ------------------------- readers -------------------------
    @staticmethod
    def read_side_paths(index_yaml: Union[str, Path], side: str = "right",
                        absolute: bool = True, encoding: str = "utf-8") -> List[Path]:
        """
        Load a YAML manifest and return the list of sequence paths for one side.

        YAML keys:
          - data_root: absolute path to the dataset root
          - left:  path (relative to the YAML file) to a CSV listing left sequences
          - right: path (relative to the YAML file) to a CSV listing right sequences

        CSV format: one path per row (relative to `data_root`).
        """
        side_norm = side.strip().lower()
        if side_norm not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        yaml_path = Path(index_yaml).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML not found: {yaml_path}")

        manifest = yaml.safe_load(yaml_path.read_text(encoding=encoding))
        if "data_root" not in manifest or side_norm not in manifest:
            raise KeyError("YAML must contain 'data_root' and a CSV entry for the requested side")

        data_root = Path(manifest["data_root"]).resolve()
        csv_rel = manifest[side_norm]
        csv_path = (yaml_path.parent / csv_rel).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found for side '{side_norm}': {csv_path}")

        paths: List[Path] = []
        with csv_path.open("r", newline="", encoding=encoding) as f:
            for row in csv.reader(f):
                if not row:
                    continue
                raw = row[0].strip()
                if not raw or raw.startswith("#"):
                    continue
                p = Path(raw)
                if absolute and not p.is_absolute():
                    p = (data_root / p).resolve()
                paths.append(p)

        return paths


# ------------------------- quick demo -------------------------
if __name__ == "__main__":
    # Example: build using $DEX_YCB_DIR, write outputs under ./dexYCB_dataset/config
    idx = HandSplitIndex(out_dir="dexYCB_dataset/config")
    yml = idx.build(relative=True)
    print(f"[ok] wrote manifest: {yml}")

    # Example: read right-hand absolute paths back
    right_paths = HandSplitIndex.read_side_paths(yml, side="right", absolute=True)
    print(f"[ok] right count: {len(right_paths)}")
    for p in right_paths[:3]:
        print("  R:", p)
