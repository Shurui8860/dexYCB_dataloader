from pathlib import Path
from typing import Dict, List, Tuple
import csv
from dexycbloader import DexYCBLoader
from type_split import HandSplitIndex


class ObjFinder:
    """
    Scan DexYCB dataset sequences for a given hand side,
    group them by object, and write results into CSV files.
    """

    def __init__(self, yml_path: Path, side: str = "right", out_dir: Path = Path("dexYCB_dataset/objs")):
        self.project_root = Path(__file__).resolve().parents[1]
        self.yml_path = Path(yml_path)
        self.side = side
        self.out_dir = self.project_root / out_dir
        self.obj_map: Dict[str, List[Tuple[str, str]]] = {}  # object -> list of (subject, sequence)

    def scan_sequences(self):
        """Scan all sequences and populate self.obj_map grouped by object name."""
        seq_paths = HandSplitIndex.read_paths(self.yml_path, side=self.side, absolute=True)

        for seq_ref in seq_paths:
            try:
                loader = DexYCBLoader(str(seq_ref))  # auto-reads meta.yml
                obj_name = loader.objName
                subject = seq_ref.parent.name
                sequence = seq_ref.name
                self.obj_map.setdefault(obj_name, []).append((subject, sequence))
            except Exception as e:
                print(f"[warn] Skipping {seq_ref}: {e}")
                continue

    def write_csvs(self):
        """Write one CSV file per object into self.out_dir."""
        for obj_name, records in self.obj_map.items():
            out_dir = self.out_dir / f"{obj_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / f"{obj_name}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["subject", "sequence"])
                writer.writerows(records)
            print(f"[info] Wrote {len(records)} sequences to {csv_path}")

    def get_results(self) -> Dict[str, List[str]]:
        """Return results as a dict mapping object -> list of 'subject/sequence' paths."""
        return {obj: [f"{s}/{seq}" for s, seq in records] for obj, records in self.obj_map.items()}

    def run(self) -> Dict[str, List[str]]:
        """
        Full pipeline:
        1. Scan sequences
        2. Write per-object CSVs
        3. Return results as dict
        """
        self.scan_sequences()
        self.write_csvs()
        return self.get_results()

def read_sequence(obj_name: str, out_dir: Path = Path("dexYCB_dataset/objs")) -> List[str]:
    csv_path = out_dir / obj_name / f"{obj_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    sequences: List[str] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(f"{row['subject']}/{row['sequence']}")
    print(f"[info] {len(sequences)} sequences loaded for {obj_name}")
    return sequences

if __name__ == "__main__":
    # yml = Path("dexYCB_dataset/config/hand_splits.yaml")
    # finder = ObjFinder(yml, side="right")
    # results = finder.run()
    # print(f"[done] Processed {len(results)} objects")

    # Example: read back one object’s sequences directly
    obj = "010_potted_meat_can"
    seqs = read_sequence(obj)
