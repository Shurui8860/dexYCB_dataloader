#!/usr/bin/env python3
"""
OOP image-sequence → video builder.

Requires:
  pip install opencv-python
"""

from pathlib import Path
from typing import Optional, Tuple, List
import re
import cv2

_NUM = re.compile(r"(\d+)")

def _num_key(p: Path):
    """Natural-ish sort: first integer in filename, else fallback to name."""
    m = _NUM.search(p.name)
    return (int(m.group(1)) if m else float("inf"), p.name)

class ImageSequenceToVideo:
    """
    Build a video from a folder of JPG images.

    Parameters
    input_dir : str | Path
        Folder containing frames (e.g., color_000001.jpg ...).
    output : str | Path | None
        Output video path. If None, defaults to <input_dir>.mp4 in the parent.
    fps : int
        Frames per second.
    size : (int, int) | None
        (width, height). If None, inferred from first frame.
    pattern : str
        Glob pattern for frames (default: '*.jpg').
    codec : str | None
        FourCC code (e.g., 'mp4v', 'XVID'). If None, inferred from output suffix.
    stride : int
        Use every Nth frame (>=1).
    """

    def __init__(self, input_dir: str, output: Optional[str] = None, fps: int = 15,
        size: Optional[Tuple[int, int]] = None, pattern: str = "*.jpg", codec: Optional[str] = None,
        stride: int = 1,
    ):
        self.input_dir = Path(input_dir)
        if output is None:
            self.output = self.input_dir.parent / f"{self.input_dir.name}.mp4"
        else:
            self.output = Path(output)
        self.fps = int(fps)
        self.size = size  # (w, h)
        self.pattern = pattern
        self.codec = codec
        self.stride = max(1, int(stride))

    def _collect_frames(self) -> List[Path]:
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input folder not found: {self.input_dir}")
        files = sorted(self.input_dir.glob(self.pattern), key=_num_key)
        files = files[::self.stride]
        if not files:
            raise FileNotFoundError(f"No frames matched pattern '{self.pattern}' in {self.input_dir}")
        return files

    def _infer_codec(self) -> str:
        if self.codec:
            return self.codec
        suf = self.output.suffix.lower()
        if suf == ".mp4":
            return "mp4v"
        if suf in (".avi", ".mkv"):
            return "XVID"
        return "mp4v"

    def build(self) -> Path:
        frames = self._collect_frames()

        # Determine size
        if self.size is None:
            first = cv2.imread(str(frames[0]))
            if first is None:
                raise RuntimeError(f"Failed to read first frame: {frames[0]}")
            h, w = first.shape[:2]
            self.size = (w, h)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self._infer_codec())
        writer = cv2.VideoWriter(str(self.output), fourcc, self.fps, self.size)

        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for {self.output}")

        for p in frames:
            img = cv2.imread(str(p))
            if img is None:
                print(f"[warn] skip unreadable frame: {p}")
                continue
            h, w = img.shape[:2]
            if (w, h) != self.size:
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
            writer.write(img)

        writer.release()
        return self.output


---- Example usage ----
if __name__ == "__main__":
    maker = ImageSequenceToVideo(
        input_dir="/path/to/frames",
        output="/path/to/out/clip.mp4",  # or None to auto-name
        fps=30,
        size=None,            # infer from first frame
        pattern="color_*.jpg",
        codec=None,           # infer from output suffix
        stride=1,             # use every frame
    )
    out_path = maker.build()
    print("Saved:", out_path)
