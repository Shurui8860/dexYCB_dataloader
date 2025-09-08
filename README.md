# DexYCB Dataset Preprocessing

This repo exports the **DexYCB** dataset into lightweight **per-frame `.pkl`** files
for fast loading in training/evaluation pipelines. It mirrors the original
**subject / sequence** layout and splits data by hand **side**.

The dataset itself and baseline tasks were introduced in:

**DexYCB: A Benchmark for Capturing Hand Grasping of Objects**  
Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay,
Yashraj S. Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, Jan Kautz, Dieter Fox  
CVPR 2021 — [[paper]](https://dex-ycb.github.io/assets/chao_cvpr2021.pdf) • [[supp]](https://dex-ycb.github.io/assets/chao_cvpr2021_supp.pdf) • [[video]](https://youtu.be/Q4wyBaZeBw0) • [[arXiv]](https://arxiv.org/abs/2104.04631) • [[project site]](https://dex-ycb.github.io)

## DexYCB Dataset Structure
### Directory Layout
```
dexYCB_dataset/
├─ config/
│  ├─ hand_splits.yaml        # master list of left/right sequences
│  ├─ left_side.csv           # flat list of left sequences
│  └─ right_side.csv          # flat list of right sequences
├─ left/
│  └─ <subject>/
│     └─ <sequence>/
│        └─ meta/
│           ├─ 0000.pkl
│           ├─ 0001.pkl
│           └─ ...
└─ right/
   └─ <subject>/
      └─ <sequence>/
         └─ meta/
            ├─ 0000.pkl
            ├─ 0001.pkl
            └─ ...
```


### Per-Frame Schema
Each frame file (e.g., 0000.pkl) contains a dictionary with the following structure:

| Field          |  Shape | Type            | Units         | Description                                                                                          |
| -------------- | :----: | --------------- | ------------- | ---------------------------------------------------------------------------------------------------- |
| `seqName`      |    –   | `str`           | –             | Subject/sequence identifier.                                                                         |
| `side`         |    –   | `str`           | –             | Hand side: `left` or `right`.                                                                        |
| `frame`        |    –   | `int`           | –             | Zero-based frame index within the sequence.                                                          |
| `order`        |    –   | `str`           | –             | Joint indexing convention used for `handJoints3D` (e.g., `ho3d`, `mano`).                            |
| `handPose`     |  (48,) | `numpy.ndarray` | radians       | MANO pose parameters: **3** for global wrist rotation + **15 × 3** axis-angle for articulated joints. |
| `handTrans`    |  (3,)  | `numpy.ndarray` | meters        | Global hand translation (MANO wrist/root) in **world coordinates**.                                  |
| `handBeta`     |  (10,) | `numpy.ndarray` | –             | MANO shape coefficients.                                                  |
| `objRot`       |  (3,)  | `numpy.ndarray` | radians (vec) | Object rotation as **axis-angle** (rotation vector).                                                 |
| `objTrans`     |  (3,)  | `numpy.ndarray` | meters        | Object translation in **world coordinates**.                                                         |
| `objName`      |    –   | `str`           | –             | YCB object identifier.                                                                               |
| `handJoints3D` | (21,3) | `numpy.ndarray` | meters        | 3D hand joints in **world coordinates** (HO3D-21 order when `order: ho3d`).                          | 

---

#### Notes

* **Rotation conventions**

  * `objRot` uses the **rotation-vector (axis-angle)** representation: the vector’s **direction** is the axis; its **Euclidean norm** is the rotation **angle (radians)**.
  * `handPose` stores MANO axis-angle rotations: first **3** values = global wrist orientation; remaining **45** = **15 joints × 3**.

* **Coordinate frames**

  * Quantities marked *world coordinates* share a consistent world frame per sequence. Translations are in **meters**.

* **Data types**

  * Arrays are typically `float64` unless cast to `float32` by your pipeline. Units are unchanged.

* **Joint order**

  * `handJoints3D` follows **HO3D-21** when `order: ho3d`. If your models expect MANO-21, remap with your permutation utility.


### Sample Data Structure

```python
{
    'seqName': '20200709-subject-01/20200709_143957',      # subject/sequence id
    'side': 'right',                                       # hand side
    'frame': 29,                                           # frame index (zero-based)
    'order': 'ho3d',                                       # joint order used

    'handPose': array([2.111, 0.926, -2.019, 0.412, 0.122, 0.392, ...]),        # 48 MANO pose params
    'handTrans': array([0.154, -0.096, 0.907]),                                 # hand position (m, world)
    'handBeta': array([0.699, -0.169, -0.896, -0.098, 0.078, 0.336, ...]),      # 10 MANO shape coeffs

    'objRot':   array([1.055, -1.610, -1.124]),                                 # object rotation (axis-angle, rad)
    'objTrans': array([0.210, -0.095, 1.059]),                                  # object translation (m, world)
    'objName':  '008_pudding_box',                                              # YCB object identifier

    'handJoints3D': array([
        [0.247, -0.089, 0.913],
        [0.232, -0.125, 0.992],
        ...
    ]),  # 3D hand joints (meters); full shape is (21, 3)
}
```

## Implementation Notes

These notes explain how to 

1. Peek a single sequence.
2. Split the dataset into **right**/**left** lists.
3. Export per-frame `.pkl` files for all sequences.

---

### 1) Peek one sequence (quick sanity check)

The loader prints shapes and basic metadata for a single sequence.

```bash
# 1) point to your DexYCB data root
export DEX_YCB_DIR=/path/to/dex-ycb/data

# 2) inspect one sequence (subject/sequence)
python dexycbloader.py --seq 20200709-subject-01/20200709_141754
```

* `dexycbloader.py` expects a `--seq` argument and internally reads `meta.yml` + `pose.npz`, computes MANO joints, and prints field shapes and the grasped object name.&#x20;
* The CLI in `dexycbloader.py` uses the default joint order (`order="mano"`). If you need HO3D reindexing for joints, instantiate `DexYCBLoader(..., order="ho3d")` programmatically. Conversion utilities live in `loader_utils.py`. &#x20;

---

### 2) Split sequences into **right**/**left**

We scan all `meta.yml` files under `$DEX_YCB_DIR`, read `mano_sides`, and write:

```
dexYCB_dataset/config/
  ├─ hand_splits.yaml
  ├─ left_side.csv
  └─ right_side.csv
```

**Build the split files**

```bash
# Uses $DEX_YCB_DIR automatically; writes to ./dexYCB_dataset/config/
python type_split.py
```

* The splitter records **relative** `subject/sequence` paths by default and produces the YAML manifest + two CSVs.&#x20;
* You can also import and run it from Python if you want a different output folder:

  ```python
  from type_split import HandSplitIndex
  idx = HandSplitIndex(out_dir="dexYCB_dataset/config")
  yml = idx.build(relative=True)  # writes CSVs + YAML, returns YAML path
  ```



**Read one side back (optional)**

```python
from type_split import HandSplitIndex
right_paths = HandSplitIndex.read_side_paths("dexYCB_dataset/config/hand_splits.yaml",
                                             side="right", absolute=True)
```



> **Why split first?**
> The exporter consumes the `hand_splits.yaml` + CSVs to know which sequences to process. Processing **all** sequences must happen **after** this split step is complete.

---

### 3) Export per-frame pickles for all sequences

The exporter now runs from **`dex_ycb_loader/processor.py`** and reads its settings from a **`config.yaml`** file (same folder as the script). It writes per-frame pickles to:

```
dexYCB_dataset/<side>/<subject>/<sequence>/meta/0000.pkl, 0001.pkl, ...
```

(Behavior implemented in `processor.py`.)&#x20;

#### Basic usage (config-driven)

1. Edit `config.yaml` (example below), then run:

```bash
python dex_ycb_loader/processor.py
```

**Example `config.yaml`:**

```yaml
# Where to put the pickles (absolute or project-relative)
out_root: dexYCB_dataset

# Which split to process: "right", "left", or "both"
side: right

# Joint order passed to DexYCBLoader: "ho3d" (default) or "mano"
order: 
    name: "HO3D"
    joints:
      wrist:  [0]
      index:  [1, 2, 3, 17]
      middle: [4, 5, 6, 18]
      ring:   [10, 11, 12, 19]
      pinky:  [7, 8, 9, 20]
      thumb:  [13, 14, 15, 16]

# Path to your hand-split manifest (absolute or project-relative)
hand_splits: dexYCB_dataset/config/hand_splits.yaml
```

What it does:

* Preserves the canonical `subject/sequence` layout and writes zero-padded `0000.pkl, 0001.pkl, …`.&#x20;
* If `side: both`, it processes left **and** right splits in one run.&#x20;
* If `hand_splits` is omitted, it falls back to `project_root/dexYCB_dataset/config/hand_splits.yaml`.&#x20;

#### Process the left-hand split (or both)

Set in `config.yaml`:

```yaml
side: left    # or: both
```

…and rerun `python dex_ycb_loader/processor.py`.&#x20;

#### (Optional) Choose a different joint order

Set in `config.yaml`:

```yaml
order: mano   # default is ho3d
```

This value is passed directly to `DexYCBLoader`.&#x20;

#### Programmatic use (override config in code)

```python
from dex_ycb_loader.processor import DexYCBPickleExporter

exporter = DexYCBPickleExporter(
    out_root="dexYCB_dataset",
    side="right",                 # "left" or "both"
    order="mano",                 # or "ho3d"
    yml="dexYCB_dataset/config/hand_splits.yaml"
)
exporter.process_all()
```

This preserves `subject/sequence` and writes per-frame pickles under `<out_root>/<side>/.../meta/*.pkl`.&#x20;
