"""data3D.csv read/write for any keypoint count (shared by yolo_clean and body_span)."""
from __future__ import annotations
from pathlib import Path

import numpy as np


def load_any(path: Path):
    """(header lines, F x K x 4 array) for a data3D.csv of any keypoint count."""
    with open(path) as fh:
        head = [fh.readline(), fh.readline()]
    names = head[0].rstrip("\n").split(",")
    k = len(names) // 4
    arr = np.genfromtxt(path, delimiter=",", skip_header=2, usecols=range(4 * k),
                        dtype=np.float32, missing_values=("", "NaN"),
                        filling_values=np.nan)
    return head, arr.reshape(len(arr), k, 4)


def write_any(head, data, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.writelines(head)
        np.savetxt(fh, data.reshape(len(data), -1), delimiter=",", fmt="%.4f")
