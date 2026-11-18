import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(rel):
    p = ROOT / rel
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_config_loads_example():
    from tailcycle_run import config
    c = config.load(ROOT / "config.example.toml")
    assert config.get("cluster.queue") == "gpu_l4"
    assert config.get("cluster.work_dir").startswith("~/")
    assert config.sealed_animals() == set()
    assert "example" in config.sources()
    assert config.sources()["example"]["min_cams"] == 16
    with pytest.raises(SystemExit):
        config.get("cluster.no_such_key")
    assert config.get("cluster.no_such_key", 7) == 7


def test_ranges_parser(tmp_path):
    from tailcycle_run.plan import parse_ranges
    f = tmp_path / "r.txt"
    f.write_text("# windows\n5403 5790\n\n8363 8792  # second\n")
    assert parse_ranges(f) == [(5403, 5790), (8363, 8792)]
    f.write_text("10 5\n")
    with pytest.raises(ValueError):
        parse_ranges(f)
    f.write_text("1 2 3\n")
    with pytest.raises(ValueError):
        parse_ranges(f)


def test_plan_projection(tmp_path):
    from tailcycle_run.plan import plan_ranges, projection
    # 119,618 frames in 263 windows -> 23.92 + 4.38 = 28.31 GPU-h
    assert projection(119618, 263) == pytest.approx(28.307, abs=1e-3)
    assert projection(0, 60) == pytest.approx(1.0)
    f = tmp_path / "r.txt"
    f.write_text("0 5000\n10000 15000\n")
    p = plan_ranges(f)
    assert p["windows"] == 2 and p["frames"] == 10000
    assert p["gpu_h"] == pytest.approx(2 + 2 / 60)


def test_body_span_max_pairwise():
    bs = _load("tailcycle_run/post/body_span.py")
    x = np.full((3, 4, 3), np.nan, np.float32)
    # frame 0: points on a 3-4-5 triangle -> max pairwise 5
    x[0, 0] = [0, 0, 0]; x[0, 1] = [3, 0, 0]; x[0, 2] = [0, 4, 0]
    # frame 1: two points 10 apart along z, others missing
    x[1, 0] = [1, 1, 0]; x[1, 3] = [1, 1, 10]
    # frame 2: a single point -> no pair -> nan
    x[2, 1] = [5, 5, 5]
    out = bs.max_pairwise(x)
    assert out[0] == pytest.approx(5.0)
    assert out[1] == pytest.approx(10.0)
    assert out[2] == pytest.approx(0.0) or np.isnan(out[2])


def test_submit_refuses_sealed_animal(tmp_path):
    import os
    import subprocess
    cfg = (ROOT / "config.example.toml").read_text().replace(
        "sealed_animals = []", 'sealed_animals = ["heldout"]')
    c = tmp_path / "c.toml"
    c.write_text(cfg)
    r = subprocess.run([sys.executable, str(ROOT / "tailcycle_run/cluster/cluster_run.py"),
                        "--animal", "heldout", "--session", "2025_01_01_00_00_00"],
                       capture_output=True, text=True, timeout=60,
                       env={**os.environ, "TAILCYCLE_RUN_CONFIG": str(c)})
    assert r.returncode != 0 and "SEALED" in (r.stderr + r.stdout)
