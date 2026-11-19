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
    # 119,618 frames in 263 windows -> 23.92 + 11.84 = 35.76 GPU-h (measured 35.7)
    assert projection(119618, 263) == pytest.approx(35.759, abs=1e-3)
    assert round(projection(119618, 263), 1) == 35.8
    assert projection(0, 100) == pytest.approx(4.5)
    f = tmp_path / "r.txt"
    f.write_text("0 5000\n10000 15000\n")
    p = plan_ranges(f)
    assert p["windows"] == 2 and p["frames"] == 10000
    assert p["gpu_h"] == pytest.approx(2 + 2 * 0.045)


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


def test_lsf_index_spec_and_limit(tmp_path, monkeypatch):
    cr = _load("tailcycle_run/cluster/cluster_run.py")
    assert cr._idx_spec(range(1, 264)) == "1-263"
    assert cr._idx_spec([1, 2, 3, 4, 5, 9] + list(range(12, 21))) == "1-5,9,12-20"
    assert cr._idx_spec([7]) == "7"
    assert cr._idx_spec([3, 1, 2, 2]) == "1-3"
    monkeypatch.setattr(cr, "LOCAL_LOG_DIR", tmp_path)
    monkeypatch.delenv("TAILCYCLE_MAX_CONCURRENT", raising=False)
    assert cr._concurrency_limit("a__s") == ""
    (tmp_path / "tailcycle_max_concurrent.a__s").write_text("8\n")
    assert cr._concurrency_limit("a__s") == "%8"
    monkeypatch.setenv("TAILCYCLE_MAX_CONCURRENT", "12")
    assert cr._concurrency_limit("a__s") == "%12"


def test_gate_override_budget(tmp_path, monkeypatch):
    cr = _load("tailcycle_run/cluster/cluster_run.py")
    monkeypatch.setattr(cr, "LOCAL_LOG_DIR", tmp_path)
    monkeypatch.delenv("GATE_OVERRIDE_BUDGET", raising=False)
    assert "GATE_OVERRIDE_BUDGET" not in cr._gate_env("p")
    (tmp_path / "gate_override_budget").write_text("approved batch\n")
    assert "GATE_OVERRIDE_BUDGET='approved batch'" in cr._gate_env("p")
    monkeypatch.setenv("GATE_OVERRIDE_BUDGET", "from env")
    assert "GATE_OVERRIDE_BUDGET='from env'" in cr._gate_env("p")


def test_hurdle_windows_merge_split():
    hw = _load("tailcycle_run/post/hurdle_windows.py")
    # pad 2: frames 10 and 13 overlap -> one window; 30 alone; clipped at 0 / n_total
    assert hw.windows_from_frames([13, 10, 30, 0], 2, 32, 100) == [(0, 3), (8, 16), (28, 32)]
    # a 20-frame window split at max_frames 8
    assert hw.windows_from_frames([10], 10, 100, 8) == [(0, 8), (8, 16), (16, 21)]


def test_hurdle_windows_empty_map(tmp_path):
    import json
    import subprocess
    sd = tmp_path / "a" / "s"
    (sd / "objects").mkdir(parents=True)
    (sd / "objects" / "hurdle_map.json").write_text(json.dumps({"objects": []}))
    out = tmp_path / "r.txt"
    r = subprocess.run([sys.executable, str(ROOT / "tailcycle_run/post/hurdle_windows.py"),
                        "--pred-root", str(tmp_path), "--animal", "a", "--session", "s",
                        "--out", str(out)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "no mapped sheet" in r.stdout and not out.exists()


def test_run_stage_skips_and_stops_at_cap(tmp_path):
    """The plan driver skips a session that has output and stops before one
    whose projection exceeds the total cap -- both before any cluster call."""
    import os
    import subprocess
    vroot, proot = tmp_path / "v", tmp_path / "p"
    (proot / "m1" / "s1" / "tailcycle47_v2").mkdir(parents=True)
    (proot / "m1" / "s1" / "tailcycle47_v2" / "data3D.csv").write_text("x\n")
    cfg = (ROOT / "config.example.toml").read_text() + (
        f'\n[sources.t]\nvideo_root = "{vroot}"\npred_root = "{proot}"\n'
        f'animals = ["m1"]\nmin_cams = 8\n')
    cfg = cfg.replace('log_dir             = "/tmp/tailcycle_run"',
                      f'log_dir             = "{tmp_path / "logs"}"')
    c = tmp_path / "c.toml"
    c.write_text(cfg)
    plan = tmp_path / "plan.tsv"
    plan.write_text("tab\tanimal\tsession\tranges\tproj_gpuh\n"
                    "t\tm1\ts1\tr1.txt\t5.0\n"
                    "t\tm1\ts2\tr2.txt\t50.0\n")
    env = {**os.environ, "TAILCYCLE_RUN_CONFIG": str(c)}
    r = subprocess.run([sys.executable, "-m", "tailcycle_run", "run", "--plan", str(plan),
                        "--total-cap", "10"], capture_output=True, text=True, timeout=60,
                       env=env, cwd=ROOT)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "already has tailcycle47_v2, skipping" in out
    assert "STOP: spent 0.0 + projected 50.0 > total cap 10" in out
    assert "BATCH END: total 0.0 GPU-h" in out
    plan.write_text("tab\tanimal\tsession\tranges\tproj_gpuh\nnope\tm1\ts1\tr\t1\n")
    r = subprocess.run([sys.executable, "-m", "tailcycle_run", "run", "--plan", str(plan),
                        "--total-cap", "10"], capture_output=True, text=True, timeout=60,
                       env=env, cwd=ROOT)
    assert r.returncode != 0 and "not in the config" in r.stderr


def test_remux_staged_submit_and_wait(tmp_path, monkeypatch):
    """_remux_staged with the mount and ssh faked: the bsub carries the config
    CPU queue, project and log dir; success = REMUX_OK; a failed log raises."""
    cr = _load("tailcycle_run/cluster/cluster_run.py")
    monkeypatch.setattr(cr, "LOCAL_CIFS", tmp_path)
    monkeypatch.setattr(cr, "_rsync_up", lambda *a, **k: None)
    monkeypatch.setattr(cr, "CPU_QUEUE", "cpuq")
    monkeypatch.setattr(cr, "DEFAULT_PROJECT", "projx")
    monkeypatch.setattr(cr, "CLUSTER_LOG_DIR", "~/joblogs")
    monkeypatch.setattr(cr.time, "sleep", lambda s: None)
    staged = tmp_path / "staged"
    staged.mkdir()
    sent = []

    def fake_ssh(cmd, timeout=60.0, check=True):
        sent.append(cmd)
        (staged / "REMUX_OK").touch()
        return "Job <42> is submitted to queue <cpuq>."
    monkeypatch.setattr(cr, "_ssh", fake_ssh)
    cr._remux_staged("a__s", staged, n_ts=3)
    cmd = sent[0]
    assert "-q cpuq " in cmd and "-P projx " in cmd
    assert "-o ~/joblogs/remux_a__s_%J.out" in cmd
    assert f"{cr.CF_REMOTE}/remux_ts.sh {cr.CF_REMOTE}/tailcycle_videos/a__s 4" in cmd
    assert (tmp_path / "joblogs").is_dir()

    def fake_ssh_fail(cmd, timeout=60.0, check=True):
        (tmp_path / "joblogs" / "remux_a__s_43.out").write_text("x: MISMATCH\nREMUX_HAD_FAILURES\n")
        return "Job <43> is submitted to queue <cpuq>."
    monkeypatch.setattr(cr, "_ssh", fake_ssh_fail)
    with pytest.raises(SystemExit, match="rewrap job 43 failed"):
        cr._remux_staged("a__s", staged, n_ts=3)
    monkeypatch.setattr(cr, "CLUSTER_LOG_DIR", "/abs/logs")
    with pytest.raises(SystemExit, match="home-relative"):
        cr._remux_staged("a__s", staged, n_ts=3)
