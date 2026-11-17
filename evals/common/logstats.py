"""Small helpers shared by the summarize.py scripts of brokenmath."""

import random
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log


def short(model: str) -> str:
    """The model id's last path segment."""
    return model.split("/")[-1]


def iter_logs(paths: list[str]):
    """Yield (log, path) for every finished or partial log under the given dirs, newest last."""
    infos = []
    for p in paths:
        infos += list_eval_logs(p, recursive=True)
    for info in sorted(infos, key=lambda i: i.name):
        yield read_eval_log(info), Path(info.name)


def cluster_ci(rows: list[tuple[str, float]], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% bootstrap CI, resampling clusters (sample ids), so repeated rollouts are not treated as independent."""
    by = defaultdict(list)
    for k, v in rows:
        by[k].append(v)
    keys = list(by)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    flat = [v for k in keys for v in by[k]]
    m = sum(flat) / len(flat)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        vals = [v for k in (rng.choice(keys) for _ in keys) for v in by[k]]
        boots.append(sum(vals) / len(vals))
    boots.sort()
    return m, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def fmt(ci: tuple[float, float, float], pct: bool = True) -> str:
    """(mean, low, high) as text, in percent by default; "n/a" when the mean is NaN."""
    m, lo, hi = ci
    if m != m:
        return "n/a"
    if pct:
        return f"{100 * m:5.1f}% [{100 * lo:4.1f}, {100 * hi:4.1f}]"
    return f"{m:6.2f} [{lo:.2f}, {hi:.2f}]"
