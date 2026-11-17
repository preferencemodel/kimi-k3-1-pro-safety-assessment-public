"""Equivalence of migrated panel scorers on existing logs (no model calls).


Logs are read from EQUIV_LOGS (a directory laid out like logs/; default <repo>/logs). Tests that find no logs
still run their constructed-record checks and report the log check as skipped.

    from evals.common.equivalence import Spec, check_logs, log_files
    rep = check_logs(Spec("xstest_panel", panel_value), log_files(["xstest", "xstest_safe"]))
    assert rep.ok, rep.summary()
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Spec:
    """How to check one scorer: its value function, how to read the context of a log that stored none, and which
    samples to skip."""

    scorer: str                                             # registered scorer name in the log
    value: Callable[[dict, dict], Any]
    legacy_context: Callable[[Any, Any], dict] = lambda score, sample: {}  # context from a log without panel_context
    unresolved: Literal["unscored", "noanswer", "value"] = "unscored"     # what the scorer does when unresolved
    skip: Callable[[Any, Any], str | None] = lambda score, sample: None    # reason to leave a sample out (gates)
    project: Callable[[Any], Any] = lambda v: v             # compare only this part of the value (e.g. one key)
    min_valid: int | None = None                            # judged-under min_valid when the stored rule is wrong
    owner_note: str = ""
    fields: list[str] | None = None                         # the scorer's fields, when a record may have no valid vote


@dataclass
class Report:
    """Counts of an equivalence check (records compared, matched, skipped by reason) and the mismatches found."""

    scorer: str
    logs: int = 0
    records: int = 0
    compared: int = 0
    matched: int = 0
    skipped: Counter = field(default_factory=Counter)
    mismatches: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.compared == self.matched

    def summary(self) -> str:
        s = f"{self.scorer}: {self.matched}/{self.compared} identical over {self.logs} logs"
        if self.skipped:
            s += f"; skipped {dict(self.skipped)}"
        if self.mismatches:
            s += f"; first mismatches {self.mismatches[:3]}"
        return s


def same(a: Any, b: Any) -> bool:
    """Deep equality with NaN == NaN, NOANSWER "N" == NaN, and floats to 1e-9."""
    def unres(x: Any) -> bool:
        return x is None or x == "N" or (isinstance(x, float) and math.isnan(x))
    if unres(a) and unres(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)
    return a == b


def expected_value(spec: Spec, meta: dict, score: Any, sample: Any, target: str | None) -> tuple[str, Any]:
    """('ok', value) from the stored votes, or (skip reason, None)."""
    from evals.common.judge_panel import panel_outcome_from_meta

    if spec.min_valid is not None:
        meta = {**meta, "panel_rule": {"min_valid": spec.min_valid}}
    out = panel_outcome_from_meta(meta, default_target=target, original=True)
    if out is None:
        return "no_panel_record", None
    verdicts = _verdicts(spec, out)
    if not out.resolved and spec.unresolved != "value":
        return "ok", float("nan")
    ctx = out.context or spec.legacy_context(score, sample)
    return "ok", spec.value(verdicts, ctx)


def _verdicts(spec: Spec, out: Any) -> dict:
    """The outcome's verdicts; a record with no valid vote at all reads as one unnamed field, so it is mapped to the
    scorer's own fields (every one unresolved)."""
    from evals.common.judge_panel import SINGLE
    if spec.fields and list(out.kinds) == [SINGLE] and not any(v.valid for v in out.votes):
        return {f: None for f in spec.fields}
    return out.verdicts


def check_samples(spec: Spec, samples: Iterable[Any], target: str | None, rep: Report | None = None,
                  where: str = "") -> Report:
    """Recompute `spec.scorer`'s value from each sample's stored votes and compare it with the stored value; adds to
    `rep` (a new Report by default) and returns it."""
    rep = rep or Report(spec.scorer)
    for s in samples:
        sc = (getattr(s, "scores", None) or {}).get(spec.scorer)
        if sc is None:
            continue
        meta = getattr(sc, "metadata", None) or {}
        why = spec.skip(sc, s)
        if why:
            rep.skipped[why] += 1
            continue
        status, want = expected_value(spec, meta, sc, s, target)
        if status != "ok":
            rep.skipped[status] += 1
            continue
        rep.records += 1
        rep.compared += 1
        if same(spec.project(sc.value), spec.project(want)):
            rep.matched += 1
        else:
            rep.mismatches.append({"where": where, "id": getattr(s, "id", None), "epoch": getattr(s, "epoch", None),
                                   "logged": sc.value, "rescored": want})
    return rep


def log_files(dirs: list[str], root: str | Path | None = None) -> list[Path]:
    """Every .eval file under `dirs` of the logs root (`root`, else EQUIV_LOGS, else logs/), sorted per dir."""
    base = Path(root or os.environ.get("EQUIV_LOGS") or ROOT / "logs")
    out: list[Path] = []
    for d in dirs:
        p = base / d
        if p.is_dir():
            out += sorted(p.rglob("*.eval"))
    return out


def check_logs(spec: Spec, files: list[Path]) -> Report:
    """Read each log's samples (read-only) and compare. Unreadable logs are counted as skipped."""
    from inspect_ai.log import read_eval_log, read_eval_log_samples

    rep = Report(spec.scorer)
    for f in files:
        try:
            target = read_eval_log(str(f), header_only=True).eval.model
            samples = list(read_eval_log_samples(str(f), all_samples_required=False))
        except Exception as e:  # noqa: BLE001
            rep.skipped[f"unreadable_log:{type(e).__name__}"] += 1
            continue
        rep.logs += 1
        check_samples(spec, samples, target, rep, where=f.name)
    return rep


def roundtrip_samples(spec: Spec, samples: Iterable[Any], target: str | None, limit: int = 200) -> tuple[int, list]:
    """Old-shape record -> standard metadata (PanelOutcome.meta) -> value: must equal the value from the old record."""
    from evals.common.judge_panel import panel_outcome_from_meta

    n, bad = 0, []
    for s in samples:
        sc = (getattr(s, "scores", None) or {}).get(spec.scorer)
        if sc is None or n >= limit:
            continue
        meta = sc.metadata or {}
        if spec.min_valid is not None:
            meta = {**meta, "panel_rule": {"min_valid": spec.min_valid}}
        old = panel_outcome_from_meta(meta, default_target=target, original=True)
        if old is None:
            continue
        ctx = old.context or spec.legacy_context(sc, s)
        std = {**old.meta, "panel_context": ctx}
        new = panel_outcome_from_meta(std, default_target=target, original=True)
        n += 1
        a = old.verdicts if not old.resolved else spec.value(_verdicts(spec, old), ctx)
        b = new.verdicts if not new.resolved else spec.value(_verdicts(spec, new), ctx)
        if new.verdicts != old.verdicts or not same(spec.project(a), spec.project(b)):
            bad.append((getattr(s, "id", None), old.verdicts, new.verdicts))
    return n, bad


