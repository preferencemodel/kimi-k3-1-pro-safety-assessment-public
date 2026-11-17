"""Folder layout of the released logs: <eval>/<model slug>/<token config>[/<variant>]/<file>.eval.

The Hugging Face datasets keep the logs under final/ in this layout. The evaluation-awareness analysis reads it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path, PurePosixPath

_LIB = str(Path(__file__).resolve().parent)
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
import config  # noqa: E402

EVAL_IDS = set(config.load()["evals"])
SCRATCH = {"_smoke", "_superseded", "_invalid"}
# Inspect task name prefix -> eval id, for tasks whose name is not the eval id.
TASK_PREFIXES = [("agentharm", "agentharm"), ("agentic_misalignment", "agentic_misalignment"), ("fortress", "fortress"),
                 ("impossiblebench_lcb", "impossiblebench_lcb"), ("lcb_", "impossiblebench_lcb"),
                 ("impossiblebench_swebench", "impossiblebench_swebench"), ("swebench_", "impossiblebench_swebench"),
                 ("wmdp", "wmdp"), ("lab_bench", "lab_bench"), ("evilgenie", "evilgenie"), ("hvtb", "hvtb"),
                 ("null_blue_team", "linuxarena"), ("honest_attack", "linuxarena"), ("ipi_coding_agent", "codeipi"),
                 ("xstest", "xstest")]


class Unresolved(Exception):
    pass


def slug(model: str) -> str:
    """Folder name of a model: the config target's slug, else derived from the model name."""
    t = config.target_for_model(model) if model and "/" in model else None
    return t["slug"] if t else re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower()).strip("-")


def task_eval(task: str) -> str | None:
    """Eval id for an Inspect task name, or None."""
    t = task.split("/")[-1].lower().replace("-", "_")
    if t in EVAL_IDS:
        return t
    for pre, ev in TASK_PREFIXES:
        if t.startswith(pre):
            return ev
    return None


def target(rel: PurePosixPath, task: str = "", model: str = "", status: str = "success", dir_eval=None):
    """The layout path of a log (released logs are already in the layout)."""
    if not rel.parts or rel.parts[0] not in EVAL_IDS:
        raise Unresolved(f"not a layout path: {rel}")
    return rel, "already in layout"
