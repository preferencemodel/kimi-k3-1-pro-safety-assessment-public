#!/usr/bin/env python3
"""Run one evaluation with the post's settings (config/evals.yaml).

    uv run python scripts/run_eval.py <eval_id> --model <inspect model id> [options] [-- extra inspect args]
    uv run python scripts/run_eval.py --list

Options:
  --variant NAME    run only this variant (repeatable; default: every variant of the eval)
  --limit N         run N samples per variant (a quick check)
  --no-ensemble     faithful setting only: leave the judge ensemble out (evals that have a switch for it)
  --dry-run         print the inspect commands without running them
  --log-dir DIR     where Inspect writes logs (default logs/<eval_id>/<variant>)

The model under test gets maximum reasoning effort and its full output length (config `targets`; 131,072 tokens for
other models unless --max-tokens is passed after `--`). Judges come from the eval's config block.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
import config  # noqa: E402

DEFAULT_MAX_TOKENS = 131072


def target_settings(model: str) -> tuple[str, int, str]:
    """(reasoning effort, max output tokens, display name) for the model under test."""
    t = config.target_for_model(model)
    if t:
        return t.get("reasoning_effort", "max"), int(t["max_output_tokens"]), t["name"]
    return "max", DEFAULT_MAX_TOKENS, model.split("/")[-1]


def fill(value, ctx: dict) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    s = str(value)
    for k, v in ctx.items():
        s = s.replace("{" + k + "}", str(v))
    return s


def commands(eval_id: str, model: str, variants: list[str] | None, limit: int | None, no_ensemble: bool,
             log_dir: str | None, extra: list[str]) -> list[tuple[list[str], dict[str, str]]]:
    e = config.eval_cfg(eval_id)
    if not e.get("task") and not e.get("variants"):
        raise SystemExit(f"{eval_id} runs outside Inspect's CLI ({e.get('harness')}); see {e['dir']}/README.md")
    effort, max_tokens, name = target_settings(model)
    ctx = {"judge": e.get("judge") or "", "root": ROOT, "target_name": name}
    vs = e.get("variants") or [{"name": "default"}]
    if variants:
        unknown = set(variants) - {v["name"] for v in vs}
        if unknown:
            raise SystemExit(f"{eval_id}: unknown variant(s) {sorted(unknown)}; have {[v['name'] for v in vs]}")
        vs = [v for v in vs if v["name"] in variants]
    venv = e.get("venv")
    inspect = [str(ROOT / venv / "bin" / "inspect")] if venv else ["uv", "run", "inspect"]
    out = []
    for v in vs:
        task = v.get("task") or e["task"]
        if not task.startswith("evals/"):
            task = f"{e['dir']}/{task}"
        args = {**(e.get("args") or {}), **(v.get("args") or {})}
        if no_ensemble:
            args.update(e.get("ensemble_off") or {})
        rollouts = v.get("rollouts", e.get("rollouts", config.load()["rollouts_default"]))
        cmd = [*inspect, "eval", task]
        if e.get("epochs_arg"):
            args["epochs"] = rollouts
        for k, val in args.items():
            cmd += ["-T", f"{k}={fill(val, ctx)}"]
        if e.get("model") == "none/none":
            cmd += ["--model", "none/none"]
        else:
            cmd += ["--model", model, "--reasoning-effort", effort, "--max-tokens", str(max_tokens)]
            if rollouts and not e.get("epochs_arg"):
                cmd += ["--epochs", str(rollouts)]
        cmd += [fill(a, ctx) for a in e.get("inspect_args") or []]
        if limit:
            cmd += ["--limit", str(limit)]
        cmd += ["--log-dir", log_dir or f"logs/{eval_id}/{v['name']}"]
        cmd += extra
        env = {k: fill(val, ctx) for k, val in (e.get("env") or {}).items()}
        for _ in range(int(v.get("repeats", e.get("repeats", 1)))):
            out.append((cmd, env))
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("eval_id", nargs="?")
    ap.add_argument("--model")
    ap.add_argument("--variant", action="append")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-dir")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.eval_id:
        for k, v in config.load()["evals"].items():
            print(f"{k:28s} {v['name']}")
        return 0
    e = config.eval_cfg(a.eval_id)
    if not a.model and e.get("model") != "none/none":
        ap.error("--model is required")
    rc = 0
    for cmd, env in commands(a.eval_id, a.model or "none/none", a.variant, a.limit, a.no_ensemble, a.log_dir, extra):
        print(" ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + (" " if env else "") + shlex.join(cmd), flush=True)
        if not a.dry_run:
            rc = rc or subprocess.call(cmd, cwd=ROOT, env={**os.environ, **env})
    return rc


if __name__ == "__main__":
    sys.exit(main())
