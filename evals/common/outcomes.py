"""Answer outcomes: keep provider-filtered, truncated, empty and errored samples out of every metric (the outcome rule).


    ok               the model produced a visible answer (a refusal in the model's own words is `ok`)
    provider_filter  a provider safety system stopped it: stop_reason=content_filter, or a policy/filter API error
    truncated        stopped by the token cap (stop_reason max_tokens / model_length), visible answer empty or cut off
    empty            stopped normally but no visible text and no tool calls (e.g. all tokens went to reasoning)
    error            no usable output (sample or model error)
    limit            (episode mode only) the episode was ended by a message/token/time/working limit
    harness_intervention  we terminated a hung sandbox command in the episode (scripts/data/harness_interventions.json);
                     assigned at summarize time (`intervened`), always excluded, whatever an eval counts

Non-`ok` samples are excluded from the eval's metrics and reported as counts, one row per reason. Judges follow the
same rule: a judge output that is filtered, truncated or empty is an invalid vote and is counted
(`judge_vote_outcome`, used by evals/common/judge_panel.py).


    1. from evals.common.outcomes import with_outcome, outcome_scorer
    2. wrap the eval's scorer: scorer=[with_outcome(orig_scorer(...)), outcome_scorer()]  (mode="episode" for agents);
       for an inspect_evals task, in a local task file: task_with(upstream_task(...), scorer=[...]) (inspect_ai.task_with)
    3. excluded samples become Score.unscored() (NaN) with metadata["outcome"]; Inspect drops NaN from every metric and
       reducer, and `outcome_scorer` reports the share per reason as metrics (ok, provider_filter, truncated, ...).
       With epochs > 1 the wrapped metrics are per rollout (each computed over the valid rollouts as at epochs=1;
       `per_rollout_metrics` below), not over Inspect's epoch-reduced scores

"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

Mode = Literal["final", "episode"]

OK = "ok"
PROVIDER_FILTER = "provider_filter"
TRUNCATED = "truncated"
EMPTY = "empty"
ERROR = "error"
LIMIT = "limit"
HARNESS_INTERVENTION = "harness_intervention"  # a hung sandbox command we terminated (scripts/data/harness_interventions.json)
OUTCOMES = (OK, PROVIDER_FILTER, TRUNCATED, EMPTY, ERROR, LIMIT)  # what a run classifies (outcome_scorer, tasks)
SUMMARY_OUTCOMES = (*OUTCOMES, HARNESS_INTERVENTION)  # plus what is assigned at summarize time only
EXCLUDED_DEFAULT = frozenset(SUMMARY_OUTCOMES) - {OK}
ALWAYS_EXCLUDED = frozenset({HARNESS_INTERVENTION})  # excluded whatever an eval counts (limit hits may be counted)
INTERVENTIONS_FILE = Path(__file__).resolve().parents[2] / "scripts" / "data" / "harness_interventions.json"
_INTERVENTIONS: dict[tuple[str, str, int], dict[str, Any]] | None = None

TRUNCATION_STOPS = ("max_tokens", "model_length")
# API errors that are a provider's safety system declining the request (they arrive as exceptions or ModelOutput.error).
FILTER_ERROR_RE = re.compile(
    r"content[_ ]?filter|content[_ ]policy|policy[_ ]violation|usage polic|flagged as potentially|"
    r"invalid_prompt|safety (?:system|filter)|responsible ai polic|output blocked|request was blocked", re.I)
_THINK_RE = re.compile(r"<think(?:ing)?>.*?(?:</think(?:ing)?>|\Z)", re.S | re.I)


@dataclass
class Outcome:
    """How one answer or episode ended: the outcome class (ok, provider_filter, truncated, empty, error, limit) and
    the evidence behind it."""

    outcome: str
    detail: str = ""                  # stop_reason / limit type / error excerpt
    stop_reason: str | None = None
    visible_chars: int = 0
    filter_events: int = 0            # episode mode: content_filter stops anywhere in the episode (log level only)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome == OK

    def as_metadata(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(d.pop("extra"))
        return {k: v for k, v in d.items() if v not in (None, "", {}, 0) or k == "outcome"}


# ---------------------------------------------------------------- harness interventions
def interventions(reload: bool = False) -> dict[tuple[str, str, int], dict[str, Any]]:
    """(log task id, sample id, epoch) -> entry, from scripts/data/harness_interventions.json (cached)."""
    global _INTERVENTIONS
    if _INTERVENTIONS is None or reload:
        data = json.loads(INTERVENTIONS_FILE.read_text()) if INTERVENTIONS_FILE.exists() else {}
        _INTERVENTIONS = {(str(e["task_id"]), str(e["sample_id"]), int(e["epoch"])): e
                          for e in data.get("interventions", [])}
    return _INTERVENTIONS


def sample_log_keys(sample: Any, log: Any = None) -> tuple[set[str], int]:
    """Log task ids a sample may be listed under, and its epoch in that log: and, for a sample merged from another log, its source task id and epoch; else the
    log's own task id."""
    md = getattr(sample, "metadata", None) or {}
    prov = md.get("consolidation") if isinstance(md, dict) else None
    tids = {getattr(sample, "log_task_id", None), getattr(getattr(log, "eval", None), "task_id", None)}
    epoch = getattr(sample, "epoch", None)
    if isinstance(prov, dict):
        tids.add(prov.get("source_task_id"))
        epoch = prov.get("original_epoch") or epoch
    return {str(t) for t in tids if t}, int(epoch or 0)


def intervention_for(task_ids: Iterable[str], sample_id: Any, epoch: int) -> dict[str, Any] | None:
    """The harness-intervention entry listing this sample in one of these logs, or None."""
    table = interventions()
    return next((table[k] for t in task_ids if (k := (str(t), str(sample_id), int(epoch))) in table), None)


def intervened(sample: Any, log: Any = None) -> Outcome | None:
    """Outcome harness_intervention for a listed sample (counted, never scored), else None."""
    tids, epoch = sample_log_keys(sample, log)
    hit = intervention_for(tids, getattr(sample, "id", None), epoch)
    if hit is None:
        return None
    return Outcome(HARNESS_INTERVENTION, detail=str(hit.get("note") or hit.get("reason") or ""),
                   extra={"intervention_time": hit.get("time")})


def outcome_of(sample: Any, mode: Mode | Literal["auto"] = "auto", target_model: str | None = None,
               log: Any = None) -> Outcome:
    """classify_sample, unless the sample is a listed harness intervention."""
    return intervened(sample, log) or classify_sample(sample, mode, target_model)


# ---------------------------------------------------------------- pure classification
def visible_text(text: str | None) -> str:
    """Answer text with any inline <think>...</think> reasoning removed (an unclosed <think> swallows the rest)."""
    return _THINK_RE.sub("", text or "").strip()


def _stop_reason(output: Any) -> str | None:
    """output.stop_reason, or None: Inspect's ModelOutput.stop_reason raises IndexError when there are no choices
    (an agent that ended without a model call, a scripted solver, an errored generate)."""
    try:
        return getattr(output, "stop_reason", None)
    except (IndexError, AttributeError):
        return None


def _err_outcome(error: str) -> Outcome:
    """A sample error as an Outcome: provider_filter when the error text is a content-filter stop, else error."""
    kind = PROVIDER_FILTER if FILTER_ERROR_RE.search(error) else ERROR
    return Outcome(kind, detail=error[-300:])


def classify_output(output: Any, error: str | None = None) -> Outcome:
    """Classify one final ModelOutput (single-turn evals, and judge outputs).

    `error` is a sample/model error message if the output never arrived. A refusal in the model's own words is `ok`;
    Anthropic's filter stop is `provider_filter` even though Inspect fills in "Sorry, but I am unable to help ...".
    """
    if error:
        return _err_outcome(str(error))
    if output is None:
        return Outcome(ERROR, detail="no output")
    choices = getattr(output, "choices", None) or []
    stop = _stop_reason(output)
    details = getattr(output, "stop_details", None) or (getattr(choices[0], "stop_details", None) if choices else None)
    out_err = getattr(output, "error", None)
    text = visible_text(getattr(output, "completion", "") or "")
    msg = getattr(output, "message", None) if choices else None
    tool_calls = bool(getattr(msg, "tool_calls", None))
    common = dict(stop_reason=stop, visible_chars=len(text))
    if stop == "content_filter" or getattr(details, "type", None) == "content_filter":
        return Outcome(PROVIDER_FILTER, detail=str(getattr(details, "category", "") or stop or ""), **common)
    if out_err:
        o = _err_outcome(str(out_err))
        o.stop_reason, o.visible_chars = stop, len(text)
        return o
    if stop in TRUNCATION_STOPS:
        return Outcome(TRUNCATED, detail=stop, **common)
    if not text and not tool_calls:
        return Outcome(EMPTY, detail=stop or "", **common)
    return Outcome(OK, **common)


def judge_vote_outcome(output: Any, error: str | None = None) -> str:
    """Outcome of one judge call; anything but `ok` is an invalid vote (count it, never treat it as a verdict)."""
    return classify_output(output, error).outcome


def _limit_from_live() -> str | None:
    """Limit that ended the current sample, if any (scorer time; best effort)."""
    try:
        from inspect_ai.util._limit import sample_limits
        lims = sample_limits()
    except Exception:
        return None
    for name in ("message", "token", "turn", "time", "working", "cost"):
        lim = getattr(lims, name, None)
        try:
            # Inside an agent (e.g. react's attempts scoring a submission mid-episode) the message limit's `usage`
            # raises NotImplementedError; no sample limit has ended the episode at that point.
            cap, usage = getattr(lim, "limit", None), getattr(lim, "usage", None)
        except Exception:
            continue
        if cap is not None and usage is not None and usage >= cap:
            return name
    return None


def classify_episode(messages: Iterable[Any], output: Any, limit: str | None = None, error: str | None = None,
                     filter_events: int = 0) -> Outcome:
    """Episode-level outcome for agentic evals.

    The episode is non-ok if it ended in an error, a limit, a provider filter stop or a token-cap stop on the last
    model turn, or if the model never produced any text or tool call. A filter stop earlier in an episode that then
    continued is recorded in `filter_events` but does not exclude the episode.
    """
    if error:
        o = _err_outcome(str(error))
        o.filter_events = filter_events
        return o
    last = classify_output(output) if output is not None and getattr(output, "choices", None) else None
    if last is not None and last.outcome in (PROVIDER_FILTER, TRUNCATED, ERROR):
        last.filter_events = filter_events
        return last
    if limit:
        return Outcome(LIMIT, detail=str(limit), filter_events=filter_events)
    acted = any(getattr(m, "role", None) == "assistant" and not (getattr(m, "metadata", None) or {}).get("seeded")
                and (visible_text(getattr(m, "text", "")) or getattr(m, "tool_calls", None)) for m in messages)
    if not acted:
        return Outcome(EMPTY, detail="no assistant text or tool call in episode", filter_events=filter_events)
    return Outcome(OK, stop_reason=_stop_reason(output), filter_events=filter_events)


# Sample metadata key in which an agent loop that catches its own limits records which one ended the episode
# (e.g. HVTB's solver: apply_limits(..., catch_errors=True) -> metadata["agent_limit"] = "time" | "message"). Such
# limits never reach Inspect's sample limit, so episode mode reads this key too.
AGENT_LIMIT_KEY = "agent_limit"


def _limit_from_metadata(metadata: Any) -> str | None:
    """The agent limit a scaffold recorded in sample metadata (AGENT_LIMIT_KEY), or None."""
    v = (metadata or {}).get(AGENT_LIMIT_KEY) if isinstance(metadata, dict) else None
    return str(v) if v else None


def classify_state(state: Any, mode: Mode = "final") -> Outcome:
    """Outcome of a running sample (scorer time)."""
    if mode == "episode":
        return classify_episode(state.messages, state.output,
                                limit=_limit_from_live() or _limit_from_metadata(getattr(state, "metadata", None)))
    return classify_output(state.output)


def _sample_error(sample: Any) -> str | None:
    """The sample's error message, or None if it did not error."""
    err = getattr(sample, "error", None)
    return None if err is None else str(getattr(err, "message", err))


def _model_events(sample: Any) -> list[Any]:
    """Every model event of the sample, in order."""
    return [e for e in (getattr(sample, "events", None) or []) if getattr(e, "event", None) == "model"]


def _has_choices(output: Any) -> bool:
    return bool(getattr(output, "choices", None))


def _filter_stop_without_output(sample: Any, target_model: str | None) -> Outcome | None:
    """`provider_filter` when the sample ended with no output because the target's last model call was filter-stopped.

    An agent scaffold (e.g. SHADE-Arena's) appends no message when the target's call is stopped by its provider filter,
    so the sample's output is empty while the last target ModelEvent carries the stop. The event is classified with
    `classify_output` (stop_reason / stop_details content_filter, or a policy/filter API error); anything else is
    None and the caller keeps its own outcome. The target is `target_model`, else the model named on the empty output.
    """
    target = target_model or getattr(getattr(sample, "output", None), "model", None)
    if not target:
        return None
    last = next((e for e in reversed(_model_events(sample)) if _same_model(getattr(e, "model", None), target)), None)
    if last is None:
        return None
    err = getattr(last, "error", None)
    if not err and not _has_choices(getattr(last, "output", None)):
        return None
    oc = classify_output(getattr(last, "output", None), err)
    if oc.outcome != PROVIDER_FILTER:
        return None
    oc.detail = f"no output; last target call: {oc.detail}".rstrip(": ")
    return oc


LastTurnFilter = Literal["rule", "as_acted"]
LAST_TURN_FILTER: LastTurnFilter = "rule"


def set_last_turn_filter(mode: LastTurnFilter) -> None:
    """Set the episode-mode last-turn filter rule ("rule" or "as_acted") for this process."""
    global LAST_TURN_FILTER
    if mode not in ("rule", "as_acted"):
        raise ValueError(f"last_turn_filter must be 'rule' or 'as_acted', not {mode!r}")
    LAST_TURN_FILTER = mode


def _solver_model_events(sample: Any) -> list[Any]:
    """Model events before scoring began (judges and monitors run in scorer spans; a judge may be the target model)."""
    out = []
    for e in getattr(sample, "events", None) or []:
        ev = getattr(e, "event", None)
        if ev == "score" or (ev == "span_begin" and getattr(e, "type", None) in ("scorer", "scorers")):
            break
        if ev == "model":
            out.append(e)
    return out


def last_target_call_filtered(sample: Any, target_model: str | None) -> Outcome | None:
    """provider_filter Outcome when the target's last model call before scoring was stopped by its provider filter."""
    target = target_model or getattr(getattr(sample, "output", None), "model", None)
    if not target:
        return None
    last = next((e for e in reversed(_solver_model_events(sample)) if _same_model(getattr(e, "model", None), target)),
                None)
    if last is None:
        return None
    err = getattr(last, "error", None)
    if not err and not _has_choices(getattr(last, "output", None)):
        return None
    oc = classify_output(getattr(last, "output", None), err)
    if oc.outcome != PROVIDER_FILTER:
        return None
    oc.detail = f"last target call filter-stopped: {oc.detail}".rstrip(": ")
    return oc


def outcome_events(sample: Any) -> list[Any]:
    """The model events `classify_sample` reads, for readers that drop a sample's events to save memory
    empty output, the last model event of each model."""
    keep: set[int] = set()
    evs = _model_events(sample)
    pos = {id(e): i for i, e in enumerate(evs)}
    last = {getattr(e, "model", None): pos[id(e)] for e in _solver_model_events(sample) if id(e) in pos}
    keep.update(last.values())
    if not _has_choices(getattr(sample, "output", None)):
        keep.update({getattr(e, "model", None): i for i, e in enumerate(evs)}.values())
    return [evs[i] for i in sorted(keep)]


def target_output(sample: Any, target_model: str | None) -> Any:
    """The target's own final output: `sample.output`, unless that holds another model's call (a scorer that runs a
    judge through the task state leaves the last judge call there, e.g. a GLM-5.3 panel vote), in which case the output
    of the target's last model event before scoring (else its last model event at all). With no `target_model`, or an
    output that names no model, `sample.output` as it is."""
    out = getattr(sample, "output", None)
    om = getattr(out, "model", None)
    if not target_model or not om or _same_model(om, target_model):
        return out
    mine = lambda evs: [e for e in evs if _same_model(getattr(e, "model", None), target_model)]  # noqa: E731
    evs = mine(_solver_model_events(sample)) or mine(_model_events(sample))
    return getattr(evs[-1], "output", None) if evs else out


def classify_sample(sample: Any, mode: Mode | Literal["auto"] = "auto", target_model: str | None = None,
                    last_turn_filter: LastTurnFilter | None = None) -> Outcome:
    """Outcome of a logged EvalSample. mode=auto: `episode` if the sample has tool messages, else `final`.

    In final mode the output read is the target's own (`target_output`: not a judge call a scorer left on the sample;
    episode mode reads the transcript). A sample with no output and no sample error whose last target call was stopped by the provider filter is
    `provider_filter` in either mode (`_filter_stop_without_output`); `target_model` names the target (default: the
    model on the sample's output). In episode mode, an episode that would be ok but whose last target call before
    scoring was filter-stopped is provider_filter under `last_turn_filter` "rule" (default: module LAST_TURN_FILTER)
    and ok under "as_acted"."""
    msgs = list(getattr(sample, "messages", None) or [])
    if mode == "auto":
        mode = "episode" if any(getattr(m, "role", None) == "tool" for m in msgs) else "final"
    err = _sample_error(sample)
    if mode == "final":
        output = target_output(sample, target_model)
        if err:
            return classify_output(None, err)
        if not _has_choices(output):
            return _filter_stop_without_output(sample, target_model) or Outcome(ERROR, detail="no output")
        return classify_output(output)
    no_output = not _has_choices(getattr(sample, "output", None))
    filt = sum(1 for e in _model_events(sample)
               if (target_model is None or _same_model(e.model, target_model))
               and getattr(e.output, "stop_reason", None) == "content_filter")
    if no_output and not err:
        fo = _filter_stop_without_output(sample, target_model)
        if fo is not None:
            fo.filter_events = filt
            return fo
    lim = getattr(sample, "limit", None)
    lim_type = (getattr(lim, "type", None) if lim else None) or _limit_from_metadata(getattr(sample, "metadata", None))
    oc = classify_episode(msgs, sample.output, limit=lim_type, error=err, filter_events=filt)
    if oc.outcome == OK and (last_turn_filter or LAST_TURN_FILTER) == "rule":
        lf = last_target_call_filtered(sample, target_model)
        if lf is not None:
            lf.filter_events = filt
            return lf
    return oc


def _same_model(a: str | None, b: str | None) -> bool:
    """Whether two model ids name the same model (equal, or the same last path segment ignoring case)."""
    if not a or not b:
        return False
    return a == b or a.rstrip("/").split("/")[-1].lower() == b.rstrip("/").split("/")[-1].lower()


def _one_rollout_reducer(reducer: Any = None) -> Any:
    """The task's reducer (a name or an instance) to apply to one rollout's score; the mean reducer by default."""
    if reducer is None or isinstance(reducer, str):
        from inspect_ai.scorer._reducer.registry import create_reducers
        return (create_reducers(reducer) or [None])[0] if reducer else _mean_reducer()
    return reducer


def _mean_reducer() -> Any:
    from inspect_ai.scorer import mean_score
    return mean_score()


def _as_epochs1(scores: list, reducer: Any) -> list:
    """Each SampleScore with its value as `reducer` gives it at epochs=1; the rollout's metadata is kept."""
    out = []
    for s in scores:
        r = reducer([s.score])
        if r.metadata is None and s.score.metadata is not None:
            r = r.model_copy(update={"metadata": s.score.metadata})
        out.append(s.model_copy(update={"score": r}))
    return out


def _sample_clustered_stderr(to_float: Any, reducer: Any) -> Any:
    """A stderr metric over valid rollouts, clustered by sample id (equal to the plain stderr at epochs=1)."""
    from inspect_ai.scorer import SampleScore, value_to_float
    from inspect_ai.scorer._metrics.std import _clustered_stderr

    tf = to_float or value_to_float()

    def fn(scores):
        groups: dict[str, list[float]] = {}
        for s in _as_epochs1(scores, reducer):
            groups.setdefault(str(s.sample_id), []).append(tf(s.score.value))
        return float(_clustered_stderr(list(groups.values()))) if groups else 0.0
    fn.__annotations__ = {"scores": list[SampleScore], "return": float}
    return fn


def _unreduced_copy(m: Any, reducer: Any = None) -> Any:
    """The same metric (same registry name and params, so log names do not change) reading one score per rollout."""
    from inspect_ai._eval.task.results import call_metric
    from inspect_ai._util.registry import (registry_info, registry_params, registry_unqualified_name,
                                           set_registry_info, set_registry_params)
    from inspect_ai.scorer import SampleScore
    from inspect_ai.scorer._metric import METRIC_SCORES, metric_scores

    if metric_scores(m) != "auto":
        return m
    red = _one_rollout_reducer(reducer)
    try:
        params = registry_params(m)
    except ValueError:
        params = {}
    if registry_unqualified_name(m) == "stderr" and not params.get("cluster"):
        fn = _sample_clustered_stderr(params.get("to_float"), red)
    else:
        def fn(scores):
            return call_metric(m, _as_epochs1(scores, red))
        fn.__annotations__ = {"scores": list[SampleScore], "return": Any}
    info = registry_info(m)
    set_registry_info(fn, info.model_copy(update={"metadata": {**info.metadata, METRIC_SCORES: "unreduced"}}))
    set_registry_params(fn, params)
    return fn


def per_rollout_metrics(metrics: Any, reducer: Any = None) -> Any:
    """A scorer's metrics (list / dict / list with dicts) with every default-view metric made per-rollout (above).

    `reducer`: the task's epoch reducer (a ScoreReducer or its name; default mean), applied to each rollout alone."""
    if isinstance(metrics, dict):
        return {k: [_unreduced_copy(m, reducer) for m in ms] for k, ms in metrics.items()}
    return [{k: [_unreduced_copy(x, reducer) for x in ms] for k, ms in m.items()} if isinstance(m, dict)
            else _unreduced_copy(m, reducer) for m in metrics]


# ---------------------------------------------------------------- Inspect scorer wrapper + metrics
def with_outcome(inner: Any, mode: Mode = "final", exclude: Iterable[str] = EXCLUDED_DEFAULT,
                 score_excluded: bool = False, per_rollout: bool = True, rollout_reducer: Any = None) -> Any:
    """Wrap a scorer: excluded samples get Score.unscored() (NaN) with metadata["outcome"]; the rest score as before.

    Keeps the inner scorer's name and metric names, so log metric names do not change; Inspect drops NaN scores from
    metrics and epoch reducers and counts them in EvalScore.unscored_samples. `score_excluded=True` still runs the
    inner scorer on excluded samples (its value is kept in metadata["raw_value"]); by default it is skipped, so no
    judge is paid to grade a filter stop.

    `per_rollout=True` (default): the inner metrics read one score per rollout (`per_rollout_metrics`), so each is
    computed over the valid rollouts (samples x epochs minus excluded ones) as at epochs=1, and `unscored_samples`
    counts excluded rollouts. `rollout_reducer`: the task's own epoch reducer if it is not mean (e.g. "mode"), applied
    to each rollout alone. Pass per_rollout=False for a task whose result combines a sample's rollouts (at_least_1).
    """
    from inspect_ai._util.registry import registry_unqualified_name
    from inspect_ai.scorer import Score, scorer
    from inspect_ai.scorer._scorer import scorer_metrics

    excl = frozenset(exclude)
    bad = excl - set(SUMMARY_OUTCOMES)
    if bad:
        raise ValueError(f"unknown outcomes {sorted(bad)}; valid: {SUMMARY_OUTCOMES}")
    name = registry_unqualified_name(inner)
    metrics = scorer_metrics(inner)

    @scorer(metrics=per_rollout_metrics(metrics, rollout_reducer) if per_rollout else metrics, name=name)
    def _wrapped():
        async def score(state, target):
            oc = classify_state(state, mode)
            if oc.outcome in excl:
                meta = {"outcome": oc.outcome, "outcome_detail": oc.as_metadata()}
                if score_excluded:
                    raw = await inner(state, target)
                    meta["raw_value"] = None if raw is None else raw.value
                return Score.unscored(reason=_reason(oc.outcome), explanation=f"excluded: {oc.outcome} {oc.detail}",
                                      metadata=meta)
            s = await inner(state, target)
            if s is None:
                return s
            s.metadata = {**(s.metadata or {}), "outcome": oc.outcome}
            if oc.filter_events:
                s.metadata["filter_events"] = oc.filter_events
            return s
        return score
    return _wrapped()


def _reason(outcome: str) -> str:
    """Score.reason for an excluded outcome: Inspect's standard ScoreReason where one fits, our own label otherwise."""
    return {EMPTY: "no_response"}.get(outcome, outcome)


def outcome_scorer(mode: Mode = "final"):
    """Companion scorer: one-hot dict {outcome_<reason>: 0/1}; metrics = share of rollouts per reason.

    In the log these appear as scores outcome_ok, outcome_provider_filter, outcome_truncated, ... (metric `mean`,
    unreduced: rollouts with that outcome / scored rollouts, so a sample with fewer scored epochs is not over-weighted).
    """
    from inspect_ai.scorer import Score, mean, scorer

    @scorer(metrics={f"outcome_{k}": [_unreduced_copy(mean())] for k in OUTCOMES}, name="outcome")
    def _outcome():
        async def score(state, target):
            oc = classify_state(state, mode)
            return Score(value={f"outcome_{k}": int(k == oc.outcome) for k in OUTCOMES}, answer=oc.outcome,
                         explanation=oc.detail or None, metadata=oc.as_metadata())
        return score
    return _outcome()


# ---------------------------------------------------------------- log-level summary
def _input_key(e: Any) -> str:
    """A stable text key of a model event's input messages, to recognize repeated calls with the same input."""
    import json  # noqa: PLC0415
    inp = getattr(e, "input", None)
    try:
        return json.dumps([m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in (inp or [])],
                          sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return repr(inp)


def final_attempt_events(events: list[Any], target: str | None) -> list[Any]:
    """Model events with retried judge attempts removed, so each judge call counts once, by its last attempt.

    failed. Events without an error and the target's own events are kept as they are."""
    def is_judge(e: Any) -> bool:
        return getattr(e, "event", "model") == "model" and not _same_model(getattr(e, "model", None), target)

    if not any(getattr(e, "error", None) and is_judge(e) for e in events):
        return list(events)
    last: dict[tuple, int] = {}
    keys: dict[int, tuple] = {}
    for i, e in enumerate(events):
        if is_judge(e):
            keys[i] = (getattr(e, "model", None), _input_key(e))
            last[keys[i]] = i
    return [e for i, e in enumerate(events)
            if not (i in keys and getattr(e, "error", None) and last[keys[i]] != i)]


def _score_float(v: Any) -> float | None:
    """A score value as a float: bools and numbers as they are (NaN as None), C/I/P and yes/no/true/false labels
    mapped; anything else None."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return None if isinstance(v, float) and math.isnan(v) else float(v)
    if isinstance(v, str):
        m = {"C": 1.0, "I": 0.0, "P": 0.5, "yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0}
        if v in m or v.lower() in m:
            return m.get(v, m.get(v.lower()))
        try:
            return float(v)
        except ValueError:
            return None
    return None


def summarize_log(log: Any, mode: Mode | Literal["auto"] = "auto", exclude: Iterable[str] = EXCLUDED_DEFAULT,
                  judge_models: bool = True, per_sample: bool = False) -> dict[str, Any]:
    """Per-reason outcome counts for one EvalLog (read with samples), plus each scalar score's mean over ok samples.

    """
    excl = frozenset(exclude) | ALWAYS_EXCLUDED
    target = getattr(getattr(log, "eval", None), "model", None)
    counts: Counter[str] = Counter()
    sums: dict[str, list[float]] = {}
    judges: dict[str, Counter[str]] = {}
    samples = list(getattr(log, "samples", None) or [])
    oks = [outcome_of(s, mode, target, log) for s in samples]
    from evals.common.judge_panel import reaggregate_log_scores  # noqa: PLC0415 (judge_panel imports scripts/lib)
    scored = [i for i, oc in enumerate(oks) if oc.outcome not in excl]
    new_vals, panel = reaggregate_log_scores([samples[i] for i in scored], default_target=target)
    new_vals = {(scored[j], name): v for (j, name), v in new_vals.items()}
    rows: list[dict[str, Any]] = []
    for s_i, (s, oc) in enumerate(zip(samples, oks)):
        counts[oc.outcome] += 1
        if per_sample:
            rows.append({"outcome": oc.outcome, "scores": {name: new_vals.get((s_i, name), sc.value)
                                                           for name, sc in (s.scores or {}).items()}})
        if judge_models:  # each judge call counts once, by its last attempt (final_attempt_events)
            for e in final_attempt_events(_model_events(s), target):
                if _same_model(e.model, target):
                    continue
                jo = judge_vote_outcome(e.output, getattr(e, "error", None))
                judges.setdefault(e.model, Counter())[jo] += 1
        if oc.outcome in excl:
            continue
        for name, sc in (s.scores or {}).items():
            if name == "outcome" or name.startswith("outcome_"):
                continue
            value = new_vals.get((s_i, name), sc.value)
            vals = value.items() if isinstance(value, dict) else [(None, value)]
            for k, v in vals:
                f = _score_float(v)
                if f is not None:
                    sums.setdefault(name if k is None else f"{name}.{k}", []).append(f)
    n = sum(counts.values())
    extra = {"per_sample": rows} if per_sample else {}
    return {
        **extra,
        "n": n,
        "counts": {k: counts.get(k, 0) for k in SUMMARY_OUTCOMES},
        "n_valid": sum(c for k, c in counts.items() if k not in excl),
        "rows": outcome_rows(counts, n),
        "ok_means": {k: sum(v) / len(v) for k, v in sorted(sums.items()) if v},
        "ok_n": {k: len(v) for k, v in sorted(sums.items()) if v},
        "judge_invalid": {m: {k: c for k, c in jc.items() if k != OK} for m, jc in judges.items()
                          if any(k != OK for k in jc)},
        "judge_calls": {m: sum(jc.values()) for m, jc in judges.items()},
        "panel": panel,
        "panel_votes_dropped": sum(p["votes_dropped"] for p in panel.values()),
    }


def outcome_rows(counts: dict[str, int] | Counter, n: int | None = None) -> list[dict[str, Any]]:
    """One row per outcome reason (ok first): {"outcome", "n", "share"}. Zero rows are kept, so tables line up."""
    n = n if n is not None else sum(counts.values())
    return [{"outcome": k, "n": int(counts.get(k, 0)), "share": (counts.get(k, 0) / n) if n else 0.0}
            for k in SUMMARY_OUTCOMES]


def merge_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool several summaries (e.g. three --epochs 1 runs of MASK/FORTRESS): counts add; each ok mean is weighted by
    the number of values behind it (`ok_n`; n_valid for summaries made before it existed), so the pooled mean is the
    mean over all valid rollouts."""
    counts: Counter[str] = Counter()
    means: dict[str, tuple[float, int]] = {}
    judges: dict[str, Counter[str]] = {}
    panel: dict[str, dict] = {}
    for s in summaries:
        counts.update(s["counts"])
        for k, v in s["ok_means"].items():
            tot, w = means.get(k, (0.0, 0))
            n_k = s.get("ok_n", {}).get(k, s["n_valid"])
            means[k] = (tot + v * n_k, w + n_k)
        for m, jc in s.get("judge_invalid", {}).items():
            judges.setdefault(m, Counter()).update(jc)
        for name, st in s.get("panel", {}).items():
            acc = panel.setdefault(name, {"labels": {}})
            for k, v in st.items():
                if isinstance(v, dict):  # labels, needs_rescore_reasons: add per key
                    sub = acc.setdefault(k, {})
                    for lab, c in v.items():
                        sub[lab] = sub.get(lab, 0) + c
                elif isinstance(v, str):  # shape
                    acc[k] = v
                else:
                    acc[k] = acc.get(k, 0) + v
    n = sum(counts.values())
    n_valid = sum(s["n_valid"] for s in summaries)
    return {"n": n, "counts": {k: counts.get(k, 0) for k in SUMMARY_OUTCOMES}, "n_valid": n_valid,
            "rows": outcome_rows(counts, n), "ok_means": {k: t / w for k, (t, w) in means.items() if w},
            "ok_n": {k: w for k, (t, w) in means.items() if w},
            "judge_invalid": {m: dict(c) for m, c in judges.items()}, "runs": len(summaries),
            "panel": panel, "panel_votes_dropped": sum(p.get("votes_dropped", 0) for p in panel.values())}


def results_fields(summary: dict[str, Any]) -> dict[str, Any]:
    """Summary fields per model: n_valid, n_total and excluded counts per reason."""
    return {"n_valid": summary["n_valid"], "n_total": summary["n"],
            "excluded": {k: v for k, v in summary["counts"].items() if k != OK}}
