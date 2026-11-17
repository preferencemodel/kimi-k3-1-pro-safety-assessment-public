"""Judge generation settings that match the released logs.

Inspect's `together/` provider sends `max_tokens=2048` on every call that does not set it (Inspect's
DEFAULT_MAX_TOKENS). The released runs called the open-weight judges (GLM-5.3, DeepSeek V4.1 Flash) through an
OpenAI-compatible route with no client-side cap: their logged calls carry no `max_tokens`, and the longest judge
outputs run past 40,000 tokens with no truncated vote. With the 2,048-token default, a reasoning judge such as GLM-5.3
often stops at the cap, and the outcome rule drops its vote as truncated.

`apply()` removes that default for `together/` models, so a call without `max_tokens` sends none, as in the released
runs. Calls that set `max_tokens` (the model under test, and judges whose task code sets one) are unchanged. It runs
when any module under `evals/common` is imported, which every task does.
"""
from __future__ import annotations


def apply() -> None:
    """Remove Inspect's client-side 2,048-token default for `together/` models (idempotent)."""
    try:
        from inspect_ai.model._providers.together import TogetherAIAPI
    except Exception:  # noqa: BLE001  (provider not importable: nothing to change)
        return
    if getattr(TogetherAIAPI, "_no_default_max_tokens", False):
        return

    def max_tokens(self) -> int | None:  # noqa: ARG001
        return None

    TogetherAIAPI.max_tokens = max_tokens  # type: ignore[method-assign]
    TogetherAIAPI._no_default_max_tokens = True  # type: ignore[attr-defined]
