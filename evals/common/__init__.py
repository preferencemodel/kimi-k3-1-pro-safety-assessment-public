"""Shared code for every evaluation. Importing it applies the judge generation settings of the released runs
(provider_defaults.py)."""
from evals.common import provider_defaults as _provider_defaults

_provider_defaults.apply()
