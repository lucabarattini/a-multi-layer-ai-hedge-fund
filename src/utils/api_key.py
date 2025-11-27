"""Helpers for pulling API keys from configuration/state."""

from __future__ import annotations

import os
from typing import Any, Mapping


def get_api_key_from_state(state: Mapping[str, Any], env_var: str) -> str | None:
    # Prefer explicit key in state, fall back to environment
    data = state.get("secrets") or {}
    return data.get(env_var) or os.environ.get(env_var)
