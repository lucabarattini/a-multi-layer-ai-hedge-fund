"""Simple state container used by agents."""

from __future__ import annotations

from typing import Any, Dict


AgentState = Dict[str, Any]


def show_agent_reasoning(reasoning: Any, agent_id: str) -> None:
    print(f"\n--- {agent_id} reasoning ---")
    print(reasoning)
    print("--- end ---\n")
