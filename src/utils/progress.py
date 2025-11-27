"""Lightweight progress reporter for CLI runs."""

from __future__ import annotations

from typing import Optional


class _Progress:
    def update_status(self, agent: str, ticker: Optional[str], status: str, analysis: str | None = None) -> None:
        prefix = f"[{agent}]"
        middle = f" {ticker}" if ticker else ""
        suffix = f" - {status}"
        if analysis:
            suffix += f" | {analysis}"
        print(prefix + middle + suffix)


progress = _Progress()
