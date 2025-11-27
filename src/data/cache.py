"""Lightweight in-memory cache used for API responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SimpleCache:
    prices: Dict[str, List[dict]] = field(default_factory=dict)
    financial_metrics: Dict[str, List[dict]] = field(default_factory=dict)
    insider_trades: Dict[str, List[dict]] = field(default_factory=dict)
    company_news: Dict[str, List[dict]] = field(default_factory=dict)

    def get_prices(self, key: str) -> Optional[List[dict]]:
        return self.prices.get(key)

    def set_prices(self, key: str, value: List[dict]) -> None:
        self.prices[key] = value

    def get_financial_metrics(self, key: str) -> Optional[List[dict]]:
        return self.financial_metrics.get(key)

    def set_financial_metrics(self, key: str, value: List[dict]) -> None:
        self.financial_metrics[key] = value

    def get_insider_trades(self, key: str) -> Optional[List[dict]]:
        return self.insider_trades.get(key)

    def set_insider_trades(self, key: str, value: List[dict]) -> None:
        self.insider_trades[key] = value

    def get_company_news(self, key: str) -> Optional[List[dict]]:
        return self.company_news.get(key)

    def set_company_news(self, key: str, value: List[dict]) -> None:
        self.company_news[key] = value


_CACHE = SimpleCache()


def get_cache() -> SimpleCache:
    return _CACHE
