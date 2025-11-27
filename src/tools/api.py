"""HTTP helpers to query financialdatasets.ai with basic caching."""

from __future__ import annotations

import datetime
import os
import time
from typing import List, Optional

import pandas as pd
import requests

from src.data.cache import get_cache
from src.data.models import (
    CompanyFactsResponse,
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    InsiderTrade,
    InsiderTradeResponse,
    LineItem,
    LineItemResponse,
    Price,
    PriceResponse,
)

_cache = get_cache()


def _make_api_request(
    url: str,
    headers: dict,
    method: str = "GET",
    json_data: dict | None = None,
    max_retries: int = 3,
) -> requests.Response:
    for attempt in range(max_retries + 1):
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)

        if response.status_code == 429 and attempt < max_retries:
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s...")
            time.sleep(delay)
            continue

        return response


def _headers(api_key: Optional[str]) -> dict:
    headers = {}
    if api_key:
        headers["X-API-KEY"] = api_key
    return headers


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None) -> List[Price]:
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    url = (
        f"https://api.financialdatasets.ai/prices/?ticker={ticker}"
        f"&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    )
    response = _make_api_request(url, _headers(api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")))
    if response.status_code != 200:
        raise RuntimeError(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    price_response = PriceResponse(**response.json())
    prices = price_response.prices
    if prices:
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str | None = None,
) -> List[FinancialMetrics]:
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    url = (
        f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}"
        f"&report_period_lte={end_date}&limit={limit}&period={period}"
    )
    response = _make_api_request(url, _headers(api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")))
    if response.status_code != 200:
        raise RuntimeError(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    metrics_response = FinancialMetricsResponse(**response.json())
    financial_metrics = metrics_response.financial_metrics
    if financial_metrics:
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: List[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str | None = None,
) -> List[LineItem]:
    url = "https://api.financialdatasets.ai/financials/search/line-items"
    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = _make_api_request(
        url=url,
        headers=_headers(api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")),
        method="POST",
        json_data=body,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    return response_model.search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
) -> List[InsiderTrade]:
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    all_trades: list[InsiderTrade] = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, _headers(api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")))
        if response.status_code != 200:
            raise RuntimeError(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        if not start_date or len(insider_trades) < limit:
            break

        current_end_date = min(trade.filing_date for trade in insider_trades if trade.filing_date).split("T")[0]
        if current_end_date <= start_date:
            break

    if all_trades:
        _cache.set_insider_trades(cache_key, [t.model_dump() for t in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
) -> List[CompanyNews]:
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    all_news: list[CompanyNews] = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, _headers(api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")))
        if response.status_code != 200:
            raise RuntimeError(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        if not company_news:
            break

        all_news.extend(company_news)

        if not start_date or len(company_news) < limit:
            break

        current_end_date = min(news.date for news in company_news if news.date).split("T")[0]
        if start_date and current_end_date <= start_date:
            break

    if all_news:
        _cache.set_company_news(cache_key, [n.model_dump() for n in all_news])
    return all_news


def get_market_cap(ticker: str, end_date: str, api_key: str | None = None) -> float | None:
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        response = _make_api_request(url, _headers(api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")))
        if response.status_code != 200:
            print(f"Error fetching company facts: {ticker} - {response.status_code}")
            return None

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if not financial_metrics:
        return None
    return financial_metrics[0].market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str | None = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
