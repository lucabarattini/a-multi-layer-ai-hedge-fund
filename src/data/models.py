"""Typed data models used by the hedge fund pipeline."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class BaseAPIModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Price(BaseAPIModel):
    ticker: str
    time: str
    open: float
    close: float
    high: float
    low: float
    volume: float


class PriceResponse(BaseAPIModel):
    prices: List[Price] = Field(default_factory=list)


class FinancialMetrics(BaseAPIModel):
    report_period: Optional[str] = None
    market_cap: Optional[float] = None
    revenue_growth: Optional[float] = None
    free_cash_flow_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_invested_capital: Optional[float] = None
    debt_to_equity: Optional[float] = None
    operating_margin: Optional[float] = None
    current_ratio: Optional[float] = None
    enterprise_value: Optional[float] = None
    enterprise_value_to_ebitda_ratio: Optional[float] = None
    price_to_book_ratio: Optional[float] = None
    book_value_growth: Optional[float] = None
    interest_coverage: Optional[float] = None


class FinancialMetricsResponse(BaseAPIModel):
    financial_metrics: List[FinancialMetrics] = Field(default_factory=list)


class LineItem(BaseAPIModel):
    net_income: Optional[float] = None
    depreciation_and_amortization: Optional[float] = None
    capital_expenditure: Optional[float] = None
    working_capital: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    interest_expense: Optional[float] = None
    revenue: Optional[float] = None
    operating_income: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    outstanding_shares: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    shareholders_equity: Optional[float] = None
    dividends_and_other_cash_distributions: Optional[float] = None
    issuance_or_purchase_of_equity_shares: Optional[float] = None
    gross_profit: Optional[float] = None
    gross_margin: Optional[float] = None
    free_cash_flow: Optional[float] = None


class LineItemResponse(BaseAPIModel):
    search_results: List[LineItem] = Field(default_factory=list)


class InsiderTrade(BaseAPIModel):
    filing_date: Optional[str] = None
    transaction_shares: Optional[float] = None
    transaction_type: Optional[str] = None


class InsiderTradeResponse(BaseAPIModel):
    insider_trades: List[InsiderTrade] = Field(default_factory=list)


class CompanyNews(BaseAPIModel):
    date: Optional[str] = None
    title: Optional[str] = None
    sentiment: Optional[str] = None


class CompanyNewsResponse(BaseAPIModel):
    news: List[CompanyNews] = Field(default_factory=list)


class CompanyFacts(BaseAPIModel):
    market_cap: Optional[float] = None


class CompanyFactsResponse(BaseAPIModel):
    company_facts: CompanyFacts
