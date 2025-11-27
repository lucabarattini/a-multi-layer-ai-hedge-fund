"""Buffett-style analytical helpers reused by multiple agents."""

from __future__ import annotations

from typing import Any, Dict, List

from src.data.models import FinancialMetrics, LineItem


def analyze_fundamentals(metrics: List[FinancialMetrics]) -> dict[str, Any]:
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}

    latest = metrics[0]
    score = 0
    reasoning: list[str] = []

    if latest.return_on_equity and latest.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"ROE {latest.return_on_equity:.1%} (>15%)")
    elif latest.return_on_equity:
        reasoning.append(f"ROE {latest.return_on_equity:.1%}")
    else:
        reasoning.append("ROE missing")

    if latest.debt_to_equity is not None:
        if latest.debt_to_equity < 0.5:
            score += 2
            reasoning.append("Low leverage (D/E < 0.5)")
        else:
            reasoning.append(f"High leverage (D/E {latest.debt_to_equity:.1f})")
    else:
        reasoning.append("D/E missing")

    if latest.operating_margin is not None:
        if latest.operating_margin > 0.15:
            score += 2
            reasoning.append("Healthy operating margin")
        else:
            reasoning.append(f"Thin margin {latest.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin missing")

    if latest.current_ratio is not None:
        if latest.current_ratio > 1.5:
            score += 1
            reasoning.append("Good liquidity")
        else:
            reasoning.append(f"Weak liquidity (CR {latest.current_ratio:.1f})")
    else:
        reasoning.append("Current ratio missing")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_consistency(financial_line_items: List[LineItem]) -> dict[str, Any]:
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient historical earnings data"}

    earnings = [i.net_income for i in financial_line_items if i.net_income is not None]
    if len(earnings) < 3:
        return {"score": 0, "details": "Net income missing across periods"}

    growth_periods = sum(1 for idx in range(len(earnings) - 1) if earnings[idx] > earnings[idx + 1])
    growth_rate = growth_periods / (len(earnings) - 1)

    score = 0
    if growth_rate >= 0.7:
        score = 3
        detail = "Consistent earnings growth"
    elif growth_rate >= 0.5:
        score = 2
        detail = "Decent earnings growth"
    else:
        detail = "Earnings growth inconsistent"

    return {"score": score, "details": detail}


def analyze_moat(metrics: List[FinancialMetrics]) -> dict[str, Any]:
    if len(metrics) < 4:
        return {"score": 0, "max_score": 4, "details": "Need more history for moat check"}

    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    margins = [m.operating_margin for m in metrics if m.operating_margin is not None]

    score = 0
    reasoning: list[str] = []

    if roes and sum(1 for r in roes if r > 0.15) / len(roes) > 0.6:
        score += 2
        reasoning.append("High ROE consistency")
    if margins and sum(1 for m in margins if m > 0.2) / len(margins) > 0.6:
        score += 1
        reasoning.append("Strong margins (pricing power)")
    if score == 0:
        reasoning.append("Moat not evident in ROE/margins")

    return {"score": score, "max_score": 4, "details": "; ".join(reasoning)}


def analyze_pricing_power(financial_line_items: List[LineItem]) -> dict[str, Any]:
    gross_margins = [item.gross_margin for item in financial_line_items if item.gross_margin is not None]
    if len(gross_margins) < 2:
        return {"score": 0, "details": "Gross margin history missing"}

    recent, older = gross_margins[0], gross_margins[-1]
    score = 0
    reasoning = []
    if recent > older + 0.02:
        score += 2
        reasoning.append("Improving gross margins")
    elif abs(recent - older) < 0.01:
        score += 1
        reasoning.append("Stable gross margins")
    else:
        reasoning.append("Gross margins weakening")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_book_value_growth(financial_line_items: List[LineItem]) -> dict[str, Any]:
    book_values = [
        item.shareholders_equity / item.outstanding_shares
        for item in financial_line_items
        if item.shareholders_equity and item.outstanding_shares
    ]
    if len(book_values) < 2:
        return {"score": 0, "details": "Book value data missing"}

    growth = book_values[0] - book_values[-1]
    score = 2 if growth > 0 else 0
    return {"score": score, "details": f"Book value change: {growth:,.2f}"}


def calculate_owner_earnings(financial_line_items: List[LineItem]) -> dict[str, Any]:
    if not financial_line_items:
        return {"owner_earnings": None, "details": ["No financials"]}
    latest = financial_line_items[0]
    if latest.net_income is None or latest.depreciation_and_amortization is None or latest.capital_expenditure is None:
        return {"owner_earnings": None, "details": ["Missing NI/DA/CapEx"]}

    # Simple working capital adjustment if available
    working_cap_change = 0.0
    if len(financial_line_items) >= 2:
        prev = financial_line_items[1]
        if latest.working_capital is not None and prev.working_capital is not None:
            working_cap_change = latest.working_capital - prev.working_capital

    owner_earnings = latest.net_income + latest.depreciation_and_amortization - abs(latest.capital_expenditure) - working_cap_change
    return {
        "owner_earnings": owner_earnings,
        "details": [
            f"NI {latest.net_income:,.0f}",
            f"D&A {latest.depreciation_and_amortization:,.0f}",
            f"CapEx {latest.capital_expenditure:,.0f}",
            f"WC Δ {working_cap_change:,.0f}",
        ],
    }


def calculate_intrinsic_value(financial_line_items: List[LineItem]) -> dict[str, Any]:
    owner_data = calculate_owner_earnings(financial_line_items)
    owner_earnings = owner_data.get("owner_earnings")
    latest = financial_line_items[0] if financial_line_items else None

    if not latest or owner_earnings is None or not latest.outstanding_shares:
        return {"intrinsic_value": None, "details": owner_data.get("details", [])}

    # Two-stage DCF with conservative assumptions
    stage1_growth = 0.05
    stage2_growth = 0.02
    discount_rate = 0.10
    years_stage1 = 5

    pv = 0.0
    for year in range(1, years_stage1 + 1):
        future = owner_earnings * (1 + stage1_growth) ** year
        pv += future / (1 + discount_rate) ** year

    terminal = owner_earnings * (1 + stage1_growth) ** years_stage1 * (1 + stage2_growth) / (discount_rate - stage2_growth)
    terminal_pv = terminal / (1 + discount_rate) ** years_stage1
    intrinsic_value_total = (pv + terminal_pv) * 0.85  # haircut

    per_share = intrinsic_value_total / latest.outstanding_shares
    return {
        "intrinsic_value": intrinsic_value_total,
        "intrinsic_value_per_share": per_share,
        "details": owner_data.get("details", []) + [f"IV/share {per_share:,.2f}"],
    }


def summarize_buffett_view(
    metrics: List[FinancialMetrics],
    line_items: List[LineItem],
    market_cap: float | None,
) -> Dict[str, Any]:
    fundamentals = analyze_fundamentals(metrics)
    consistency = analyze_consistency(line_items)
    moat = analyze_moat(metrics)
    pricing = analyze_pricing_power(line_items)
    book = analyze_book_value_growth(line_items)
    intrinsic = calculate_intrinsic_value(line_items)

    score = fundamentals["score"] + consistency["score"] + moat["score"] + pricing["score"] + book["score"]
    max_score = 2 + 3 + moat["max_score"] + 2 + 2

    mos = None
    if intrinsic["intrinsic_value"] and market_cap:
        mos = (intrinsic["intrinsic_value"] - market_cap) / market_cap

    return {
        "score": score,
        "max_score": max_score,
        "fundamentals": fundamentals,
        "consistency": consistency,
        "moat": moat,
        "pricing_power": pricing,
        "book_value": book,
        "intrinsic_value": intrinsic,
        "market_cap": market_cap,
        "margin_of_safety": mos,
    }
