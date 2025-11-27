"""Multi-agent pipeline implementing the decomposer -> actor -> monitor -> predictor -> evaluator -> orchestrator chain."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.analysis.buffett import summarize_buffett_view
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    get_company_news,
    get_insider_trades,
    search_line_items,
)
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class TaskPlan(BaseModel):
    steps: List[str]


class BuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="0-100")
    reasoning: str


class InternReport(BaseModel):
    ticker: str
    score: int
    max_score: int
    margin_of_safety: Optional[float]
    summary: str
    raw: Dict[str, Any]


class MonitorReport(BaseModel):
    status: Literal["pass", "revise"]
    notes: str


class ScenarioReport(BaseModel):
    base: float
    bear: float
    bull: float
    commentary: str


class EvaluationReport(BaseModel):
    decision: BuffettSignal
    monitor: MonitorReport
    scenarios: ScenarioReport
    intern_report: InternReport


def task_decomposer_agent(state: AgentState, agent_id: str = "task_decomposer") -> dict:
    """Break the Buffett analysis into actionable steps using LLM."""
    goal = state["goal"]
    tickers = state["data"]["tickers"]
    debug = state["metadata"].get("debug", False)
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the CEO agent. Break the goal into concise ordered steps that a junior analyst can execute. "
                "Keep steps practical and focused on Buffett-style fundamentals.",
            ),
            (
                "human",
                "Goal: {goal}\nTickers: {tickers}\nReturn JSON exactly: {{\"steps\":[\"step1\",...]}}",
            ),
        ]
    )
    prompt = template.invoke({"goal": goal, "tickers": ", ".join(tickers)})

    model = state["metadata"].get("model")
    plan = call_llm(
        prompt=prompt,
        pydantic_model=TaskPlan,
        model=model,
        default_factory=lambda: TaskPlan(steps=["Collect financial data", "Assess quality", "Estimate intrinsic value"]),
        debug=debug,
        agent_name=agent_id,
    )
    message = HumanMessage(content=plan.model_dump_json(), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(plan.steps, agent_id)

    state["data"]["plan"] = plan.steps
    return {"messages": [message], "data": state["data"]}


def intern_actor_agent(state: AgentState, agent_id: str = "intern_actor") -> dict:
    """Fetch data and run Buffett analysis."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    debug = state["metadata"].get("debug", False)
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    reports: Dict[str, InternReport] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Collecting metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=8, api_key=api_key)
        line_items = search_line_items(
            ticker,
            [
                "net_income",
                "depreciation_and_amortization",
                "capital_expenditure",
                "working_capital",
                "outstanding_shares",
                "gross_margin",
                "shareholders_equity",
                "total_assets",
                "total_liabilities",
                "revenue",
                "free_cash_flow",
            ],
            end_date=end_date,
            period="ttm",
            limit=8,
            api_key=api_key,
        )
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Synthesizing Buffett view")
        view = summarize_buffett_view(metrics, line_items, market_cap)
        if debug:
            print(f"\n[data][{agent_id}][{ticker}] metrics:\n{json.dumps([m.model_dump() for m in metrics], indent=2)}")
            print(f"\n[data][{agent_id}][{ticker}] line_items:\n{json.dumps([li.model_dump() for li in line_items], indent=2)}")
            print(f"\n[calc][{agent_id}][{ticker}] buffett_view:\n{json.dumps(view, indent=2)}\n")
        if view["margin_of_safety"] is not None:
            mos_text = f"MoS {view['margin_of_safety']:.1%}"
        else:
            mos_text = "MoS n/a"
        summary = f"Score {view['score']}/{view['max_score']}; {mos_text}"
        reports[ticker] = InternReport(
            ticker=ticker,
            score=view["score"],
            max_score=view["max_score"],
            margin_of_safety=view["margin_of_safety"],
            summary=summary,
            raw=view,
        )
        progress.update_status(agent_id, ticker, "Done", analysis=summary)

    message = HumanMessage(content=json.dumps({k: v.model_dump() for k, v in reports.items()}), name=agent_id)
    state["data"]["intern_reports"] = reports
    return {"messages": [message], "data": state["data"]}


def monitor_agent(state: AgentState, agent_id: str = "monitor") -> dict:
    """Quality-check the intern's analysis using the decomposed plan."""
    reports: Dict[str, InternReport] = state["data"].get("intern_reports", {})
    plan_steps = state["data"].get("plan", [])
    debug = state["metadata"].get("debug", False)

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the monitor. Verify that the intern covered every planned step, flagged missing data, "
                "and avoided unsupported conclusions. Keep feedback terse.",
            ),
            (
                "human",
                "Plan steps: {steps}\nIntern report: {report}\nReturn JSON exactly: {{\"status\":\"pass|revise\",\"notes\":\"...\"}}",
            ),
        ]
    )

    monitor_results: Dict[str, MonitorReport] = {}
    model = state["metadata"].get("model")
    for ticker, report in reports.items():
        prompt = template.invoke({"steps": plan_steps, "report": report.model_dump_json()})
        out = call_llm(
            prompt=prompt,
            pydantic_model=MonitorReport,
            model=model,
            default_factory=lambda: MonitorReport(status="pass", notes="Auto-pass"),
            debug=debug,
            agent_name=agent_id,
        )
        monitor_results[ticker] = out
        progress.update_status(agent_id, ticker, f"{out.status.upper()}: {out.notes}")

    message = HumanMessage(content=json.dumps({k: v.model_dump() for k, v in monitor_results.items()}), name=agent_id)
    state["data"]["monitor_reports"] = monitor_results
    return {"messages": [message], "data": state["data"]}


def predictor_agent(state: AgentState, agent_id: str = "predictor") -> dict:
    """Run simple what-if scenarios based on margin of safety."""
    reports: Dict[str, InternReport] = state["data"].get("intern_reports", {})
    scenario_results: Dict[str, ScenarioReport] = {}

    for ticker, report in reports.items():
        mos = report.margin_of_safety or 0
        base = mos
        bear = mos - 0.15
        bull = mos + 0.15
        commentary = (
            f"Base MoS {base:.1%}, bear assumes 15% worse valuation gap, bull 15% better. "
            "Negative means overvalued."
        )
        scenario_results[ticker] = ScenarioReport(base=base, bear=bear, bull=bull, commentary=commentary)
        progress.update_status(agent_id, ticker, "Scenarios ready", analysis=commentary)

    message = HumanMessage(content=json.dumps({k: v.model_dump() for k, v in scenario_results.items()}), name=agent_id)
    state["data"]["scenarios"] = scenario_results
    return {"messages": [message], "data": state["data"]}


def evaluator_agent(state: AgentState, agent_id: str = "evaluator") -> dict:
    """Score the intern output, monitor notes, and scenarios to produce a Buffett-style signal."""
    intern_reports: Dict[str, InternReport] = state["data"].get("intern_reports", {})
    monitor_reports: Dict[str, MonitorReport] = state["data"].get("monitor_reports", {})
    scenarios: Dict[str, ScenarioReport] = state["data"].get("scenarios", {})
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    end_date = state["data"]["end_date"]

    decisions: Dict[str, EvaluationReport] = {}

    debug = state["metadata"].get("debug", False)
    model = state["metadata"].get("model")
    for ticker, intern_report in intern_reports.items():
        monitor_report = monitor_reports.get(ticker) or MonitorReport(status="pass", notes="No monitor feedback")
        scenario = scenarios.get(ticker) or ScenarioReport(base=0.0, bear=0.0, bull=0.0, commentary="No scenarios")

        # Pull lightweight sentiment signals (insider + news count) to enrich the evaluator
        insider = get_insider_trades(ticker, end_date=end_date, limit=200, api_key=api_key)
        news = get_company_news(ticker, end_date=end_date, limit=50, api_key=api_key)
        insider_bias = sum(1 for t in insider if (t.transaction_shares or 0) < 0) - sum(
            1 for t in insider if (t.transaction_shares or 0) > 0
        )
        news_bias = sum(1 for n in news if n.sentiment == "positive") - sum(1 for n in news if n.sentiment == "negative")

        evaluator_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the evaluator agent. Decide bullish, bearish, or neutral based on quality score, "
                    "margin of safety scenarios, monitor notes, and light sentiment. Be concise.",
                ),
                (
                    "human",
                    "Ticker: {ticker}\n"
                    "Intern summary: {summary}\n"
                    "Monitor: {monitor}\n"
                    "Scenarios: {scenarios}\n"
                    "Sentiment counts: insider_bias={insider_bias}, news_bias={news_bias}\n"
                    'Return JSON exactly: {{"signal": "...", "confidence": 0-100, "reasoning": "..."}}',
                ),
            ]
        )
        prompt = evaluator_prompt.invoke(
            {
                "ticker": ticker,
                "summary": intern_report.model_dump(),
                "monitor": monitor_report.model_dump(),
                "scenarios": scenario.model_dump(),
                "insider_bias": insider_bias,
                "news_bias": news_bias,
            }
        )

        def _default_signal():
            return BuffettSignal(signal="neutral", confidence=50, reasoning="Default evaluator decision")

        decision = call_llm(
            prompt=prompt,
            pydantic_model=BuffettSignal,
            model=model,
            default_factory=_default_signal,
            debug=debug,
            agent_name=agent_id,
        )

        decisions[ticker] = EvaluationReport(
            decision=decision,
            monitor=monitor_report,
            scenarios=scenario,
            intern_report=intern_report,
        )
        progress.update_status(agent_id, ticker, f"{decision.signal.upper()} ({decision.confidence}%)", analysis=decision.reasoning)

    message = HumanMessage(content=json.dumps({k: v.model_dump() for k, v in decisions.items()}), name=agent_id)
    state["data"]["decisions"] = decisions
    return {"messages": [message], "data": state["data"]}


def orchestrator(state: AgentState) -> AgentState:
    """Run the full chain of agents for the provided tickers."""
    # 1) task decomposer
    task_decomposer_agent(state)
    # 2) intern actor
    intern_actor_agent(state)
    # 3) monitor
    monitor_agent(state)
    # 4) predictor
    predictor_agent(state)
    # 5) evaluator
    evaluator_agent(state)
    return state
