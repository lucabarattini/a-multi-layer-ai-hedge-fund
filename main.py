"""CLI entrypoint for the multi-agent Buffett hedge fund prototype."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

from dotenv import load_dotenv
from src.agents.pipeline import orchestrator
from src.graph.state import AgentState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent Buffett hedge fund prototype")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT", help="Comma-separated tickers")
    parser.add_argument("--end-date", type=str, default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
    )
    parser.add_argument("--api-key", type=str, help="financialdatasets.ai API key (otherwise uses env var)")
    parser.add_argument("--model", type=str, default=None, help="LLM model (e.g., ollama:llama3)")
    parser.add_argument("--show-reasoning", action="store_true", help="Print intermediate reasoning")
    parser.add_argument("--debug", action="store_true", help="Verbose logging: raw LLM output and data dumps")
    return parser.parse_args()


def main() -> None:
    # Load environment variables from .env if present
    load_dotenv()

    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    state: AgentState = {
        "goal": "Run Warren Buffett-style analysis through a chain of agents",
        "metadata": {"show_reasoning": args.show_reasoning, "model": args.model, "debug": args.debug},
        "data": {
            "tickers": tickers,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "analyst_signals": {},
        },
        "messages": [],
        "secrets": {"FINANCIAL_DATASETS_API_KEY": args.api_key} if args.api_key else {},
    }

    final_state = orchestrator(state)
    decisions = final_state["data"].get("decisions", {})
    print("\n=== Final decisions ===")
    print(json.dumps({k: v.decision.model_dump() for k, v in decisions.items()}, indent=2))


if __name__ == "__main__":
    main()
