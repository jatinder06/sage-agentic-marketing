"""Evaluation runner for the ADK multi-agent system.

Usage:
    python -m src.adk_agent.eval.run_eval                   # Run full eval set
    python -m src.adk_agent.eval.run_eval --agent sentiment  # Run sentiment only
    python -m src.adk_agent.eval.run_eval --runs 3           # 3 runs per case
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from google.adk.evaluation import AgentEvaluator
from google.adk.evaluation.eval_config import EvalConfig

from src.adk_agent.eval.eval_sets import get_full_eval_set, get_agent_eval_set

logger = logging.getLogger(__name__)

AGENT_MODULE = "src.adk_agent"


async def run_evaluation(
    agent_name: str | None = None,
    num_runs: int = 2,
) -> None:
    """Run the evaluation and print results."""
    if agent_name:
        eval_set = get_agent_eval_set(agent_name)
        print(f"\nRunning evaluation for: {agent_name} agent")
    else:
        eval_set = get_full_eval_set()
        print("\nRunning full evaluation suite")

    print(f"  Cases: {len(eval_set.eval_cases)}")
    print(f"  Runs per case: {num_runs}")
    print("-" * 60)

    eval_config = EvalConfig(
        criteria={
            "tool_trajectory_avg_score": 0.5,
        },
    )

    await AgentEvaluator.evaluate_eval_set(
        agent_module=AGENT_MODULE,
        eval_set=eval_set,
        eval_config=eval_config,
        num_runs=num_runs,
        print_detailed_results=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation suite for the AI Marketing agent system"
    )
    parser.add_argument(
        "--agent", type=str, default=None,
        help="Evaluate a specific agent (sentiment, churn, segmentation, "
             "support, content, recommendation, routing)",
    )
    parser.add_argument(
        "--runs", type=int, default=2,
        help="Number of runs per eval case (default: 2)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(run_evaluation(
        agent_name=args.agent,
        num_runs=args.runs,
    ))


if __name__ == "__main__":
    main()
