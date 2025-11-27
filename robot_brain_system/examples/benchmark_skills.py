#!/usr/bin/env python3
"""Benchmark script for testing skill success rates and performance."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from robot_brain_system.utils.skill_harness import SkillHarness

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark robot skills for success rate and performance"
    )
    parser.add_argument("--skill", required=True, help="Name of the skill to benchmark")
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to Hydra/OmegaConf config file",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of benchmark runs per skill",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout for each skill run (seconds)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run simulator in headless mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save benchmark results as JSON",
    )

    args, unknown = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # Parse simulator overrides from unknown args
    simulator_overrides = {}
    if args.headless:
        simulator_overrides["headless"] = True

    for arg in unknown:
        if arg.startswith("--simulator."):
            key_value = arg[12:]  # Remove "--simulator."
            if "=" in key_value:
                key, value = key_value.split("=", 1)
                try:
                    simulator_overrides[key] = json.loads(value)
                except json.JSONDecodeError:
                    simulator_overrides[key] = value

    with SkillHarness(
        config_path=args.config_path,
        simulator_overrides=simulator_overrides,
    ) as harness:
        available = list(harness.list_skills())
        if args.skill not in available:
            raise SystemExit(
                f"Skill '{args.skill}' not found. "
                f"Available skills: {', '.join(sorted(available))}"
            )

        print(f"\n{'=' * 60}")
        print(f"Benchmarking skill: {args.skill}")
        print(f"Number of runs: {args.num_runs}")
        print(f"Timeout per run: {args.timeout}s")
        print(f"{'=' * 60}\n")

        result = harness.benchmark_skill(
            args.skill,
            num_runs=args.num_runs,
            timeout=args.timeout,
        )

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK RESULTS: {result.skill_name}")
        print(f"{'=' * 60}")
        print(f"Total runs:      {result.total_runs}")
        print(f"Successes:       {result.successes}")
        print(f"Failures:        {result.failures}")
        print(f"Timeouts:        {result.timeouts}")
        print(f"Success rate:    {result.success_rate:.1%}")
        print(f"Avg time:        {result.avg_time_seconds:.2f}s")
        print(f"Min time:        {result.min_time_seconds:.2f}s")
        print(f"Max time:        {result.max_time_seconds:.2f}s")
        print(f"{'=' * 60}\n")

        # Print individual results
        print("Individual results:")
        for idx, run_result in enumerate(result.individual_results, 1):
            status_symbol = "✓" if run_result.status.value == "success" else "✗"
            print(
                f"  {status_symbol} Run {idx:2d}: {run_result.status.value:8s} "
                f"({run_result.elapsed_seconds:6.2f}s) - {run_result.status_info}"
            )

        # Save to JSON if requested
        if args.output:
            output_data = {
                "skill_name": result.skill_name,
                "total_runs": result.total_runs,
                "successes": result.successes,
                "failures": result.failures,
                "timeouts": result.timeouts,
                "success_rate": result.success_rate,
                "avg_time_seconds": result.avg_time_seconds,
                "min_time_seconds": result.min_time_seconds,
                "max_time_seconds": result.max_time_seconds,
                "individual_results": [
                    {
                        "run": idx + 1,
                        "status": r.status.value,
                        "status_info": r.status_info,
                        "elapsed_seconds": r.elapsed_seconds,
                    }
                    for idx, r in enumerate(result.individual_results)
                ],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
