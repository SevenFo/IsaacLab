#!/usr/bin/env python3
"""Run multiple skills in sequence for integration testing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from robot_brain_system.utils.skill_harness import SkillHarness

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple skills in sequence")
    parser.add_argument(
        "--skills",
        nargs="+",
        required=True,
        help="List of skills to run in sequence (e.g., open_box grasp_spanner)",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to Hydra/OmegaConf config file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout for each skill (seconds)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the sequence if any skill fails",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run simulator in headless mode",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # Prepare skill sequence (skill_name, parameters)
    skills = [(skill_name, None) for skill_name in args.skills]

    simulator_overrides = {}
    if args.headless:
        simulator_overrides["headless"] = True

    test_times = 30
    success_count = 0
    with SkillHarness(
        config_path=args.config_path,
        simulator_overrides=simulator_overrides,
    ) as harness:
        print(f"\n{'=' * 60}")
        print("Running skill sequence:")
        for idx, (skill_name, _) in enumerate(skills, 1):
            print(f"  {idx}. {skill_name}")
        print(f"{'=' * 60}\n")
        for count in range(test_times):
            results = harness.run_skill_sequence(
                skills,
                timeout=args.timeout,
                stop_on_failure=args.stop_on_failure,
            )

            # Print summary
            print(f"\n{'=' * 60}")
            print("SEQUENCE RESULTS")
            print(f"{'=' * 60}")
            success = True
            for idx, result in enumerate(results, 1):
                status_symbol = "✓" if result.status.value == "completed" else "✗"
                success = success and (result.status.value == "completed")
                print(
                    f"{status_symbol} {idx}. {result.name:20s} - {result.status.value:8s} "
                    f"({result.elapsed_seconds:6.2f}s)"
                )
            print(f"{'=' * 60}\n")

            success_count += int(success)
        print(
            f"Skill sequence succeeded {success_count} / {test_times} times == {success_count / test_times * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
