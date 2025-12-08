#!/usr/bin/env python3
"""
Simple script to run a single skill via RobotBrainSystem.

This initializes the full system (simulator, skill executor, etc.) and runs
a single skill. For skills that need UI (like human_intervention), use
test_human_intervention.py instead.

Usage:
    cd IsaacLab
    python -m robot_brain_system.examples.run_single_skill --list
    python -m robot_brain_system.examples.run_single_skill --skill grasp_spanner
    python -m robot_brain_system.examples.run_single_skill --skill open_box --timeout 120
    python -m robot_brain_system.examples.run_single_skill --skill move_to --param target="red box"
"""

import argparse
import json
import time

import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

from robot_brain_system.core.types import SkillStatus


def parse_param(param_str: str) -> tuple:
    """Parse a key=value parameter string."""
    key, _, value = param_str.partition("=")
    if not key:
        raise ValueError(f"Invalid parameter: {param_str}")
    # Try to parse as JSON, fallback to string
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


# Store CLI args globally (parsed before Hydra)
_cli_args = None


def parse_cli_args():
    """Parse CLI arguments before Hydra runs."""
    parser = argparse.ArgumentParser(
        description="Run a single skill via RobotBrainSystem",
        add_help=False,  # Avoid conflict with Hydra's --help
    )
    parser.add_argument("--skill", help="Name of the skill to run")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Skill parameter in key=value format (can be repeated)",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available skills and exit"
    )
    # Parse known args to avoid conflict with Hydra
    args, _ = parser.parse_known_args()
    return args


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig):
    global _cli_args
    args = _cli_args

    # Import here to avoid slow imports when just checking args
    from robot_brain_system.core.system import RobotBrainSystem

    if not args.skill and not args.list:
        print("Error: --skill is required (or use --list to see available skills)")
        print(
            "Usage: python -m robot_brain_system.examples.run_single_skill --skill <name>"
        )
        return 1

    # Initialize system
    print("Initializing RobotBrainSystem...")
    system = RobotBrainSystem(cfg)

    if not system.initialize():
        print("❌ Failed to initialize system")
        return 1

    print("✅ System initialized")

    # Get skill executor
    skill_executor = system.skill_executor
    if not skill_executor:
        print("❌ SkillExecutor not available")
        system.shutdown()
        return 1

    available_skills = list(skill_executor.registry.list_skills())

    # List skills and exit if requested
    if args.list:
        print("\n📋 Available skills:")
        for name in sorted(available_skills):
            info = skill_executor.registry.get_skill_info(name)
            desc = info.get("description", "")[:60] if info else ""
            print(f"  - {name}: {desc}...")
        system.shutdown()
        return 0

    # Check skill exists
    if args.skill not in available_skills:
        print(f"❌ Skill '{args.skill}' not found")
        print(f"Available: {', '.join(sorted(available_skills))}")
        system.shutdown()
        return 1

    # Parse parameters
    params = {}
    for p in args.param:
        key, value = parse_param(p)
        params[key] = value

    # Run skill
    print(f"\n🚀 Running skill: {args.skill}")
    if params:
        print(f"   Parameters: {params}")

    success = skill_executor.start_skill(args.skill, params)
    if not success:
        print("❌ Failed to start skill")
        system.shutdown()
        return 1

    print("✅ Skill started")

    # Poll until complete or timeout
    start_time = time.time()
    try:
        while True:
            time.sleep(1.0)
            status = skill_executor.get_status()
            elapsed = time.time() - start_time

            is_running = status.get("is_running", False)
            skill_status = status.get("status", "unknown")

            print(f"  [{elapsed:.0f}s] {skill_status}    ", end="\r")

            if not is_running:
                print(f"\n✅ Skill finished: {skill_status}")
                print(f"   Info: {status.get('status_info', '')}")
                print(f"   Time: {elapsed:.1f}s")
                break

            if elapsed > args.timeout:
                print(f"\n⚠️ Timeout ({args.timeout}s), terminating...")
                skill_executor.terminate_skill(SkillStatus.TIMEOUT, "timeout")
                break
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        skill_executor.terminate_skill()

    system.shutdown()
    return 0


def hydra_main():
    """Entry point that parses CLI args before Hydra."""
    global _cli_args
    _cli_args = parse_cli_args()

    # Clear Hydra if previously initialized
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    main()


if __name__ == "__main__":
    hydra_main()
