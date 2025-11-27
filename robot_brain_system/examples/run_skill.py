#!/usr/bin/env python3
"""Command line helper for launching a single skill with :mod:`SkillHarness`."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List

from omegaconf import DictConfig, OmegaConf

from robot_brain_system.utils.skill_harness import SkillHarness
from robot_brain_system.utils.config_utils import (
    load_config,
    ensure_default_resolvers,
    default_config_path,
)

DEFAULT_CONFIG_PATH = default_config_path()


def _decode_value(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_parameters(raw_params: List[str]) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for item in raw_params:
        key, sep, value = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(
                f"Skill parameter '{item}' is missing an '=' separator. Use key=value format."
            )
        parsed[key] = _decode_value(value)
    return parsed


def _parse_override_tokens(tokens: List[str]) -> List[str]:
    overrides: List[str] = []
    for token in tokens:
        if not token.startswith("--"):
            raise argparse.ArgumentTypeError(
                f"Unrecognized argument '{token}'. Overrides must look like --path.to.key=value"
            )
        key_value = token[2:]
        if "=" not in key_value:
            raise argparse.ArgumentTypeError(
                f"Override '{token}' is missing an '=' separator. Use --path.to.key=value format."
            )
        overrides.append(key_value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Command line helper for launching a single skill with SkillHarness.\n"
            "Extra arguments of the form --path.to.key=value will override the loaded config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skill", required=True, help="Name of the skill to execute")
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Optional path to a Hydra/OmegaConf config file (defaults to conf/config.yaml)",
    )
    parser.add_argument(
        "--param",
        dest="params",
        action="append",
        default=[],
        help="Skill parameter in key=value format. May be provided multiple times.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float("inf"),
        help="Maximum time in seconds to wait for the skill to finish",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force simulator to run headless irrespective of the configuration file",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between skill status polls",
    )

    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    override_tokens = _parse_override_tokens(unknown)
    if args.headless:
        override_tokens.append("simulator.headless=true")

    ensure_default_resolvers()
    base_config = load_config(args.config_path, overrides=override_tokens)
    if not isinstance(base_config, DictConfig):
        raise TypeError("Composed configuration is not a DictConfig")

    OmegaConf.set_struct(base_config, False)

    skill_params = _parse_parameters(args.params)

    with SkillHarness(
        config=base_config,
        poll_interval=args.poll_interval,
    ) as harness:
        available = list(harness.list_skills())
        if args.skill not in available:
            raise SystemExit(
                f"Skill '{args.skill}' not found. Available skills: {', '.join(sorted(available))}"
            )
        result = harness.run_skill(
            args.skill,
            parameters=skill_params,
            timeout=args.timeout,
        )
        print(
            f"Skill '{result.name}' finished with status={result.status.value} "
            f"after {result.elapsed_seconds:.2f}s. Info: {result.status_info}"
        )


if __name__ == "__main__":
    main()
