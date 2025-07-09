#!/usr/bin/env python
"""
YAML-based Training Script for LeRobot Policies

This script allows training LeRobot policies using YAML configuration files,
while leveraging the official LeRobot training infrastructure.

Usage:
    python train_with_yaml.py --config configs/train_act_default.yaml
    python train_with_yaml.py --config configs/train_act_fast.yaml --training.steps 10000
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict
import inspect

import yaml

# Import LeRobot components
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.scripts.train import train
from lerobot.utils.utils import init_logging

# Import available policy configurations
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
# Add more policy imports as needed

# Policy type mapping
POLICY_CONFIG_MAPPING = {
    "act": ACTConfig,
    "diffusion": DiffusionConfig,
    # Add more policy types here:
    # "tdmpc": TDMPCConfig,
    # "vqbet": VQBETConfig,
    # "sac": SACConfig,
}


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def create_policy_config_from_yaml(policy_cfg: Dict[str, Any]) -> PreTrainedConfig:
    """Create policy configuration from YAML, supporting multiple policy types."""
    policy_type = policy_cfg.get("type", "act").lower()

    if policy_type not in POLICY_CONFIG_MAPPING:
        available_types = ", ".join(POLICY_CONFIG_MAPPING.keys())
        raise ValueError(
            f"Unsupported policy type: {policy_type}. Available types: {available_types}"
        )

    policy_config_class = POLICY_CONFIG_MAPPING[policy_type]

    # Handle special conversions
    policy_kwargs = policy_cfg.copy()

    # Remove 'type' as it's not a parameter for the config class
    policy_kwargs.pop("type", None)

    # Convert normalization_mapping if present
    if "normalization_mapping" in policy_kwargs:
        normalization_mapping = policy_kwargs["normalization_mapping"]
        if normalization_mapping and isinstance(normalization_mapping, dict):
            from lerobot.configs.types import NormalizationMode

            converted_mapping = {}
            for key, value in normalization_mapping.items():
                if isinstance(value, str):
                    converted_mapping[key] = getattr(NormalizationMode, value)
                else:
                    converted_mapping[key] = value
            policy_kwargs["normalization_mapping"] = converted_mapping

    # Get the constructor signature to filter valid parameters
    sig = inspect.signature(policy_config_class.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}

    # Filter kwargs to only include valid parameters for this policy config
    filtered_kwargs = {k: v for k, v in policy_kwargs.items() if k in valid_params}

    # Log which parameters are being used and which are ignored
    used_params = set(filtered_kwargs.keys())
    ignored_params = set(policy_kwargs.keys()) - used_params

    if ignored_params:
        logging.warning(
            f"Ignoring unknown parameters for {policy_type} policy: {ignored_params}"
        )

    logging.info(
        f"Creating {policy_type} policy config with parameters: {list(used_params)}"
    )

    # Create the policy configuration with only valid parameters
    try:
        policy_config = policy_config_class(**filtered_kwargs)
        return policy_config
    except Exception as e:
        logging.error(f"Failed to create {policy_type} policy config: {e}")
        logging.error(f"Valid parameters for {policy_type}: {valid_params}")
        logging.error(f"Provided parameters: {filtered_kwargs}")
        raise


def yaml_to_train_config(yaml_config: Dict[str, Any]) -> TrainPipelineConfig:
    """Convert YAML configuration to TrainPipelineConfig."""

    # Create dataset config
    dataset_cfg = yaml_config.get("dataset", {})
    dataset_config = DatasetConfig(
        repo_id=dataset_cfg["repo_id"],
        root=dataset_cfg.get("root", None),
        episodes=dataset_cfg.get("episodes", None),
        video_backend=dataset_cfg.get("video_backend", "pyav"),
    )

    # Create policy config using the new flexible function
    policy_cfg = yaml_config.get("policy", {})
    policy_config = create_policy_config_from_yaml(policy_cfg)

    # Create optimizer config
    optimizer_config = None
    optimizer_cfg = yaml_config.get("optimizer", {})
    if not optimizer_cfg.get("use_policy_training_preset", True):
        optimizer_config = AdamWConfig(
            lr=optimizer_cfg.get("lr", 1e-5),
            weight_decay=optimizer_cfg.get("weight_decay", 1e-4),
            grad_clip_norm=optimizer_cfg.get("grad_clip_norm", 10.0),
        )

    # Create scheduler config
    scheduler_config = None
    scheduler_cfg = yaml_config.get("scheduler", {})
    if (
        not optimizer_cfg.get("use_policy_training_preset", True)
        and scheduler_cfg.get("type") == "cosine_decay_with_warmup"
    ):
        scheduler_config = CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=scheduler_cfg.get("warmup_steps", 500),
            num_decay_steps=scheduler_cfg.get("num_decay_steps", 50000),
            peak_lr=scheduler_cfg.get("peak_lr", 1e-5),
            decay_lr=scheduler_cfg.get("decay_lr", 1e-6),
        )

    # Create evaluation config
    eval_cfg = yaml_config.get("eval", {})
    eval_config = EvalConfig(
        n_episodes=eval_cfg.get("n_episodes", 10),
        batch_size=eval_cfg.get("batch_size", 1),
        use_async_envs=eval_cfg.get("use_async_envs", False),
    )

    # Create wandb config
    wandb_cfg = yaml_config.get("wandb", {})
    wandb_config = WandBConfig(
        enable=wandb_cfg.get("enable", False),
        project=wandb_cfg.get("project", "lerobot_training"),
        run_id=wandb_cfg.get("run_id", None),
        disable_artifact=True,
    )

    # Create training config
    training_cfg = yaml_config.get("training", {})
    train_config = TrainPipelineConfig(
        # Required configs
        dataset=dataset_config,
        policy=policy_config,
        # Optional configs
        env=None,  # We don't support env config from YAML yet
        # Training parameters
        output_dir=Path(training_cfg.get("output_dir", "outputs/train/default")),
        job_name=training_cfg.get("job_name", None),
        resume=training_cfg.get("resume", False),
        seed=training_cfg.get("seed", 1000),
        num_workers=training_cfg.get("num_workers", 4),
        batch_size=training_cfg.get("batch_size", 8),
        steps=training_cfg.get("steps", 100000),
        eval_freq=training_cfg.get("eval_freq", 0),
        log_freq=training_cfg.get("log_freq", 200),
        save_checkpoint=training_cfg.get("save_checkpoint", True),
        save_freq=training_cfg.get("save_freq", 20000),
        # Use policy preset or custom optimizer/scheduler
        use_policy_training_preset=optimizer_cfg.get(
            "use_policy_training_preset", True
        ),
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        # Other configs
        eval=eval_config,
        wandb=wandb_config,
    )

    return train_config


def override_config_with_cli_args(
    config: TrainPipelineConfig, cli_overrides: Dict[str, Any]
) -> TrainPipelineConfig:
    """Override configuration with CLI arguments using dot notation."""
    for key, value in cli_overrides.items():
        # Split the key by dots to navigate nested attributes
        keys = key.split(".")
        obj = config

        # Navigate to the parent object
        for k in keys[:-1]:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                logging.warning(f"Unknown config path: {key}")
                break
        else:
            # Set the final attribute
            final_key = keys[-1]
            if hasattr(obj, final_key):
                # Convert value type to match the existing attribute type
                current_value = getattr(obj, final_key)
                if current_value is not None:
                    target_type = type(current_value)
                    try:
                        if target_type == bool:
                            # Handle boolean conversion
                            if isinstance(value, str):
                                converted_value = value.lower() in (
                                    "true",
                                    "1",
                                    "yes",
                                    "on",
                                )
                            else:
                                converted_value = bool(value)
                        elif target_type == Path:
                            converted_value = Path(value)
                        else:
                            converted_value = target_type(value)
                        setattr(obj, final_key, converted_value)
                        logging.info(f"Override: {key} = {converted_value}")
                    except (ValueError, TypeError) as e:
                        logging.warning(
                            f"Failed to convert {key}={value} to {target_type}: {e}"
                        )
                else:
                    setattr(obj, final_key, value)
                    logging.info(f"Override: {key} = {value}")
            else:
                logging.warning(f"Unknown config attribute: {key}")

    return config


def parse_cli_overrides(args: list[str]) -> Dict[str, Any]:
    """Parse CLI override arguments in the format --key=value or --key value."""
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]  # Remove '--'

            if "=" in key:
                # Format: --key=value
                key, value = key.split("=", 1)
            else:
                # Format: --key value
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    value = args[i + 1]
                    i += 1
                else:
                    # Boolean flag
                    value = True

            # Try to convert value to appropriate type
            if isinstance(value, str):
                # Try to parse as number
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string
                    pass

            overrides[key] = value
        i += 1

    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Train LeRobot policies with YAML configuration"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Parse known args to separate config file from overrides
    args, unknown_args = parser.parse_known_args()

    # Initialize logging
    init_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load YAML configuration
    logging.info(f"Loading configuration from: {args.config}")
    yaml_config = load_yaml_config(args.config)

    # Convert to TrainPipelineConfig
    train_config = yaml_to_train_config(yaml_config)

    # Parse and apply CLI overrides
    if unknown_args:
        cli_overrides = parse_cli_overrides(unknown_args)
        if cli_overrides:
            logging.info(f"Applying CLI overrides: {cli_overrides}")
            train_config = override_config_with_cli_args(train_config, cli_overrides)

    # Setup wandb offline mode if specified in config
    wandb_cfg = yaml_config.get("wandb", {})
    if wandb_cfg.get("offline", False):
        ...
    elif wandb_cfg.get("local_server", False):
        ...
        # os.environ["WANDB_MODE"] = "online"
        # os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
        # logging.info("Wandb set to use local server")

    # Log final configuration
    logging.info("Final training configuration:")
    logging.info(f"  Dataset: {train_config.dataset.repo_id}")
    policy_type = (
        getattr(train_config.policy, "type", "ACT") if train_config.policy else "ACT"
    )
    logging.info(f"  Policy: {policy_type}")
    logging.info(f"  Output dir: {train_config.output_dir}")
    logging.info(f"  Steps: {train_config.steps}")
    logging.info(f"  Batch size: {train_config.batch_size}")
    device = (
        getattr(train_config.policy, "device", "unknown")
        if train_config.policy
        else "unknown"
    )
    logging.info(f"  Device: {device}")
    logging.info(
        f"  Use policy training preset: {train_config.use_policy_training_preset}"
    )

    # Run training
    logging.info("Starting training...")
    train(train_config)


if __name__ == "__main__":
    main()
