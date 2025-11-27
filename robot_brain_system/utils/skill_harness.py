"""Utility harness for spinning up the Isaac simulator and running a single skill.

This module provides a light-weight testing harness that removes the need to
initialize the full ``RobotBrainSystem`` stack when you only want to validate an
individual skill.  It is intentionally self-contained so it can be used from
unit tests, notebooks, or quick command-line experiments.
"""

from __future__ import annotations

import logging
import time
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, cast

from omegaconf import DictConfig, OmegaConf

from robot_brain_system.core.isaac_simulator import IsaacSimulator
from robot_brain_system.core.types import SkillStatus
from robot_brain_system.utils.config_utils import load_config

_logger = logging.getLogger(__name__)


@dataclass
class SkillRunResult:
    """Container describing the outcome of a harness executed skill."""

    name: str
    status: SkillStatus
    status_info: str
    elapsed_seconds: float
    raw_status: Dict[str, Any]


@dataclass
class SkillBenchmarkResult:
    """Container for skill benchmark statistics."""

    skill_name: str
    total_runs: int
    successes: int
    failures: int
    timeouts: int
    success_rate: float
    avg_time_seconds: float
    min_time_seconds: float
    max_time_seconds: float
    individual_results: list[SkillRunResult]


class SkillHarness(AbstractContextManager["SkillHarness"]):
    """Spin up the Isaac simulator in a subprocess and execute skills directly.

    Parameters
    ----------
    config_path:
        Optional path to an OmegaConf/Hydra YAML file.  If omitted we fall back
        to :data:`robot_brain_system.conf.config.DEVELOPMENT_CONFIG` which keeps
        the environment configuration in Python and avoids the need for Hydra.
    config:
        Pre-loaded configuration dictionary or ``DictConfig``.  Takes precedence
        over ``config_path`` when provided.
    simulator_overrides:
        Optional dictionary that is shallow-merged onto the simulator portion of
        the configuration before launch (e.g. ``{"headless": True}``).
    poll_interval:
        Seconds between successive status polls when monitoring an executing
        skill.
    """

    def __init__(
        self,
        *,
        config_path: Optional[str | Path] = None,
        config: Optional[Dict[str, Any] | DictConfig] = None,
        simulator_overrides: Optional[Dict[str, Any]] = None,
        poll_interval: float = 1.0,
    ) -> None:
        self._config_path = Path(config_path) if config_path else None
        self._provided_config = config
        self._simulator_overrides = simulator_overrides or {}
        self._poll_interval = poll_interval

        self.config: DictConfig | None = None
        self.simulator: IsaacSimulator | None = None
        self.skill_registry = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Context manager helpers
    def __enter__(self) -> "SkillHarness":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.shutdown()

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Load configuration, boot the simulator, and fetch the skill registry."""
        if self._initialized:
            _logger.debug("SkillHarness already initialized; skipping re-init.")
            return

        self.config = self._load_config()

        if "simulator" not in self.config:
            raise KeyError(
                "SkillHarness configuration missing required 'simulator' section"
            )

        sim_cfg_node = self.config["simulator"]
        if isinstance(sim_cfg_node, DictConfig):
            sim_cfg: DictConfig = OmegaConf.create(sim_cfg_node)
        else:
            sim_cfg = OmegaConf.create(sim_cfg_node or {})

        if self._simulator_overrides:
            override_cfg = OmegaConf.create(deepcopy(self._simulator_overrides))
            sim_cfg = cast(DictConfig, OmegaConf.merge(sim_cfg, override_cfg))

        if not isinstance(sim_cfg, DictConfig):
            raise TypeError(
                f"Simulator configuration must be a mapping, got {type(sim_cfg)}"
            )
        self.simulator = IsaacSimulator(sim_config=cast(Any, sim_cfg))
        if not self.simulator.initialize():
            raise RuntimeError("Failed to initialize Isaac simulator subprocess.")

        self.skill_registry = self.simulator.get_skill_registry_from_sim()
        if not self.skill_registry:
            self.simulator.shutdown()
            raise RuntimeError("Failed to retrieve skill registry from simulator.")

        _logger.info(
            "SkillHarness initialized with %d skills.",
            len(self.skill_registry.list_skills()),
        )
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the simulator subprocess if it is running."""
        if self.simulator:
            try:
                self.simulator.shutdown()
            finally:
                self.simulator = None
        self._initialized = False

    # ------------------------------------------------------------------
    def list_skills(self) -> Iterable[str]:
        """Return an iterable of available skill names."""
        if not self._initialized or not self.skill_registry:
            raise RuntimeError(
                "SkillHarness is not initialized. Call initialize() first."
            )
        return self.skill_registry.list_skills()

    def run_skill(
        self,
        skill_name: str,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: float = 300.0,
        terminate_on_timeout: bool = True,
        reset_on_finish: bool = False,
    ) -> SkillRunResult:
        """Execute a single skill inside the simulator.

        Parameters
        ----------
        skill_name:
            Name of the registered skill to execute.
        parameters:
            Optional keyword arguments forwarded to the skill constructor.
        timeout:
            Maximum number of seconds to wait for the skill to finish.
        terminate_on_timeout:
            When ``True`` we actively interrupt the skill with
            :class:`SkillStatus.TIMEOUT` before raising ``TimeoutError``.
        """

        if not self._initialized or not self.simulator or not self.skill_registry:
            raise RuntimeError(
                "SkillHarness is not initialized. Call initialize() first."
            )

        if skill_name not in self.skill_registry.list_skills():
            raise ValueError(
                f"Skill '{skill_name}' is not registered in the simulator."
            )

        parameters = parameters or {}
        start_success = self.simulator.start_skill_non_blocking(skill_name, parameters)
        if not start_success:
            status = self.simulator.get_skill_executor_status()
            raise RuntimeError(
                f"Failed to start skill '{skill_name}'. Status info: {status}"
            )

        start_time = time.monotonic()
        last_status: Dict[str, Any] = {}

        while True:
            time.sleep(self._poll_interval)
            last_status = self.simulator.get_skill_executor_status()
            if not last_status.get("is_running"):
                break
            if (time.monotonic() - start_time) > timeout:
                if terminate_on_timeout:
                    self.simulator.terminate_current_skill(
                        SkillStatus.TIMEOUT, status_info="harness_timeout"
                    )
                raise TimeoutError(
                    f"Skill '{skill_name}' timed out after {timeout} seconds."
                )

        elapsed = time.monotonic() - start_time
        status_value = last_status.get("status", SkillStatus.IDLE.value)
        try:
            status_enum = SkillStatus(status_value)
        except ValueError:
            status_enum = SkillStatus.IDLE

        status_info = last_status.get("status_info", "")
        _logger.info(
            "Skill '%s' finished with status=%s, info='%s' in %.2fs",
            skill_name,
            status_enum.value,
            status_info,
            elapsed,
        )

        # Reset environment if requested
        if reset_on_finish and self.simulator:
            _logger.debug("Resetting environment after skill completion")
            self.simulator.reset_env()

        return SkillRunResult(
            name=skill_name,
            status=status_enum,
            status_info=status_info,
            elapsed_seconds=elapsed,
            raw_status=last_status,
        )

    def benchmark_skill(
        self,
        skill_name: str,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        num_runs: int = 10,
        timeout: float = 300.0,
    ) -> SkillBenchmarkResult:
        """Run a skill multiple times and collect statistics.

        Parameters
        ----------
        skill_name:
            Name of the registered skill to benchmark.
        parameters:
            Optional keyword arguments forwarded to the skill constructor.
        num_runs:
            Number of times to run the skill.
        timeout:
            Maximum number of seconds to wait for each skill run.
        """
        if not self._initialized or not self.simulator or not self.skill_registry:
            raise RuntimeError(
                "SkillHarness is not initialized. Call initialize() first."
            )

        if skill_name not in self.skill_registry.list_skills():
            raise ValueError(
                f"Skill '{skill_name}' is not registered in the simulator."
            )

        _logger.info(
            f"Starting benchmark for skill '{skill_name}' with {num_runs} runs"
        )

        results: list[SkillRunResult] = []
        successes = 0
        failures = 0
        timeouts = 0

        for run_idx in range(num_runs):
            _logger.info(f"Benchmark run {run_idx + 1}/{num_runs}")

            try:
                result = self.run_skill(
                    skill_name,
                    parameters=parameters,
                    timeout=timeout,
                    terminate_on_timeout=True,
                    reset_on_finish=True,
                )
                results.append(result)

                if result.status == SkillStatus.COMPLETED:
                    successes += 1
                elif result.status == SkillStatus.TIMEOUT:
                    timeouts += 1
                else:
                    failures += 1

            except Exception as e:
                _logger.error(f"Benchmark run {run_idx + 1} failed with exception: {e}")
                # Create a failed result
                results.append(
                    SkillRunResult(
                        name=skill_name,
                        status=SkillStatus.FAILED,
                        status_info=f"Exception: {str(e)}",
                        elapsed_seconds=0.0,
                        raw_status={},
                    )
                )
                failures += 1

        # Calculate statistics
        times = [r.elapsed_seconds for r in results]
        success_rate = successes / num_runs if num_runs > 0 else 0.0

        benchmark_result = SkillBenchmarkResult(
            skill_name=skill_name,
            total_runs=num_runs,
            successes=successes,
            failures=failures,
            timeouts=timeouts,
            success_rate=success_rate,
            avg_time_seconds=sum(times) / len(times) if times else 0.0,
            min_time_seconds=min(times) if times else 0.0,
            max_time_seconds=max(times) if times else 0.0,
            individual_results=results,
        )

        _logger.info(
            f"Benchmark completed: {successes}/{num_runs} "
            f"({success_rate:.1%} success rate), "
            f"avg time: {benchmark_result.avg_time_seconds:.2f}s"
        )

        return benchmark_result

    def run_skill_sequence(
        self,
        skills: list[tuple[str, Optional[Dict[str, Any]]]],
        *,
        timeout: float = 300.0,
        stop_on_failure: bool = True,
    ) -> list[SkillRunResult]:
        """Run multiple skills in sequence.

        Parameters
        ----------
        skills:
            List of (skill_name, parameters) tuples to execute in order.
        timeout:
            Maximum number of seconds to wait for each skill.
        stop_on_failure:
            If True, stop the sequence if any skill fails.
        """
        if not self._initialized or not self.simulator or not self.skill_registry:
            raise RuntimeError(
                "SkillHarness is not initialized. Call initialize() first."
            )

        results: list[SkillRunResult] = []
        self.simulator.reset_env()
        for idx, (skill_name, parameters) in enumerate(skills):
            _logger.info(f"Running skill {idx + 1}/{len(skills)}: '{skill_name}'")

            try:
                result = self.run_skill(
                    skill_name,
                    parameters=parameters,
                    timeout=timeout,
                    terminate_on_timeout=True,
                    reset_on_finish=False,
                )
                results.append(result)

                if stop_on_failure and result.status != SkillStatus.COMPLETED:
                    _logger.warning(f"Skill '{skill_name}' failed, stopping sequence")
                    break

            except Exception as e:
                _logger.error(f"Skill '{skill_name}' raised exception: {e}")
                import traceback

                traceback.print_exc()
                results.append(
                    SkillRunResult(
                        name=skill_name,
                        status=SkillStatus.FAILED,
                        status_info=f"Exception: {str(e)}",
                        elapsed_seconds=0.0,
                        raw_status={},
                    )
                )
                if stop_on_failure:
                    break

        _logger.info(
            f"Skill sequence completed: {len(results)}/{len(skills)} skills executed"
        )
        return results

    # ------------------------------------------------------------------
    def _load_config(self) -> DictConfig:
        """Resolve the configuration source and normalise to ``DictConfig``."""
        return load_config(
            config_path=self._config_path,
            config=self._provided_config,
            allow_struct=False,
        )
