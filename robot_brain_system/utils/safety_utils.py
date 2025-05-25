"""
Safety utilities for the robot brain system.
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from ..core.types import Action, Observation


class SafetyLimits:
    """Safety limits configuration."""

    def __init__(self, config: Dict[str, Any]):
        self.position_limits = config.get(
            "position_limits",
            {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]},
        )
        self.velocity_limits = config.get(
            "velocity_limits", {"linear": 1.0, "angular": 3.14}
        )
        self.force_limits = config.get(
            "force_limits",
            {
                "max_force": 100.0  # Newtons
            },
        )
        self.emergency_stop_enabled = config.get(
            "emergency_stop_enabled", True
        )


def check_safety_limits(
    action: Action,
    observation: Optional[Observation] = None,
    safety_limits: Optional[SafetyLimits] = None,
) -> Tuple[bool, List[str]]:
    """
    Check if an action violates safety limits.

    Args:
        action: Action to check
        observation: Current observation (optional)
        safety_limits: Safety limits configuration

    Returns:
        Tuple of (is_safe, list_of_violations)
    """
    if safety_limits is None:
        # Default safety limits
        safety_limits = SafetyLimits({})

    violations = []

    # Check action data
    if action.data is not None:
        action_array = action.to_numpy()

        # Check for NaN or infinite values
        if np.any(np.isnan(action_array)) or np.any(np.isinf(action_array)):
            violations.append("Action contains NaN or infinite values")

        # Check velocity limits (assuming first N elements are velocities)
        if len(action_array) >= 3:
            linear_vel = np.linalg.norm(action_array[:3])
            if linear_vel > safety_limits.velocity_limits["linear"]:
                violations.append(
                    f"Linear velocity {linear_vel:.3f} exceeds limit {safety_limits.velocity_limits['linear']}"
                )

        if len(action_array) >= 6:
            angular_vel = np.linalg.norm(action_array[3:6])
            if angular_vel > safety_limits.velocity_limits["angular"]:
                violations.append(
                    f"Angular velocity {angular_vel:.3f} exceeds limit {safety_limits.velocity_limits['angular']}"
                )

    # Check observation-based limits
    if observation and observation.data is not None:
        if isinstance(observation.data, dict):
            # Check position limits if available
            position = observation.data.get(
                "position", observation.data.get("ee_position")
            )
            if position is not None and len(position) >= 3:
                pos_array = np.array(position[:3])

                if (
                    pos_array[0] < safety_limits.position_limits["x"][0]
                    or pos_array[0] > safety_limits.position_limits["x"][1]
                ):
                    violations.append(
                        f"X position {pos_array[0]:.3f} outside limits {safety_limits.position_limits['x']}"
                    )

                if (
                    pos_array[1] < safety_limits.position_limits["y"][0]
                    or pos_array[1] > safety_limits.position_limits["y"][1]
                ):
                    violations.append(
                        f"Y position {pos_array[1]:.3f} outside limits {safety_limits.position_limits['y']}"
                    )

                if (
                    pos_array[2] < safety_limits.position_limits["z"][0]
                    or pos_array[2] > safety_limits.position_limits["z"][1]
                ):
                    violations.append(
                        f"Z position {pos_array[2]:.3f} outside limits {safety_limits.position_limits['z']}"
                    )

    is_safe = len(violations) == 0
    return is_safe, violations


def emergency_stop_check(
    observation: Optional[Observation] = None,
    system_state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check if an emergency stop condition is triggered.

    Args:
        observation: Current observation
        system_state: Current system state

    Returns:
        Tuple of (should_emergency_stop, reason)
    """
    # Check for critical errors in observation
    if observation and observation.metadata:
        error_level = observation.metadata.get("error_level")
        if error_level == "critical":
            return True, "Critical error detected in observation"

    # Check system state
    if system_state:
        if system_state.get("status") == "error":
            error_msg = system_state.get("error_message", "Unknown error")
            if (
                "emergency" in error_msg.lower()
                or "critical" in error_msg.lower()
            ):
                return True, f"System error: {error_msg}"

        # Check for timeout conditions
        task_duration = system_state.get("task_duration", 0)
        max_duration = system_state.get(
            "max_task_duration", 600
        )  # 10 minutes default
        if task_duration > max_duration:
            return (
                True,
                f"Task timeout: {task_duration:.1f}s > {max_duration:.1f}s",
            )

    return False, None


def apply_safety_clamp(
    action: Action, safety_limits: Optional[SafetyLimits] = None
) -> Action:
    """
    Apply safety clamping to an action to ensure it's within limits.

    Args:
        action: Original action
        safety_limits: Safety limits configuration

    Returns:
        Clamped action
    """
    if safety_limits is None:
        safety_limits = SafetyLimits({})

    if action.data is None:
        return action

    action_array = action.to_numpy().copy()

    # Clamp velocity components
    if len(action_array) >= 3:
        linear_vel = action_array[:3]
        linear_magnitude = np.linalg.norm(linear_vel)
        if linear_magnitude > safety_limits.velocity_limits["linear"]:
            action_array[:3] = linear_vel * (
                safety_limits.velocity_limits["linear"] / linear_magnitude
            )

    if len(action_array) >= 6:
        angular_vel = action_array[3:6]
        angular_magnitude = np.linalg.norm(angular_vel)
        if angular_magnitude > safety_limits.velocity_limits["angular"]:
            action_array[3:6] = angular_vel * (
                safety_limits.velocity_limits["angular"] / angular_magnitude
            )

    # Create new action with clamped data
    clamped_action = Action(
        data=action_array,
        metadata={
            **(action.metadata or {}),
            "safety_clamped": True,
            "original_action": action.data,
        },
    )

    return clamped_action


class SafetyMonitor:
    """Real-time safety monitoring system."""

    def __init__(self, safety_limits: SafetyLimits):
        self.safety_limits = safety_limits
        self.violation_count = 0
        self.last_violation_time = 0
        self.emergency_stop_triggered = False

    def check_action(
        self, action: Action, observation: Optional[Observation] = None
    ) -> Dict[str, Any]:
        """
        Check an action and return safety assessment.

        Returns:
            Dict with safety status, violations, and recommendations
        """
        is_safe, violations = check_safety_limits(
            action, observation, self.safety_limits
        )

        if not is_safe:
            self.violation_count += 1
            self.last_violation_time = time.time()

        # Check for emergency stop conditions
        should_estop, estop_reason = emergency_stop_check(observation)

        if should_estop and not self.emergency_stop_triggered:
            self.emergency_stop_triggered = True

        return {
            "is_safe": is_safe,
            "violations": violations,
            "violation_count": self.violation_count,
            "emergency_stop": should_estop,
            "emergency_reason": estop_reason,
            "recommended_action": "clamp" if violations else "continue",
        }
