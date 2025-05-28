"""
Skill registry and management system based on intelligent_robot_system design.
Skills execute directly in the Isaac subprocess with direct environment access.
"""

import inspect
import importlib
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Any, Callable, Generator
from dataclasses import dataclass

from .types import (
    SkillDefinition,
    SkillType,
    ExecutionMode,
    SkillStatus,  # Make sure SkillStatus is defined in types.py
)


class SkillRegistry:
    """Registry for managing all available skills."""

    def __init__(self):
        self.skills: Dict[str, SkillDefinition] = {}
        # self.skill_instances: Dict[str, Any] = {} # Not typically used for function-based skills

    def register_skill(
        self,
        function: Callable,
        skill_type: SkillType,
        execution_mode: ExecutionMode,
        name: str | None = None,
        description: str = "",
        timeout: Optional[float] = None,
        requires_env: bool = False,
        criterion: Dict[str, str] | None = None,  # 新增参数
    ):
        """Register a skill in the registry."""
        name = name or function.__name__
        description = description or function.__doc__ or ""
        if name in self.skills:
            print(
                f"[SkillRegistry] Warning: Skill '{name}' already registered. Overwriting."
            )

        skill_def = SkillDefinition(
            name=name,
            skill_type=skill_type,
            execution_mode=execution_mode,
            function=function,
            description=description,
            criterion=criterion,
            timeout=timeout,
            requires_env=requires_env,
            criterion=criterion,
        )

        self.skills[name] = skill_def
        print(
            f"[SkillRegistry] Registered skill: {name} ({skill_type.value}, {execution_mode.value})"
        )

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Get a skill definition by name."""
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """Get list of all registered skill names."""
        return list(self.skills.keys())

    def list_skills_by_type(self, skill_type: SkillType) -> List[str]:
        """Get list of skills by type."""
        return [
            name
            for name, skill in self.skills.items()
            if skill.skill_type == skill_type
        ]

    def get_skill_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a skill."""
        skill = self.skills.get(name)
        if not skill:
            return None

        return {
            "name": skill.name,
            "type": skill.skill_type.value,
            "execution_mode": skill.execution_mode.value,
            "description": skill.description,
            "criterion": skill.criterion,
            "timeout": skill.timeout,
            "requires_env": skill.requires_env,
            "criterion": skill.criterion,
            "function_name": skill.function.__name__,  # function object itself is not easily serializable
        }

    def get_skill_descriptions(self) -> str:
        """
        Get a dictionary mapping skill names to their function docstrings.
        This is useful for providing detailed information about what each skill does,
        often used for generating prompts or help texts.
        The docstring is retrieved from the skill's underlying function.
        """

        formated_descriptions = [
            f"skill name: {name}: \n skill desc:\n{item.description}\n\n"
            for name, item in self.skills.items()
        ]

        return "".join(formated_descriptions)

    def discover_skills(self, skills_directory: str = "../skills"):
        """Automatically discover and register skills from a directory."""
        try:
            # Get the skills directory path relative to this file
            current_dir = os.path.dirname(
                os.path.abspath(__file__)
            )  # Use abspath for reliability
            skills_path = os.path.join(current_dir, skills_directory)

            if not os.path.exists(skills_path):
                print(
                    f"[SkillRegistry] Skills directory not found: {skills_path}"
                )
                return

            # Import all Python files in the skills directory
            for filename in os.listdir(skills_path):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = f"robot_brain_system.skills.{filename[:-3]}"  # Make module name unique
                    file_path = os.path.join(skills_path, filename)
                    try:
                        spec = importlib.util.spec_from_file_location(
                            module_name, file_path
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            # Add to sys.modules to handle relative imports within skills if any
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)

                            # Skills are registered by @skill_register decorator now
                            # self._register_skills_from_module(module) # No longer needed if decorator handles it

                    except Exception as e:
                        print(
                            f"[SkillRegistry] Failed to import {file_path}: {e}"
                        )
                        import traceback

                        traceback.print_exc()

        except Exception as e:
            print(f"[SkillRegistry] Error discovering skills: {e}")
            import traceback

            traceback.print_exc()

    # _register_skills_from_module is effectively replaced by the decorator's direct registration

    def clear_registry(self):
        """Clear all registered skills."""
        self.skills.clear()
        # self.skill_instances.clear()
        print("[SkillRegistry] Cleared all registered skills")


# Global skill registry instance
_global_skill_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get the global skill registry instance."""
    global _global_skill_registry
    if _global_skill_registry is None:
        _global_skill_registry = SkillRegistry()
    return _global_skill_registry


def reset_skill_registry():
    """Reset the global skill registry."""
    global _global_skill_registry
    _global_skill_registry = None


# Skill registration decorator
def skill_register(
    skill_type: SkillType,
    execution_mode: ExecutionMode,
    name: str | None = None,
    description: str = "",
    criterion: dict[str,str] = {"successed":"Skill goal successfully completed", "failed":"The current status shows that the skill execution result is seriously deviated from the Skill goal"},
    timeout: Optional[float] = None,
    requires_env: bool = False,
    criterion: Dict[str, str] | None = None,  # 新增参数
):
    """Decorator to register a function as a skill."""

    def decorator(func):
        # Automatically register the skill with the global registry
        registry = get_skill_registry()
        registry.register_skill(
            skill_type=skill_type,
            execution_mode=execution_mode,
            name=name,
            function=func,
            description=description,
            criterion=criterion,
            timeout=timeout,
            requires_env=requires_env,
            criterion=criterion,
        )
        return func

    return decorator


@dataclass
class SkillExecutionContext:  # Kept for potential future use, but SkillExecutor handles state now
    """Context for skill execution."""

    skill_name: str
    parameters: Dict[str, Any]
    simulator: Optional[Any] = (
        None  # In the new model, this is the 'env' passed to skill
    )
    timeout: Optional[float] = None
    status: SkillStatus = SkillStatus.NOT_STARTED
    result: Optional[Any] = None
    error: Optional[str] = None


class SkillExecutor:
    """Executes skills with direct environment interaction, typically within a simulator subprocess."""

    def __init__(self, skill_registry: SkillRegistry):
        self.registry = skill_registry
        self.current_skill_generator: Optional[Generator] = None
        self.current_skill_name: Optional[str] = None
        self.current_skill_params: Optional[Dict[str, Any]] = None
        self.status: SkillStatus = SkillStatus.IDLE  # Using SkillStatus enum

    def execute_skill(
        self, skill_name: str, parameters: Dict[str, Any], env: Optional[Any]
    ) -> bool:
        """
        Execute a skill completely (blocking within the context it's called).
        If it's a generator skill, it runs the generator to completion.
        """
        skill_def = self.registry.get_skill(skill_name)
        if not skill_def:
            print(f"[SkillExecutor] Skill '{skill_name}' not found")
            self.status = SkillStatus.FAILED
            return False

        if skill_def.requires_env and env is None:
            print(
                f"[SkillExecutor] Skill '{skill_name}' requires an environment but none was provided."
            )
            self.status = SkillStatus.FAILED
            return False

        print(
            f"[SkillExecutor] Executing skill (blocking): {skill_name} with params {parameters}"
        )
        self.status = SkillStatus.RUNNING
        self.current_skill_name = skill_name
        self.current_skill_params = parameters

        try:
            args_for_skill = []
            if skill_def.requires_env:
                args_for_skill.append(env)
            args_for_skill.append(parameters)

            if skill_def.execution_mode == ExecutionMode.GENERATOR:
                generator = skill_def.function(*args_for_skill)
                skill_result = True  # Default to true unless StopIteration provides otherwise
                while True:
                    try:
                        next(generator)
                    except StopIteration as e:
                        skill_result = getattr(
                            e, "value", True
                        )  # Capture return value
                        break
                print(
                    f"[SkillExecutor] Generator skill {skill_name} completed. Result: {skill_result}"
                )
                self.status = (
                    SkillStatus.COMPLETED
                    if skill_result
                    else SkillStatus.FAILED
                )
                return bool(skill_result)

            elif skill_def.execution_mode == ExecutionMode.DIRECT:
                result = skill_def.function(*args_for_skill)
                print(
                    f"[SkillExecutor] Direct skill {skill_name} completed. Result: {result}"
                )
                self.status = (
                    SkillStatus.COMPLETED if result else SkillStatus.FAILED
                )
                return bool(result)
            else:
                print(
                    f"[SkillExecutor] Unknown execution mode for skill {skill_name}"
                )
                self.status = SkillStatus.FAILED
                return False
        except Exception as e:
            print(f"[SkillExecutor] Error executing skill {skill_name}: {e}")
            import traceback

            traceback.print_exc()
            self.status = SkillStatus.FAILED
            return False
        finally:
            self.current_skill_name = None
            self.current_skill_params = None
            if (
                self.status == SkillStatus.RUNNING
            ):  # If it was interrupted or errored out before completion
                self.status = SkillStatus.IDLE

    def start_skill(
        self, skill_name: str, parameters: Dict[str, Any], env: Optional[Any]
    ) -> bool:
        """Start a skill execution (non-blocking for generator skills)."""
        if self.is_running():
            print(
                f"[SkillExecutor] Another skill '{self.current_skill_name}' is already running. Terminating it."
            )
            self.terminate_current_skill()

        skill_def = self.registry.get_skill(skill_name)
        if not skill_def:
            print(f"[SkillExecutor] Skill '{skill_name}' not found")
            self.status = SkillStatus.FAILED
            return False

        if skill_def.requires_env and env is None:
            print(
                f"[SkillExecutor] Skill '{skill_name}' requires an environment but none was provided."
            )
            self.status = SkillStatus.FAILED
            return False

        print(
            f"[SkillExecutor] Starting skill (non-blocking): {skill_name} with params {parameters}"
        )

        try:
            args_for_skill = dict()  # Use a dict to collect args
            if skill_def.requires_env:
                args_for_skill["env"] = env  # Add environment if required
            args_for_skill.update(parameters)  # Add parameters

            if skill_def.execution_mode == ExecutionMode.GENERATOR:
                self.current_skill_generator = skill_def.function(
                    **args_for_skill
                )
                self.current_skill_name = skill_name
                self.current_skill_params = parameters
                self.status = SkillStatus.RUNNING
                print(f"[SkillExecutor] Started generator skill: {skill_name}")
                return True
            elif skill_def.execution_mode == ExecutionMode.DIRECT:
                # Direct execution is inherently blocking, so it completes immediately
                result = skill_def.function(*args_for_skill)
                print(
                    f"[SkillExecutor] Direct skill {skill_name} executed. Result: {result}"
                )
                self.status = (
                    SkillStatus.COMPLETED if result else SkillStatus.FAILED
                )
                # No ongoing generator for direct skills
                self.current_skill_name = None
                self.current_skill_params = None
                return bool(result)
            else:
                print(
                    f"[SkillExecutor] Unknown execution mode for skill {skill_name}"
                )
                self.status = SkillStatus.FAILED
                return False
        except Exception as e:
            print(f"[SkillExecutor] Error starting skill {skill_name}: {e}")
            import traceback

            traceback.print_exc()
            self.status = SkillStatus.FAILED
            return False

    def step(self) -> Optional[bool]:
        """
        Step the current non-blocking (generator) skill execution.
        Returns:
            True if the skill is still running.
            False if the skill completed or failed in this step.
            None if no skill was running.
        """
        if not self.is_running() or self.current_skill_generator is None:
            return None

        try:
            return next(self.current_skill_generator)
        except StopIteration as e:
            result = getattr(e, "value", True)
            print(
                f"[SkillExecutor] Skill {self.current_skill_name} completed. Result: {result}"
            )
            self.status = (
                SkillStatus.COMPLETED if result else SkillStatus.FAILED
            )
            self._reset_current_skill_state()
            return False  # Completed
        except Exception as e:
            print(
                f"[SkillExecutor] Error stepping skill {self.current_skill_name}: {e}"
            )
            import traceback

            traceback.print_exc()
            self.status = SkillStatus.FAILED
            self._reset_current_skill_state()
            return False  # Failed

    def is_running(self) -> bool:
        """Check if a non-blocking skill is currently running."""
        return (
            self.status == SkillStatus.RUNNING
            and self.current_skill_generator is not None
        )

    def terminate_current_skill(self) -> bool:
        """Terminate the current non-blocking skill execution."""
        if self.current_skill_generator is not None:
            print(
                f"[SkillExecutor] Terminating skill: {self.current_skill_name}"
            )
            try:
                self.current_skill_generator.close()
            except GeneratorExit:
                pass  # Expected when closing a generator
            except Exception as e:
                print(
                    f"[SkillExecutor] Error closing generator for {self.current_skill_name}: {e}"
                )

            self._reset_current_skill_state()
            self.status = (
                SkillStatus.INTERRUPTED
            )  # Or IDLE, depending on desired state after termination
            return True
        return False

    def _reset_current_skill_state(self):
        self.current_skill_generator = None
        self.current_skill_name = None
        self.current_skill_params = None

    def get_status_info(
        self,
    ) -> Dict[
        str, Any
    ]:  # Renamed from get_status to avoid conflict if used elsewhere
        """Get current execution status information."""
        return {
            "status": self.status.value,
            "current_skill": self.current_skill_name,
            "skill_params": self.current_skill_params,
            "is_running": self.is_running(),
        }

    def reset_executor(self):  # Renamed from reset
        """Reset the executor to initial state."""
        self.terminate_current_skill()
        self.status = SkillStatus.IDLE
        print("[SkillExecutor] Executor reset.")


if __name__ == "__main__":
    print("Skill Exector Test")
