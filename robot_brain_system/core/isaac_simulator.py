"""
Isaac Lab simulator interface.
Runs Isaac simulation in a subprocess with direct environment access
and in-process skill execution.
"""

import multiprocessing as mp
import time
import torch  # Keep for type hinting if skills use it, though actions are numpy
from typing import Dict, Any, Optional, List, Tuple
from multiprocessing.connection import Pipe, Connection
import traceback
import os
import numpy as np
import sys  # For path manipulation if necessary for skill discovery

# robot_brain_system imports
from .skill_manager import SkillExecutor, get_skill_registry, SkillRegistry
from .types import (
    Action,
    Observation,
)  # Assuming these are used for IPC if needed


def dynamic_set_attr(obj: object, kwargs: dict, path: list):
    """Dynamically set attributes on an object from a nested dictionary."""
    if kwargs is None:
        return

    for k, v in kwargs.items():
        if hasattr(obj, k):
            attr = getattr(obj, k)
            if isinstance(v, dict) and hasattr(attr, "__dict__"):
                next_path = path.copy()
                next_path.append(k)
                dynamic_set_attr(attr, v, next_path)
            else:
                try:
                    current_val = getattr(obj, k)
                    if isinstance(
                        current_val, (int, float, bool, str)
                    ) and not isinstance(v, type(current_val)):
                        # Type conversion if needed
                        v = type(current_val)(v)
                    setattr(obj, k, v)
                    print(
                        f"Set {'.'.join(path + [k])} from {getattr(obj, k)} to {v}"
                    )
                except Exception as e:
                    print(
                        f"Error setting attribute {'.'.join(path + [k])}: {e}"
                    )
        else:
            print(f"Warning: Attribute {k} not found in {'.'.join(path)}")


class IsaacSimulator:
    """Isaac Lab simulator with subprocess execution for direct environment access."""

    def __init__(
        self,
        scene_cfg_path: Optional[str] = None,
        sim_config: Optional[Dict[str, Any]] = None,
    ):
        self.scene_cfg_path = (
            scene_cfg_path  # Example: path to a scene config file
        )
        self.sim_config = sim_config or {}  # General sim configuration
        self.process: Optional[mp.Process] = None
        self.parent_conn: Optional[Connection] = None
        # child_conn is only used in the subprocess context
        self.is_running = False
        self.is_initialized = False  # Added initialization status
        self.device = self.sim_config.get(
            "device", "cpu"
        )  # Get device from config
        self.num_envs = self.sim_config.get(
            "num_envs", 1
        )  # Get num_envs from config

    def initialize(
        self,
    ) -> bool:  # Changed from start() to initialize() for consistency
        """Start the Isaac simulation subprocess and initialize it."""
        if self.is_initialized:
            print(
                "[IsaacSimulator] Simulator already initialized and running."
            )
            return True

        try:
            # Create communication pipe
            self.parent_conn, child_conn = Pipe()

            # Start subprocess
            self.process = mp.Process(
                target=self._isaac_simulation_entry,
                args=(
                    child_conn,
                    self.scene_cfg_path,
                    self.sim_config,
                ),  # Pass sim_config
            )
            self.process.daemon = True  # Ensure subprocess closes with parent
            self.process.start()

            # Wait for initialization signal from subprocess
            if self.parent_conn.poll(
                timeout=60
            ):  # Increased timeout for Isaac Lab init
                response = self.parent_conn.recv()
                if response.get("status") == "ready":
                    self.is_running = True
                    self.is_initialized = True
                    # Store env info if sent back
                    self.action_space_info = response.get("action_space")
                    self.observation_space_info = response.get(
                        "observation_space"
                    )
                    print(
                        "[IsaacSimulator] Simulation subprocess started and environment ready."
                    )
                    return True
                else:
                    error_msg = response.get(
                        "error", "Unknown initialization error"
                    )
                    print(
                        f"[IsaacSimulator] Subprocess initialization failed: {error_msg}"
                    )
                    self._cleanup_process()
                    return False
            else:
                print(
                    "[IsaacSimulator] Simulation subprocess initialization timeout."
                )
                self._cleanup_process()
                return False

        except Exception as e:
            print(
                f"[IsaacSimulator] Failed to start simulation subprocess: {e}"
            )
            traceback.print_exc()
            self._cleanup_process()
            return False

    def _cleanup_process(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.parent_conn:
            self.parent_conn.close()
        self.process = None
        self.parent_conn = None
        self.is_running = False
        self.is_initialized = False

    def shutdown(self):  # Changed from stop() to shutdown()
        """Stop the Isaac simulation subprocess."""
        print("[IsaacSimulator] Initiating shutdown...")
        if not self.is_running and not self.is_initialized:
            print("[IsaacSimulator] Already shutdown or not started.")
            return

        try:
            if self.parent_conn:
                self.parent_conn.send({"command": "shutdown"})
                # Wait for acknowledgment or timeout
                if self.parent_conn.poll(timeout=10):
                    ack = self.parent_conn.recv()
                    print(
                        f"[IsaacSimulator] Shutdown acknowledged by subprocess: {ack}"
                    )
                else:
                    print(
                        "[IsaacSimulator] No shutdown acknowledgment from subprocess."
                    )

            self._cleanup_process()
            print("[IsaacSimulator] Simulation shutdown completed.")

        except Exception as e:
            print(f"[IsaacSimulator] Error during simulation shutdown: {e}")
            traceback.print_exc()
            # Force cleanup if error during graceful shutdown
            self._cleanup_process()

    def _send_command_and_recv(
        self, command: Dict[str, Any], timeout: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        if not self.is_initialized or not self.parent_conn:
            print(
                "[IsaacSimulator] Simulator not initialized or connection lost."
            )
            return {"success": False, "error": "Simulator not initialized"}
        try:
            self.parent_conn.send(command)
            if self.parent_conn.poll(timeout):
                response = self.parent_conn.recv()
                if "error" in response:
                    print(
                        f"[IsaacSimulator] Error from subprocess for command {command.get('command')}: {response['error']}"
                    )
                return response
            else:
                print(
                    f"[IsaacSimulator] Timeout waiting for response to command: {command.get('command')}"
                )
                return {"success": False, "error": "Timeout"}
        except (EOFError, BrokenPipeError) as e:
            print(
                f"[IsaacSimulator] Communication pipe broken: {e}. Shutting down simulator."
            )
            self.shutdown()
            return {"success": False, "error": f"Pipe broken: {e}"}
        except Exception as e:
            print(
                f"[IsaacSimulator] Error sending/receiving command {command.get('command')}: {e}"
            )
            return {"success": False, "error": str(e)}

    def execute_skill_blocking(
        self, skill_name: str, parameters: Dict[str, Any]
    ) -> bool:
        """Execute a skill in the simulation subprocess (blocks until skill completion)."""
        response = self._send_command_and_recv(
            {
                "command": "execute_skill_blocking",
                "skill_name": skill_name,
                "parameters": parameters,
            },
            timeout=parameters.get("timeout", 60.0) + 5.0,
        )  # Add buffer to timeout
        return response.get("success", False) if response else False

    def start_skill_non_blocking(
        self, skill_name: str, parameters: Dict[str, Any]
    ) -> bool:
        """Start a skill execution in the simulation subprocess (non-blocking)."""
        response = self._send_command_and_recv(
            {
                "command": "start_skill_non_blocking",
                "skill_name": skill_name,
                "parameters": parameters,
            }
        )
        return response.get("success", False) if response else False

    def get_skill_executor_status(self) -> Dict[str, Any]:
        """Get current skill executor status from the subprocess."""
        response = self._send_command_and_recv(
            {"command": "get_skill_executor_status"}
        )
        return response or {"status": "error", "error": "No response"}

    def terminate_current_skill(self) -> bool:
        """Terminate current skill execution in the subprocess."""
        response = self._send_command_and_recv(
            {"command": "terminate_current_skill"}
        )
        return response.get("success", False) if response else False

    def get_observation(self) -> Optional[Observation]:
        """Requests and retrieves the current observation from the simulator subprocess."""
        response = self._send_command_and_recv({"command": "get_observation"})
        if response and response.get("success"):
            obs_data = response.get("observation_data")
            # Assuming obs_data is a dict that can initialize Observation
            return Observation(**obs_data) if obs_data else None
        return None

    def step_env(
        self, action: Action
    ) -> Optional[Tuple[Observation, float, bool, bool, Dict[str, Any]]]:
        """Manually step the environment with a given action. For testing/direct control."""
        response = self._send_command_and_recv(
            {
                "command": "step_env",
                "action_data": action.data.tolist()
                if hasattr(action.data, "tolist")
                else action.data,  # Convert numpy to list
                "action_metadata": action.metadata,
            }
        )
        if response and response.get("success"):
            obs = (
                Observation(**response["observation_data"])
                if response.get("observation_data")
                else None
            )
            return (
                obs,
                response.get("reward"),
                response.get("terminated"),
                response.get("truncated"),
                response.get("info"),
            )
        return None

    def reset_env(self) -> Optional[Observation]:
        """Reset the simulation environment(s)."""
        response = self._send_command_and_recv({"command": "reset_env"})
        if response and response.get("success"):
            obs_data = response.get("observation_data")
            return Observation(**obs_data) if obs_data else None
        return None

    @staticmethod
    def _isaac_simulation_entry(
        child_conn: Connection,
        scene_cfg_path: Optional[str],
        sim_config: Dict[str, Any],
    ):
        """Entry point for Isaac simulation subprocess with direct environment access."""
        env = None
        skill_executor = None
        try:
            from isaaclab.app import AppLauncher
            import gymnasium as gym
            import os

            os.environ["DISPLAY"] = ":0"

            app_launcher_params = {
                "task": sim_config.get(
                    "task", "Isaac-Frank-LunarBase-Direct-v0"
                ),
                "device": sim_config.get(
                    "device", "cuda:0"
                ),  # AppLauncher might use this for 'sim_device' default
                "num_envs": sim_config.get("num_envs", 1),
                "disable_fabric": sim_config.get("disable_fabric", False),
                "mode": sim_config.get(
                    "mode", 1
                ),  # Make sure this value matches AppLauncher expectations
                "env_config_file": sim_config.get("env_config_file"),
                # Arguments AppLauncher specifically uses/pops (add more as needed):
                "enable_cameras": sim_config.get("enable_cameras", True),
                "headless": sim_config.get("headless", False),
                # "livestream": sim_config.get(
                #     "livestream", False
                # ),  # This was the one causing the error
                # "sim_device": sim_config.get(
                #     "sim_device", sim_config.get("device", "cuda:0")
                # ),  # AppLauncher often uses 'sim_device'
                # "cpu": sim_config.get("cpu", False),
                # "physics_gpu": sim_config.get("physics_gpu", -1),
                # "graphics_gpu": sim_config.get("graphics_gpu", -1),
                # "pipeline": sim_config.get("pipeline", "gpu"),
                # "fabric_gpu": sim_config.get("fabric_gpu", -1),
                # "kit_app": sim_config.get("kit_app", None),
                # "enable_ros": sim_config.get("enable_ros", False),
                # "ros_domain_id": sim_config.get("ros_domain_id", 0),
                # "verbosity": sim_config.get("verbosity", "info"),
                # "build_path": sim_config.get("build_path", None),
                # # Add any other arguments AppLauncher defines in its command-line parsing
            }

            # Launch Isaac
            app_launcher = AppLauncher(app_launcher_params)
            simulation_app = app_launcher.app

            # Import remaining modules after AppLauncher
            from isaaclab_tasks.utils import parse_env_cfg
            from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
            from isaaclab.sim.simulation_context import SimulationContext
            from argparse import Namespace
            import yaml

            print("[IsaacSubprocess] Isaac App launched.")
            cli_args = Namespace(
                **app_launcher_params
            )  # Create Namespace from the same comprehensive dict

            print("[Isaac Process] Initializing environment...")
            env_cfg = parse_env_cfg(
                cli_args.task,
                device=cli_args.device,
                num_envs=cli_args.num_envs,
                use_fabric=not cli_args.disable_fabric,
            )
            # Load custom config if provided
            if cli_args.env_config_file:
                with open(cli_args.env_config_file, "r") as f:
                    env_new_cfg = yaml.safe_load(f)
                    dynamic_set_attr(env_cfg, env_new_cfg, path=["env_cfg"])

            env = gym.make(cli_args.task, cfg=env_cfg)
            _env: DirectRLEnv|ManagerBasedRLEnv = env.unwrapped
            _sim = _env.sim
            _sim.set_camera_view(
                (-2.08, -1.12, 3.95),
                (0.6, -2.0, 2.818),
            )
            obs, info = env.reset()
            print(
                f"[Isaac Process] Environment '{cli_args.task}' created. Device: {_env.device}"
            )
            # Initialize skill system
            # Ensure skills are discoverable from the subprocess
            # This might require setting sys.path or using absolute package imports for skills
            # For simplicity, assuming skills are in a package accessible via robot_brain_system.skills

            import robot_brain_system.skills  # This should trigger skill registration

            skill_registry = (
                get_skill_registry()
            )  # Gets the global one, populated by decorators
            # skill_registry.discover_skills() # Not needed if decorators auto-register and skills are imported
            print(
                f"[IsaacSubprocess] Found {len(skill_registry.list_skills())} skills."
            )

            skill_executor = SkillExecutor(skill_registry)
            print("[IsaacSubprocess] SkillExecutor initialized.")

            # Send ready signal with env info
            action_space_info = {
                "shape": env.action_space.shape,
                "dtype": str(env.action_space.dtype),
            }  # Basic info
            obs_space_info = {
                "shape": env.observation_space.shape,
                "dtype": str(env.observation_space.dtype),
            }  # Basic info

            child_conn.send(
                {
                    "status": "ready",
                    "action_space": action_space_info,
                    "observation_space": obs_space_info,
                }
            )
            print("[IsaacSubprocess] Sent 'ready' to parent.")

            # Main loop
            active = True
            
            def _get_observation():
                if isinstance(_env, ManagerBasedRLEnv):
                    return _env.observation_manager.compute
                elif isinstance(_env, DirectRLEnv):
                    return _env._get_observations
                else:
                    raise TypeError(f"Unsupport unwrapped enviroment type: {type(_env)}")
            while active:
                try:
                    # Handle commands from parent process
                    if child_conn.poll(0.001):  # Non-blocking check for 1ms
                        command_data = child_conn.recv()
                        cmd = command_data.get("command")

                        if cmd == "shutdown":
                            print(
                                "[IsaacSubprocess] Received shutdown command."
                            )
                            active = False
                            child_conn.send({"status": "shutdown_ack"})
                            break
                        elif cmd == "execute_skill_blocking":
                            skill_name = command_data["skill_name"]
                            params = command_data["parameters"]
                            print(
                                f"[IsaacSubprocess] Executing skill (blocking): {skill_name}"
                            )
                            success = skill_executor.execute_skill(
                                skill_name, params, env
                            )
                            child_conn.send(
                                {
                                    "success": success,
                                    "status": skill_executor.status.value,
                                }
                            )
                        elif cmd == "start_skill_non_blocking":
                            skill_name = command_data["skill_name"]
                            params = command_data["parameters"]
                            print(
                                f"[IsaacSubprocess] Starting skill (non-blocking): {skill_name}"
                            )
                            success = skill_executor.start_skill(
                                skill_name, params, env
                            )
                            child_conn.send(
                                {
                                    "success": success,
                                    "status": skill_executor.status.value,
                                }
                            )
                        elif cmd == "get_skill_executor_status":
                            child_conn.send(skill_executor.get_status_info())
                        elif cmd == "terminate_current_skill":
                            success = skill_executor.terminate_current_skill()
                            child_conn.send({"success": success})
                        elif cmd == "get_observation":
                            # This should get the latest observation from the environment
                            # env.get_observations() or similar if available, or from last step
                            # For now, let's assume env.reset() or env.step() populates an obs property
                            # This part needs to align with how your Isaac Lab env exposes observations
                            # A common pattern is that step() returns it. We might need to store it.
                            # For now, we'll simulate getting it via a direct call if possible
                            # This needs careful implementation based on Isaac Lab Env specifics
                            current_obs_dict = _get_observation()
                            # Better: store obs from last step/reset call
                            obs_payload = {
                                "data": current_obs_dict, # TODO test
                                "metadata": "None",
                                "timestamp": time.time(),
                            }
                            child_conn.send(
                                {
                                    "success": True,
                                    "observation_data": obs_payload,
                                }
                            )

                        elif cmd == "step_env":
                            action_data_np = np.array(
                                command_data["action_data"]
                            )
                            action_tensor = (
                                torch.from_numpy(action_data_np)
                                .to(env.device)
                                .unsqueeze(0)
                            )  # Batch for single env

                            obs_dict, reward, terminated, truncated, info = (
                                env.step(action_tensor)
                            )

                            obs_payload = obs_dict # TODO
                            child_conn.send(
                                {
                                    "success": True,
                                    "observation_data": obs_payload,
                                    "reward": float(
                                        reward.item()
                                    ),  # Assuming single env reward
                                    "terminated": bool(terminated.item()),
                                    "truncated": bool(truncated.item()),
                                    "info": info,  # This might need serialization too
                                }
                            )
                        elif cmd == "reset_env":
                            obs_dict, info = env.reset()
                            obs_payload = obs_dict  # TODO
                            child_conn.send(
                                {
                                    "success": True,
                                    "observation_data": obs_payload,
                                    "info": info,
                                }
                            )
                        else:
                            child_conn.send(
                                {
                                    "error": f"Unknown command: {cmd}",
                                    "success": False,
                                }
                            )

                    # Step non-blocking skill if one is running
                    if skill_executor.is_running():
                        skill_executor.step()  # env is passed during start_skill

                    # Step the simulation app itself to keep it responsive
                    # Only if not paused by a skill that needs fine-grained control
                    # if (
                    #     not skill_executor.is_running()
                    # ):  # Or if skill yields control for sim step
                    #     if simulation_app.is_running():
                    #         simulation_app.update()  # Essential for Isaac Sim
                    #     else:
                    #         active = False  # Sim app stopped externally

                    # Small delay to prevent tight loop if no commands and no active skill
                    if (
                        not child_conn.poll(0)
                        and not skill_executor.is_running()
                    ):
                        time.sleep(0.001)

                except EOFError:  # Pipe closed by parent
                    print("[IsaacSubprocess] Parent connection closed.")
                    active = False
                except BrokenPipeError:
                    print("[IsaacSubprocess] Pipe broken.")
                    active = False
                except Exception as loop_e:
                    print(f"[IsaacSubprocess] Error in main loop: {loop_e}")
                    traceback.print_exc()
                    try:
                        child_conn.send(
                            {"error": str(loop_e), "success": False}
                        )
                    except:
                        pass  # Can't send if pipe is broken
                    # Decide if to continue or break based on error severity
                    # For now, continue unless it's a pipe error handled above
                    time.sleep(0.1)  # Avoid fast error loop

        except Exception as e:
            print(
                f"[IsaacSubprocess] Subprocess initialization or critical error: {e}"
            )
            traceback.print_exc()
            try:
                child_conn.send({"status": "error", "error": str(e)})
            except:
                pass  # If conn is already broken

        finally:
            print("[IsaacSubprocess] Cleaning up...")
            if env is not None:
                try:
                    env.close()
                    print("[IsaacSubprocess] Environment closed.")
                except Exception as e_close:
                    print(
                        f"[IsaacSubprocess] Error closing environment: {e_close}"
                    )
            if "simulation_app" in locals() and simulation_app.is_running():
                try:
                    simulation_app.close()  # Close the Isaac Sim application
                    print("[IsaacSubprocess] Isaac App closed.")
                except Exception as e_sim_close:
                    print(
                        f"[IsaacSubprocess] Error closing Isaac Sim App: {e_sim_close}"
                    )
            if child_conn:
                child_conn.close()
            print("[IsaacSubprocess] Terminated.")
