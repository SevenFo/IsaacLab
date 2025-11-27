"""
Isaac Lab simulator interface.
Runs Isaac simulation in a subprocess with direct environment access
and in-process skill execution.
"""

import multiprocessing as mp
import threading
from typing import (
    Dict,
    Any,
    Optional,
    Tuple,
)
from multiprocessing.connection import Pipe, Connection
import traceback
import pickle  # For pickle.UnpicklingError
import math

# robot_brain_system imports
from robot_brain_system.core.types import (
    Action,
    Observation,
    SkillStatus,
)  # Assuming these are used for IPC if needed
from robot_brain_system.utils.retry_utils import retry

mp.set_start_method("spawn", force=True)  # Ensure spawn method for subprocesses


class IsaacSimulator:
    """Isaac Lab simulator with subprocess execution for direct environment access."""

    def __init__(
        self,
        sim_config: Optional[Dict[str, Any]] = None,
    ):
        self.sim_config = sim_config or {}  # General sim configuration
        self.process: Optional[mp.Process] = None
        self.parent_conn: Optional[Connection] = None
        # child_conn is only used in the subprocess context
        self.is_running = False
        self.is_initialized = False  # Added initialization status
        self.device = self.sim_config.get("device", "cpu")  # Get device from config
        self.num_envs = self.sim_config.get("num_envs", 1)  # Get num_envs from config
        self._command_lock = threading.Lock()

    def initialize(
        self,
    ) -> bool:  # Changed from start() to initialize() for consistency
        """Start the Isaac simulation subprocess and initialize it."""
        if self.is_initialized:
            print("[IsaacSimulator] Simulator already initialized and running.")
            return True

        try:
            # Create communication pipe
            self.parent_conn, child_conn = Pipe()

            # Start subprocess - directly use _subprocess_entry as target
            self.process = mp.Process(
                target=self._subprocess_entry,
                args=(
                    child_conn,
                    self.sim_config,
                ),  # Pass sim_config
            )
            self.process.daemon = True  # Ensure subprocess closes with parent
            self.process.start()

            # Wait for initialization signal from subprocess
            if self.parent_conn.poll(
                timeout=480
            ):  # Increased timeout for Isaac Lab init
                response = self.parent_conn.recv()
                if response.get("status") == "ready":
                    self.is_running = True
                    self.is_initialized = True
                    # Store env info if sent back
                    self.action_space_info = response.get("action_space")
                    self.observation_space_info = response.get("observation_space")
                    print(
                        "[IsaacSimulator] Simulation subprocess started and environment ready."
                    )
                    return True
                else:
                    error_msg = response.get("error", "Unknown initialization error")
                    print(
                        f"[IsaacSimulator] Subprocess initialization failed: {error_msg}"
                    )
                    self._cleanup_process()
                    return False
            else:
                print("[IsaacSimulator] Simulation subprocess initialization timeout.")
                self._cleanup_process()
                return False

        except Exception as e:
            print(f"[IsaacSimulator] Failed to start simulation subprocess: {e}")
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

    def get_skill_registry_from_sim(self):
        """Get the skill registry from the subprocess."""
        response = self._send_command_and_recv({"command": "get_skill_registry"})
        if response and response.get("success"):
            print(
                f"[IsaacSimulator] Retrieved skill registry with {len(response.get('skill_registry').list_skills())} skills."
            )
            return response.get("skill_registry")
        return None

    @staticmethod
    def _should_retry_command(response: Optional[Dict[str, Any]]) -> bool:
        if isinstance(response, dict) and response.get("success") is False:
            error_message = response.get("error", "").lower()
            # Retry specifically on "Timeout"
            if "timeout" in error_message:
                print(
                    f"[IsaacSimulator._should_retry_command] Detected retryable error: {error_message}"
                )
                return True
            # Do NOT retry on "Pipe broken" because shutdown() is already called.
            # if "pipe broken" in error_message:
            #     return True # Careful: if shutdown() is called, retrying might be complex
        return False

    @retry(
        max_attempts=3,
        delay_seconds=0.5,  # Start with a shorter delay for local pipes
        backoff_factor=1.5,
        max_delay_seconds=5.0,
        exceptions_to_retry=(
            pickle.UnpicklingError,
            TimeoutError,
        ),  # TimeoutError if poll() itself times out, pickle for bad data
        retry_on_return_value_condition=_should_retry_command,  # Pass the method itself
    )
    def _send_command_and_recv(
        self, command: Dict[str, Any], timeout: float = 60
    ) -> Optional[Dict[str, Any]]:
        acquired = self._command_lock.acquire(timeout=60)
        if acquired:
            if not self.is_initialized or not self.parent_conn:
                print("[IsaacSimulator] Simulator not initialized or connection lost.")
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
            finally:
                self._command_lock.release()
        else:
            print("[IsaacSimulator] Failed to acquire command lock within timeout.")
            return {
                "success": False,
                "error": "Command lock acquisition failed",
            }

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
        self, skill_name: str, parameters: Dict[str, Any], timeout: float = 60.0
    ) -> bool:
        """Start a skill execution in the simulation subprocess (non-blocking)."""
        response = self._send_command_and_recv(
            {
                "command": "start_skill_non_blocking",
                "skill_name": skill_name,
                "parameters": parameters,
            },
            timeout=timeout,
        )
        return response.get("success", False) if response else False

    def get_skill_executor_status(self) -> Dict[str, Any]:
        """Get current skill executor status from the subprocess."""
        response = self._send_command_and_recv({"command": "get_skill_executor_status"})
        return response or {"status": "error", "error": "No response"}

    def terminate_current_skill(
        self, skill_status: SkillStatus = SkillStatus.INTERRUPTED, status_info: str = ""
    ) -> bool:
        """Terminate current skill execution in the subprocess."""
        response = self._send_command_and_recv(
            {
                "command": "terminate_current_skill",
                "skill_status": skill_status,
                "status_info": status_info,
            }
        )
        return response.get("success", False) if response else False

    def pause_current_skill(self) -> bool:
        """Terminate current skill execution in the subprocess."""
        response = self._send_command_and_recv(
            {"command": "change_current_skill_status", "status": SkillStatus.PAUSED}
        )
        return response.get("success", False) if response else False

    def recovery_current_skill(self) -> bool:
        """Terminate current skill execution in the subprocess."""
        response = self._send_command_and_recv(
            {"command": "change_current_skill_status", "status": SkillStatus.RUNNING}
        )
        return response.get("success", False) if response else False

    def change_current_skill_status(self, status: SkillStatus) -> bool:
        """Change current skill execution status in the subprocess."""
        response = self._send_command_and_recv(
            {"command": "change_current_skill_status_force", "status": status}
        )
        return response.get("success", False) if response else False

    def get_observation(self) -> Optional[list[Observation]]:
        """Requests and retrieves the current observation from the simulator subprocess."""
        response = self._send_command_and_recv({"command": "get_observation"})
        obss = []
        if response and response.get("success"):
            obs_data = response.get("observation_data")
            for obs in obs_data:
                obss.append(Observation(**obs))
                # Assuming obs_data is a dict that can initialize Observation
            return obss
        return None

    def get_current_observation(self) -> Optional[Observation]:
        response = self._send_command_and_recv({"command": "get_current_observation"})
        if response and response.get("success"):
            obs_data = response.get("observation_data")
            return Observation(**obs_data)
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

    def reset_env(self) -> bool:
        """Reset the simulation environment(s)."""
        response = self._send_command_and_recv({"command": "reset_env"})
        if response and response.get("success"):
            return True
        return False

    def cleanup_skill(self) -> bool:
        response = self._send_command_and_recv({"command": "cleanup_skill"})
        if response and response.get("success"):
            return True
        return False

    @staticmethod
    def _subprocess_entry(child_conn: Connection, sim_config: Dict[str, Any]):
        """
        Subprocess entry point that initializes Isaac Sim and runs the main simulation loop.
        This replaces the separate isaac_launcher.py file.
        """
        from omegaconf import OmegaConf

        print("[IsaacSimulator] Subprocess started. Initializing Isaac Sim...")

        try:
            from isaaclab.app import AppLauncher
        except ImportError as e:
            print(
                f"[IsaacSimulator] Critical Error: Failed to import isaaclab.app. {e}"
            )
            child_conn.send(
                {
                    "status": "error",
                    "error": "Failed to import isaaclab.app. Check Isaac Lab installation.",
                }
            )
            child_conn.close()
            return

        # Convert config to dict and create AppLauncher
        app_launcher_params = OmegaConf.to_container(sim_config, resolve=True)
        assert isinstance(app_launcher_params, dict), (
            "Config must be convertible to dict"
        )

        app_launcher = AppLauncher(app_launcher_params)

        # Now run the main simulation logic
        IsaacSimulator._isaac_simulation_entry(
            child_conn, app_launcher, app_launcher_params
        )

        print(
            "[IsaacSimulator] Main simulation logic has exited. Subprocess terminating."
        )

    @staticmethod
    def _isaac_simulation_entry(
        child_conn: Connection,
        app_launcher: Any,
        app_launcher_params,
    ):
        """Entry point for Isaac simulation subprocess with direct environment access."""
        env = None
        skill_executor = None
        simulation_app = app_launcher.app

        try:
            import gymnasium as gym
            from isaaclab_tasks.utils import parse_env_cfg
            from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
            from isaaclab.managers import TerminationTermCfg
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
            from isaaclab_tasks.manager_based.manipulation.move_rot_0923.mdp import (
                lift_eef,
                leave_spanner,
                box_open,
            )
            from argparse import Namespace
            import yaml
            import torch  # We can import torch here again if we want to be strict
            import numpy as np
            import queue
            import time
            import traceback

            from robot_brain_system.core.skill_manager import (
                SkillExecutor,
                get_skill_registry,
            )
            from robot_brain_system.utils import dynamic_set_attr
            import robot_brain_system.skills  # noqa: F401
            from robot_brain_system.skills.alice_control_skills import AliceControl

            print(
                "[IsaacSubprocess] Isaac App was launched by launcher. Continuing initialization."
            )
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
            if hasattr(env_cfg.terminations, "success"):
                env_cfg.terminations.success = None
            env_cfg.terminations.time_out = None
            # Load custom config if provided
            if hasattr(cli_args, "env_config_file"):
                with open(cli_args.env_config_file, "r") as f:
                    env_new_cfg = yaml.safe_load(f)
                    dynamic_set_attr(env_cfg, env_new_cfg, path=["env_cfg"])

            env = gym.make(cli_args.task, cfg=env_cfg)
            _env: DirectRLEnv | ManagerBasedRLEnv = env.unwrapped
            _sim = _env.sim
            _sim.set_camera_view(
                (-2.08, -1.12, 3.95),
                (0.6, -2.0, 2.818),
            )

            obs_queue = queue.Queue()
            obs, info = env.reset()

            # -- WARM-UP CORRECTION START --
            print("[IsaacSubprocess] Starting environment warm-up...")
            # Check if the unwrapped environment has an action_manager and total_action_dim
            # This is the robust way to get the action dimension for ManagerBasedRLEnv.
            if hasattr(_env, "action_manager") and hasattr(
                _env.action_manager, "total_action_dim"
            ):
                # Get the correct action dimension directly from the action manager.
                action_dim = _env.action_manager.total_action_dim

                # Create a zero action tensor with the correct shape: (num_envs, total_action_dim)
                zero_action = torch.zeros(
                    (_env.num_envs, action_dim),
                    device=_env.device,
                    dtype=torch.float32,
                )
                zero_action[..., -1] = 1  # Ensure gripper is open
                # joint InitialStateCfg not working, so we set it manually
                disable_alice = True
                if "alice" in _env.scene.keys():
                    disable_alice = False
                    init_alice_joint_position_target = torch.zeros_like(
                        _env.scene["alice"].data.joint_pos_target
                    )
                    init_alice_joint_position_target[:, :9] = torch.tensor(
                        [
                            0.0,  # D6Joint_1:0
                            math.radians(66.7),  # D6Joint_1:1
                            math.radians(50.7),  # D6Joint_1:2
                            0.0,  # D6Joint_2:0
                            math.radians(25.9),  # D6Joint_2:1
                            math.radians(-23.2),  # D6Joint_2:2
                            math.radians(-141.8),  # D6Joint_3:0
                            math.radians(-11.0),  # D6Joint_3:1
                            math.radians(-41.7),  # D6Joint_3:2
                        ],
                        device=_env.device,
                    )
                    _env.scene["alice"].set_joint_position_target(
                        init_alice_joint_position_target,
                    )
                    # write the joint state to sim to directly change the joint position at one step
                    _env.scene["alice"].write_joint_state_to_sim(
                        init_alice_joint_position_target,
                        torch.zeros_like(
                            init_alice_joint_position_target, device=_env.device
                        ),
                    )
                    marker_cfg = POSITION_GOAL_MARKER_CFG.copy()
                    marker_cfg.prim_path = "/Visuals/MoveTargetVisualizer"
                    _env.move_target_visualizer = VisualizationMarkers(marker_cfg)

                    alice_control = AliceControl()
                for i in range(cli_args.warmup_steps):
                    # Step the environment with the correctly shaped zero action
                    # alice_control.apply_action(_env)
                    obs, reward, terminated, truncated, info = env.step(zero_action)
                    print(
                        f"[IsaacSubprocess] Warm-up step {i + 1}/{cli_args.warmup_steps}, action: {zero_action}"
                    )
                    print(
                        f"[IsaacSubprocess] Warm-up step {i + 1}/{cli_args.warmup_steps}, obs: {obs['policy']['gripper_pos'].clone()}"
                    )

                print(
                    f"[IsaacSubprocess] Warm-up complete after {cli_args.warmup_steps} steps with action shape {zero_action.shape}."
                )
            else:
                print(
                    "[IsaacSubprocess] Skipping warm-up: could not determine action dimension from action_manager."
                )
            if not disable_alice:
                alice_articulator = _env.scene["alice"]
                if alice_articulator is not None:
                    print(
                        f"[IsaacSubprocess] Alice articulator joint_names: {alice_articulator.joint_names}"
                    )
            # The 'obs' variable now holds the observation from the final warm-up step.
            # -- WARM-UP CORRECTION END --

            obs_payload = {
                "data": obs,  # Assuming obs is already in a dict-like format
                "metadata": "None",
                "timestamp": time.time(),
            }  # Add metadata and timestamp
            obs_queue.put(obs_payload)
            print(f"[IsaacSubprocess] Environment reset complete, obs {obs}")
            latest_obs_payload = obs_payload  # Store latest obs
            print(
                f"[Isaac Process] Environment '{cli_args.task}' created. Device: {_env.device}"
            )
            # Initialize skill system
            # Ensure skills are discoverable from the subprocess
            # This might require setting sys.path or using absolute package imports for skills
            # For simplicity, assuming skills are in a package accessible via robot_brain_system.skills

            skill_registry = (
                get_skill_registry()
            )  # Gets the global one, populated by decorators
            print(
                f"[IsaacSubprocess] Found {len(skill_registry.list_skills())} skills."
            )

            skill_executor = SkillExecutor(skill_registry, env=env)
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
                    return _env.observation_manager.compute()
                elif isinstance(_env, DirectRLEnv):
                    return _env._get_observations()
                else:
                    raise TypeError(
                        f"Unsupport unwrapped enviroment type: {type(_env)}"
                    )

            obs_dict = obs

            success_item_map = {
                "open_box": {
                    "success_item": TerminationTermCfg(
                        func=box_open,
                    ),
                },
                "grasp_spanner": {
                    "success_item": TerminationTermCfg(
                        func=leave_spanner,
                    ),
                },
                "move_box_to_suitable_position": {
                    "success_item": TerminationTermCfg(
                        func=lift_eef,
                    ),
                },
            }
            skill_executor.skill_success_item_map = success_item_map
            while active:
                try:
                    # Handle commands from parent process
                    if child_conn.poll(0.001):  # Non-blocking check for 1ms
                        command_data = child_conn.recv()
                        cmd = command_data.get("command")

                        if cmd == "shutdown":
                            print("[IsaacSubprocess] Received shutdown command.")
                            active = False
                            child_conn.send({"status": "shutdown_ack"})
                            break
                        elif cmd == "get_skill_registry":
                            tmp = {"success": True, "skill_registry": skill_registry}
                            child_conn.send(tmp)
                        elif cmd == "execute_skill_blocking":
                            assert False, "Unsupported now!"
                        elif cmd == "start_skill_non_blocking":
                            skill_name = command_data["skill_name"]
                            params = command_data["parameters"]
                            print(
                                f"[IsaacSubprocess] Starting skill (non-blocking): {skill_name}"
                            )

                            success, obs_dict = skill_executor.initialize_skill(
                                skill_name,
                                params,
                                obs_dict=obs_dict,
                                success_item=success_item_map.get(skill_name, {}).get(
                                    "success_item", None
                                ),
                                timeout_item=success_item_map.get(skill_name, {}).get(
                                    "timeout_item", None
                                ),
                            )
                            child_conn.send(
                                {
                                    "success": success,
                                    "status": skill_executor.status.value,
                                }
                            )
                        elif cmd == "get_skill_executor_status":
                            child_conn.send(skill_executor.get_status_info())
                        elif cmd == "cleanup_skill":
                            skill_executor.cleanup_skill()
                            child_conn.send({"success": True})
                        elif cmd == "terminate_current_skill":
                            skill_status = command_data["skill_status"]
                            status_info = command_data.get("status_info", "")
                            success = skill_executor.terminate_current_skill(
                                skill_status, status_info=status_info
                            )
                            child_conn.send({"success": success})
                        elif cmd == "change_current_skill_status":
                            skill_status = command_data["status"]
                            success = skill_executor.change_current_skill_status(
                                skill_status=skill_status
                            )
                            child_conn.send({"success": success})
                        elif cmd == "change_current_skill_status_force":
                            skill_status = command_data["status"]
                            success = skill_executor._change_current_skill_status_force(
                                skill_status=skill_status
                            )
                            child_conn.send({"success": success})
                        elif cmd == "get_observation":
                            obss = []
                            try:
                                while True:
                                    obss.append(obs_queue.get_nowait())
                                    obs_queue.task_done()
                            except queue.Empty:
                                print("Queue is empty")
                            child_conn.send(
                                {
                                    "success": True,
                                    "observation_data": obss,
                                }
                            )
                        elif cmd == "get_current_observation":
                            child_conn.send(
                                {
                                    "success": True,
                                    "observation_data": latest_obs_payload,
                                    "info": info,
                                }
                            )

                        elif cmd == "step_env":
                            assert False, "Unsupported now!"
                            action_data_np = np.array(command_data["action_data"])
                            action_tensor = (
                                torch.from_numpy(action_data_np)
                                .to(env.device)
                                .unsqueeze(0)
                            )  # Batch for single env

                            obs_dict, reward, terminated, truncated, info = env.step(
                                action_tensor
                            )

                            obs_payload = {
                                "data": obs_dict,  # TODO test
                                "metadata": "None",
                                "timestamp": time.time(),
                            }
                            obs_queue.put(obs_payload)
                            latest_obs_payload = obs_payload  # Store latest obs

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
                            delattr(env.unwrapped, "switch_heavybox")
                            obs_dict, info = env.reset()
                            child_conn.send({"success": True})
                        else:
                            child_conn.send(
                                {
                                    "error": f"Unknown command: {cmd}",
                                    "success": False,
                                }
                            )

                    # Step non-blocking skill if one is running
                    if skill_executor.is_running():
                        skill_exec_result = skill_executor.step(
                            obs_dict
                        )  # env is passed during start_skill
                        obs_dict, reward, terminated, truncated, info = (
                            skill_exec_result
                        )
                        # if skill_executor.is_running():
                        if obs_dict is not None:
                            obs_dict_in = {obs_key: {} for obs_key in obs_dict.keys()}
                            obs_dict_in["policy"].update(
                                {
                                    key: val.clone()
                                    for key, val in obs_dict["policy"].items()
                                    if isinstance(val, torch.Tensor)
                                }
                            )
                            obs_payload = {
                                "data": obs_dict_in,
                                "metadata": "None",
                                "timestamp": time.time(),
                            }
                            obs_queue.put(obs_payload)
                            latest_obs_payload = obs_payload  # Store latest obs
                        else:
                            obs_dict = latest_obs_payload["data"]  # Use last obs
                        # print(
                        #     f"[IsaacSubprocess] obs_shape: { {key: val.shape for key, val in obs_dict['policy'].items() if isinstance(val, torch.Tensor)} }, "
                        # )
                    # Small delay to prevent tight loop if no commands and no active skill
                    else:
                        obs_dict = _get_observation()
                        obs_dict_in = {obs_key: {} for obs_key in obs_dict.keys()}
                        obs_dict_in["policy"].update(
                            {
                                key: val.clone()
                                for key, val in obs_dict["policy"].items()
                                if isinstance(val, torch.Tensor)
                            }
                        )
                        obs_payload = {
                            "data": obs_dict_in,
                            "metadata": "None",
                            "timestamp": time.time(),
                        }

                        latest_obs_payload = obs_payload
                        # _sim.set_render_mode(_sim.RenderMode.FULL_RENDERING)
                        # _sim.step(render=True)
                    if not child_conn.poll(0) and not skill_executor.is_running():
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
                        child_conn.send({"error": str(loop_e), "success": False})
                    except:
                        pass  # Can't send if pipe is broken
                    # Decide if to continue or break based on error severity
                    # For now, continue unless it's a pipe error handled above
                    time.sleep(0.1)  # Avoid fast error loop

        except Exception as e:
            print(f"[IsaacSubprocess] Subprocess initialization or critical error: {e}")
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
                    print(f"[IsaacSubprocess] Error closing environment: {e_close}")
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


if __name__ == "__main__":
    ...

    # print("Isaac Simulator Tets")
    # isim = IsaacSimulator(sim_config=DEVELOPMENT_CONFIG)
    # result = isim.initialize()
    # assert result, "Failed to initialize Isaac Simulator"
    # print("Isaac Simulator initialized successfully.")

    # # reset test
    # obs = isim.reset_env()
    # assert obs is not None, "Failed to reset Isaac Simulator environment"
    # print("Isaac Simulator environment reset successfully.")

    # # Get observation test
    # obs = isim.get_observation()
    # assert obs is not None, "Failed to get observation from Isaac Simulator"
    # print("Observation retrieved successfully:", obs)

    # skill_status = isim.get_skill_executor_status()
    # assert skill_status.get("status") == "idle", (
    #     f"Skill executor status should be 'idle', but got {skill_status.get('status')}"
    # )

    # result = isim.start_skill_non_blocking(
    #     "assemble_object",
    #     {
    #         "checkpoint_path": "assets/model_epoch_4000.pth",
    #         "horizon": 1000,
    #     },
    # )
    # assert result, "Failed to start skill in Isaac Simulator"
    # print("Skill started successfully.")

    # while isim.get_skill_executor_status().get("status") == "running":
    #     time.sleep(1)
    #     print("Waiting for skill to complete...")
    # skill_finish_status = isim.get_skill_executor_status()
    # print(f"Skill finished with status: {skill_finish_status.get('status', 'unknown')}")

    # # Terminate skill test
    # isim.start_skill_non_blocking(
    #     "assemble_object",
    #     {
    #         "checkpoint_path": "assets/model_epoch_4000.pth",
    #         "horizon": 1000,
    #     },
    # )
    # time.sleep(5)  # Give it some time to start
    # print("Attempting to terminate current skill...")
    # result = isim.terminate_current_skill()
    # assert result, "Failed to terminate skill in Isaac Simulator"
    # skill_status = isim.get_skill_executor_status()
    # assert skill_status.get("status") == "interrupted", (
    #     f"Skill executor status should be 'interrupted' after termination, but got {skill_status.get('status')}"
    # )
    # print(
    #     "Skill terminated successfully, current skill status:",
    #     isim.get_skill_executor_status().get("status"),
    # )

    # isim.shutdown()

    # # Step environment test
    # for i in range(5):
    #     action = Action(data=np.random.rand(), metadata={})
    #     step_result = isim.step_env(action)
    #     assert step_result is not None, (
    #         "Failed to step Isaac Simulator environment"
    #     )
    #     obs, reward, terminated, truncated, info = step_result
    #     print(
    #         f"Step {i + 1}: Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}"
    #     )
