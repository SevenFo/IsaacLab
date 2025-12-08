#!/usr/bin/env python3
"""
Isaac Lab Server - Pure Environment Server (Shared Memory + Unix Socket).
This script provides ONLY environment operations - no skill management, no robot_brain_system dependencies.
AppLauncher is imported first before any other packages.
"""

import argparse
import functools
import json
import os
import pickle
import socket
import struct
import sys
import time
import traceback
from typing import Any, Dict, Optional
from multiprocessing import shared_memory


def dynamic_set_attr(obj: object, kwargs: dict, path: list = []):
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
                    print(f"Set {'.'.join(path + [k])} from {getattr(obj, k)} to {v}")
                except Exception as e:
                    print(f"Error setting attribute {'.'.join(path + [k])}: {e}")
        else:
            print(f"Warning: Attribute {k} not found in {'.'.join(path)}")


def with_time(func):
    """
    一个用于测量函数执行时间的简单装饰器。
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"⏱️  Function '{func.__name__}' executed in {elapsed:.4f} seconds")
        return result

    return wrapper


# Shared Memory Manager
class SharedMemoryManager:
    """Manager for shared memory buffers used for high-performance data transfer."""

    def __init__(self, name_prefix: str):
        self.name_prefix = name_prefix
        self.shm_obs: shared_memory.SharedMemory | None = None
        self.shm_action: shared_memory.SharedMemory | None = None
        self.shm_metadata: shared_memory.SharedMemory | None = None

    def create_buffers(
        self,
        obs_size: int = 10 * 1024 * 1024,
        action_size: int = 16 * 1024,
        metadata_size: int = 4096,
    ):
        """Create shared memory buffers (server side)."""
        self.shm_obs = shared_memory.SharedMemory(
            name=f"{self.name_prefix}_obs", create=True, size=obs_size
        )
        self.shm_action = shared_memory.SharedMemory(
            name=f"{self.name_prefix}_action", create=True, size=action_size
        )
        self.shm_metadata = shared_memory.SharedMemory(
            name=f"{self.name_prefix}_metadata", create=True, size=metadata_size
        )
        # Initialize metadata
        self._write_metadata({"ready": False, "obs_size": 0, "action_size": 0})

    def cleanup(self):
        """Clean up shared memory buffers."""
        for shm in [self.shm_obs, self.shm_action, self.shm_metadata]:
            if shm:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

    def _write_metadata(self, metadata: Dict[str, Any]):
        """Write metadata to shared memory."""
        if not self.shm_metadata:
            raise RuntimeError("Metadata shared memory not initialized")
        data = pickle.dumps(metadata)
        size = len(data)
        self.shm_metadata.buf[:4] = struct.pack("!I", size)
        self.shm_metadata.buf[4 : 4 + size] = data

    def write_observation(self, obs_data: Any):
        """Write observation to shared memory."""
        if not self.shm_obs:
            raise RuntimeError("Observation shared memory not initialized")
        data = pickle.dumps(obs_data)
        size = len(data)
        if size > len(self.shm_obs.buf) - 4:
            raise ValueError(
                f"Observation too large: {size} > {len(self.shm_obs.buf) - 4}"
            )
        self.shm_obs.buf[:4] = struct.pack("!I", size)
        self.shm_obs.buf[4 : 4 + size] = data

    def read_action(self) -> Any:
        """Read action from shared memory."""
        if not self.shm_action:
            raise RuntimeError("Action shared memory not initialized")
        size = struct.unpack("!I", bytes(self.shm_action.buf[:4]))[0]
        if size == 0:
            return None
        data = bytes(self.shm_action.buf[4 : 4 + size])
        return pickle.loads(data)


def send_message(sock: socket.socket, message: Dict[str, Any]) -> None:
    """Send a message with length prefix via Unix socket."""
    data = pickle.dumps(message)
    length = struct.pack("!I", len(data))
    sock.sendall(length + data)


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    """Receive a message with length prefix via Unix socket."""
    # Read 4-byte length prefix
    raw_length = b""
    while len(raw_length) < 4:
        chunk = sock.recv(4 - len(raw_length))
        if not chunk:
            raise ConnectionError("Socket connection closed")
        raw_length += chunk

    length = struct.unpack("!I", raw_length)[0]

    # Read the message data
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(length - len(data), 4096))
        if not chunk:
            raise ConnectionError("Socket connection closed")
        data += chunk

    return pickle.loads(data)


def main():
    parser = argparse.ArgumentParser(description="Isaac Lab Pure Environment Server")
    parser.add_argument("--socket", type=str, required=True, help="Unix socket path")
    parser.add_argument(
        "--config", type=str, required=True, help="JSON config file path"
    )
    parser.add_argument(
        "--shm-name", type=str, required=True, help="Shared memory name prefix"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        sim_config = json.load(f)

    print("[IsaacLabServer] Starting PURE environment server")
    print(f"[IsaacLabServer] Unix socket: {args.socket}")
    print(f"[IsaacLabServer] Shared memory prefix: {args.shm_name}")
    print(f"[IsaacLabServer] Config: {sim_config}")

    # === CRITICAL: Import AppLauncher FIRST ===
    try:
        from omegaconf import OmegaConf
        from isaaclab.app import AppLauncher
    except ImportError as e:
        print(f"[IsaacLabServer] FATAL: Failed to import isaaclab.app: {e}")
        sys.exit(1)

    # Convert config and create AppLauncher
    app_launcher_params = OmegaConf.create(sim_config)
    app_launcher_params = OmegaConf.to_container(app_launcher_params, resolve=True)
    skill_cfgs = app_launcher_params.pop("skills", {})
    app_launcher = AppLauncher(app_launcher_params)
    simulation_app = app_launcher.app

    print("[IsaacLabServer] AppLauncher initialized successfully")

    # Now import everything else
    import torch
    import numpy as np
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.assets.articulation import Articulation

    # Import known termination helpers used by skills (extend as needed)
    try:
        from isaaclab_tasks.manager_based.manipulation.move_rot_0923.mdp import (
            lift_eef,
            leave_spanner,
            box_open,
        )
    except Exception:
        # If these are not available in this install, we'll warn and continue
        lift_eef = leave_spanner = box_open = None
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
    from argparse import Namespace
    import gymnasium as gym
    import yaml

    # Initialize shared memory
    shm_manager = SharedMemoryManager(args.shm_name)
    # Increase action buffer to accommodate pickled Action objects that may
    # include small metadata. Default is set to 16KB to avoid occasional
    # "Action too large" errors when serialized actions exceed 1KB.
    shm_manager.create_buffers(action_size=16 * 1024)
    print("[IsaacLabServer] Shared memory buffers created")

    # Create Unix socket server
    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server_sock.bind(args.socket)
    server_sock.listen(1)

    print(f"[IsaacLabServer] Listening on Unix socket: {args.socket}")

    client_sock, _ = server_sock.accept()
    print("[IsaacLabServer] Client connected")

    env = None

    try:
        # Initialize environment
        cli_args = Namespace(**app_launcher_params)
        print("[IsaacLabServer] Initializing environment...")

        env_cfg = parse_env_cfg(
            cli_args.task,
            device=cli_args.device,
            num_envs=cli_args.num_envs,
            use_fabric=not cli_args.disable_fabric,
        )

        # Remove termination conditions (client will handle)
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
        if hasattr(env_cfg, "terminations") and hasattr(
            env_cfg.terminations, "time_out"
        ):
            env_cfg.terminations.time_out = None

        # Load custom config if provided
        if hasattr(cli_args, "env_config_file") and cli_args.env_config_file:
            with open(cli_args.env_config_file, "r") as f:
                env_new_cfg = yaml.safe_load(f)
                dynamic_set_attr(env_cfg, env_new_cfg)
        print(f"[IsaacLabServer] Environment config: {env_cfg}")
        env = gym.make(cli_args.task, cfg=env_cfg)
        _env: DirectRLEnv | ManagerBasedRLEnv = env.unwrapped
        _sim = _env.sim
        _sim.set_camera_view((-2.08, -1.12, 3.95), (0.6, -2.0, 2.818))

        obs, info = env.reset()

        # Warm-up
        print("[IsaacLabServer] Starting environment warm-up...")
        if hasattr(_env, "action_manager") and hasattr(
            _env.action_manager, "total_action_dim"
        ):
            action_dim = _env.action_manager.total_action_dim
            zero_action = torch.zeros(
                (_env.num_envs, action_dim), device=_env.device, dtype=torch.float32
            )
            zero_action[..., -1] = 1  # Gripper open

            warmup_steps = getattr(cli_args, "warmup_steps", 10)
            for i in range(warmup_steps):
                obs, reward, terminated, truncated, info = env.step(zero_action)
                if (i + 1) % 5 == 0:
                    print(f"[IsaacLabServer] Warm-up step {i + 1}/{warmup_steps}")

            print(f"[IsaacLabServer] Warm-up complete after {warmup_steps} steps")

        # Helper function to prepare observation
        def _prepare_observation(obs_dict):
            """Prepare observation for shared memory transfer."""
            obs_dict_out = {}
            if isinstance(obs_dict, dict):
                for obs_key in obs_dict.keys():
                    obs_dict_out[obs_key] = {}
                    if isinstance(obs_dict[obs_key], dict):
                        for key, val in obs_dict[obs_key].items():
                            if isinstance(val, torch.Tensor):
                                obs_dict_out[obs_key][key] = val.detach().cpu()
                            else:
                                obs_dict_out[obs_key][key] = val

                    else:
                        obs_dict_out[obs_key] = obs_dict[obs_key]

            return {
                "data": obs_dict_out,
                "metadata": {},
                "timestamp": time.time(),
            }

        # Prepare initial observation
        current_obs = obs
        latest_obs = _prepare_observation(current_obs)
        shm_manager.write_observation(latest_obs)
        print("[IsaacLabServer] Initial observation written to shared memory")

        # === Build success_item_map from config if provided ===
        success_item_map: Dict[str, Dict[str, Any]] = {}

        # Helper to resolve a named success_item to a TerminationTermCfg
        def _resolve_success_item(name: str) -> Optional[TerminationTermCfg]:
            if not name:
                return None
            # Map common names to known functions
            mapping = {
                "box_open": box_open,
                "leave_spanner": leave_spanner,
                "lift_eef": lift_eef,
            }
            func = mapping.get(name)
            if func is None:
                # Try to import by dotted path if provided
                try:
                    components = name.split(".")
                    mod_name = ".".join(components[:-1])
                    fn_name = components[-1]
                    mod = __import__(mod_name, fromlist=[fn_name])
                    func = getattr(mod, fn_name)
                except Exception:
                    func = None

            if func is None:
                return None

            return TerminationTermCfg(func=func)

        # Populate from skill-specific configs
        for skill_name, cfg in skill_cfgs.items():
            if not isinstance(cfg, dict):
                continue
            si_name = cfg.get("success_item")
            if si_name:
                term = _resolve_success_item(si_name)
                if term:
                    success_item_map[skill_name] = {"success_item": term}
                else:
                    print(
                        f"[IsaacLabServer] Warning: unable to resolve success_item '{si_name}' for skill '{skill_name}'"
                    )

        if success_item_map:
            print(
                f"[IsaacLabServer] Registered success_item_map for skills: {list(success_item_map.keys())}"
            )

        # Helper function to extract space info (handles Dict spaces)
        def _extract_space_info(space):
            """Extract space information, handling both Box and Dict spaces."""
            from gymnasium import spaces

            if isinstance(space, spaces.Dict):
                # Nested dictionary of spaces
                space_info = {"type": "Dict", "spaces": {}}
                for key, subspace in space.spaces.items():
                    space_info["spaces"][key] = _extract_space_info(subspace)
                return space_info
            elif isinstance(space, spaces.Box):
                # Box space
                return {
                    "type": "Box",
                    "shape": list(space.shape),
                    "dtype": str(space.dtype),
                    "low": float(space.low.flat[0]) if space.low.size > 0 else None,
                    "high": float(space.high.flat[0]) if space.high.size > 0 else None,
                }
            else:
                # Other space types
                return {
                    "type": type(space).__name__,
                    "shape": list(space.shape) if hasattr(space, "shape") else None,
                    "dtype": str(space.dtype) if hasattr(space, "dtype") else None,
                }

        # Send ready signal
        action_space_info = _extract_space_info(env.action_space)
        obs_space_info = _extract_space_info(env.observation_space)

        send_message(
            client_sock,
            {
                "status": "ready",
                "action_space": action_space_info,
                "observation_space": obs_space_info,
            },
        )
        print(
            "[IsaacLabServer] Sent 'ready' to client - entering pure environment server mode"
        )
        latest_action = zero_action
        # Main loop - Pure environment server
        active = True
        from collections import deque

        loop_dua = deque(maxlen=30)
        while active:
            try:
                # Check for commands
                time_start = time.time()
                client_sock.settimeout(0.01)
                try:
                    command_data = receive_message(client_sock)
                    cmd = command_data.get("command")

                    if cmd == "shutdown":
                        print("[IsaacLabServer] Received shutdown command")
                        active = False
                        send_message(client_sock, {"status": "shutdown_ack"})
                        break

                    elif cmd == "get_observation":
                        # Return current observation
                        obs_payload = _prepare_observation(current_obs)
                        shm_manager.write_observation(obs_payload)
                        send_message(client_sock, {"success": True})

                    elif cmd == "get_current_observation":
                        # Same as get_observation
                        obs_payload = _prepare_observation(current_obs)
                        shm_manager.write_observation(obs_payload)
                        send_message(client_sock, {"success": True})

                    elif cmd == "step_sim":
                        return_obs = command_data.get("return_obs", False)
                        # use zero action to step sim
                        _t = time.time()
                        zero_action = torch.zeros_like(
                            latest_action, device=_env.device
                        )
                        zero_action[:, -1] = latest_action[:, -1]  # Keep gripper state
                        # obs, _, _, _, _ = _env.step(zero_action)
                        _env.sim.render(mode=_env.sim.RenderMode.FULL_RENDERING)
                        if return_obs:
                            _env.scene.update(dt=_env.sim.get_physics_dt())
                            current_obs = _env.observation_manager.compute()
                            _env.obs_buf = current_obs
                        _t_ = time.time()
                        # print(f"[IsaacLabServer-step_sim] step_take: {_t_ - _t:.4f}s")
                        # Write new observation to shared memory
                        if return_obs:
                            obs_payload = _prepare_observation(_env.obs_buf)
                            shm_manager.write_observation(obs_payload)
                        _t_t = time.time()
                        # print(
                        #     f"[IsaacLabServer-step_sim] write_obs_take: {_t_t - _t_:.4f}s"
                        # )
                        send_message(client_sock, {"success": True})

                    elif cmd == "step_env":
                        # Read action, execute step
                        _t = time.time()
                        action_dict = shm_manager.read_action()
                        if action_dict:
                            action_data_np = np.array(action_dict["data"])
                            action_tensor = torch.from_numpy(action_data_np).to(
                                env.device
                            )

                            # Ensure correct dimensions
                            if len(action_tensor.shape) == 1:
                                action_tensor = action_tensor.unsqueeze(0)

                            current_obs, reward, terminated, truncated, info = env.step(
                                action_tensor
                            )
                            latest_action = action_tensor
                            time_after_step = time.time()
                            # print(
                            #     f"[IsaacLabServer-step_env] step_take: {time_after_step - _t:.4f}s"
                            # )
                            # Evaluate success criterion if skill_name present
                            skill_success_flag = False
                            try:
                                metadata = action_dict.get("metadata", {}) or {}
                                skill_name = metadata.get("skill_name")
                                if skill_name and skill_name in success_item_map:
                                    term_cfg = success_item_map[skill_name][
                                        "success_item"
                                    ]
                                    params = getattr(term_cfg, "params", {}) or {}
                                    try:
                                        skill_success_flag = term_cfg.func(
                                            env, **params
                                        )[0].item()
                                        # Result may be tensor/array; try to interpret truth
                                        # print(
                                        #     f"[IsaacLabServer] Success item result for skill {skill_name}: {skill_success_flag}"
                                        # )
                                    except Exception as e:
                                        print(
                                            f"[IsaacLabServer] Error evaluating success_item for skill {skill_name}: {e}"
                                        )
                                else:
                                    print(
                                        f"[IsaacLabServer] No success_item configured for skill '{skill_name}' in {success_item_map.keys()}"
                                    )
                            except Exception:
                                skill_success_flag = False
                            t_after_success = time.time()
                            # print(
                            #     f"[IsaacLabServer-step_env] success_item_take: {t_after_success - time_after_step:.4f}s"
                            # )
                            # Write new observation to shared memory
                            obs_payload = _prepare_observation(current_obs)
                            shm_manager.write_observation(obs_payload)
                            time_after_write = time.time()
                            # print(
                            #     f"[IsaacLabServer-step_env] write_obs_take: {time_after_write - t_after_success:.4f}s"
                            # )
                            send_message(
                                client_sock,
                                {
                                    "success": True,
                                    "reward": float(reward.item())
                                    if hasattr(reward, "item")
                                    else float(reward),
                                    "terminated": bool(terminated.item())
                                    if hasattr(terminated, "item")
                                    else bool(terminated),
                                    "truncated": bool(truncated.item())
                                    if hasattr(truncated, "item")
                                    else bool(truncated),
                                    "info": {"skill_success": skill_success_flag},
                                },
                            )
                        else:
                            send_message(
                                client_sock,
                                {"success": False, "error": "No action data"},
                            )

                    elif cmd == "reset_env":
                        # Reset environment
                        current_obs, info = env.reset()
                        obs_payload = _prepare_observation(current_obs)
                        shm_manager.write_observation(obs_payload)
                        send_message(client_sock, {"success": True})

                    elif cmd == "get_scene_info":
                        # Provide scene state query
                        scene_info = {}
                        if hasattr(_env, "scene"):
                            for key in _env.scene.keys():
                                try:
                                    obj = _env.scene[key]
                                    if hasattr(obj, "data"):
                                        obj_info = {}
                                        if hasattr(obj.data, "root_pos_w"):
                                            obj_info["position"] = (
                                                obj.data.root_pos_w.cpu()
                                                .numpy()
                                                .tolist()
                                            )
                                        if hasattr(obj.data, "root_quat_w"):
                                            obj_info["orientation"] = (
                                                obj.data.root_quat_w.cpu()
                                                .numpy()
                                                .tolist()
                                            )
                                        if hasattr(obj.data, "joint_pos"):
                                            obj_info["joint_positions"] = (
                                                obj.data.joint_pos.cpu()
                                                .numpy()
                                                .tolist()
                                            )
                                        scene_info[key] = obj_info
                                except Exception as e:
                                    print(
                                        f"[IsaacLabServer] Error extracting info for {key}: {e}"
                                    )

                        send_message(
                            client_sock, {"success": True, "scene_info": scene_info}
                        )

                    elif cmd == "get_scene_state":
                        obj_name = command_data.get("target_name")
                        env_id = command_data.get("env_id", 0)
                        state_names = command_data.get("state_names", [])
                        try:
                            if obj_name in _env.scene.keys():
                                obj = _env.scene[obj_name]
                                state_data = {}
                                failed_states = []
                                for state_name in state_names:
                                    if hasattr(obj.data, state_name):
                                        tensor = getattr(obj.data, state_name)
                                        state_data[state_name] = (
                                            tensor[env_id : env_id + 1]
                                            .cpu()
                                            .numpy()
                                            .tolist()
                                        )
                                    else:
                                        failed_states.append(state_name)
                                send_message(
                                    client_sock,
                                    {
                                        "success": True,
                                        "state_data": state_data,
                                        "failed_states": failed_states,
                                    },
                                )
                            else:
                                send_message(
                                    client_sock,
                                    {
                                        "success": False,
                                        "error": f"Object '{obj_name}' not found",
                                    },
                                )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "get_robot_state":
                        # Get robot state
                        robot_state = {}
                        if hasattr(_env, "scene") and "robot" in _env.scene.keys():
                            robot = _env.scene["robot"]
                            if hasattr(robot.data, "joint_pos"):
                                robot_state["joint_positions"] = (
                                    robot.data.joint_pos.cpu().numpy().tolist()
                                )
                            if hasattr(robot.data, "joint_vel"):
                                robot_state["joint_velocities"] = (
                                    robot.data.joint_vel.cpu().numpy().tolist()
                                )
                            if hasattr(robot.data, "root_pos_w"):
                                robot_state["end_effector_pos"] = (
                                    robot.data.root_pos_w.cpu().numpy().tolist()
                                )

                        send_message(
                            client_sock, {"success": True, "robot_state": robot_state}
                        )

                    elif cmd == "get_object_pose":
                        # Get object pose (position + quaternion)
                        obj_name = command_data.get("name")
                        env_id = command_data.get("env_id", 0)
                        try:
                            if obj_name in _env.scene.keys():
                                obj = _env.scene[obj_name]
                                pos = (
                                    obj.data.root_pos_w[env_id : env_id + 1]
                                    .cpu()
                                    .numpy()
                                    .tolist()
                                )
                                quat = (
                                    obj.data.root_quat_w[env_id : env_id + 1]
                                    .cpu()
                                    .numpy()
                                    .tolist()
                                )
                                send_message(
                                    client_sock,
                                    {
                                        "success": True,
                                        "position": pos,
                                        "quaternion": quat,
                                    },
                                )
                            else:
                                send_message(
                                    client_sock,
                                    {
                                        "success": False,
                                        "error": f"Object '{obj_name}' not found",
                                    },
                                )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "get_object_velocity":
                        # Get object velocity (linear + angular)
                        obj_name = command_data.get("name")
                        env_id = command_data.get("env_id", 0)
                        try:
                            if obj_name in _env.scene.keys():
                                obj = _env.scene[obj_name]
                                lin_vel = (
                                    obj.data.root_lin_vel_w[env_id : env_id + 1]
                                    .cpu()
                                    .numpy()
                                    .tolist()
                                )
                                ang_vel = (
                                    obj.data.root_ang_vel_w[env_id : env_id + 1]
                                    .cpu()
                                    .numpy()
                                    .tolist()
                                )
                                send_message(
                                    client_sock,
                                    {
                                        "success": True,
                                        "linear_velocity": lin_vel,
                                        "angular_velocity": ang_vel,
                                    },
                                )
                            else:
                                send_message(
                                    client_sock,
                                    {
                                        "success": False,
                                        "error": f"Object '{obj_name}' not found",
                                    },
                                )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "set_object_pose":
                        # Set object pose
                        obj_name = command_data.get("name")
                        env_id = command_data.get("env_id", 0)
                        position = command_data.get("position")
                        quaternion = command_data.get("quaternion")
                        try:
                            if obj_name in _env.scene.keys():
                                obj = _env.scene[obj_name]
                                pose_tensor = torch.tensor(
                                    [position + quaternion],
                                    device=_env.device,
                                    dtype=torch.float32,
                                )
                                obj.write_root_pose_to_sim(
                                    pose_tensor,
                                    env_ids=torch.tensor([env_id], device=_env.device),
                                )
                                send_message(client_sock, {"success": True})
                            else:
                                send_message(
                                    client_sock,
                                    {
                                        "success": False,
                                        "error": f"Object '{obj_name}' not found",
                                    },
                                )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "set_object_velocity":
                        # Set object velocity
                        obj_name = command_data.get("name")
                        env_id = command_data.get("env_id", 0)
                        linear_velocity = command_data.get("linear_velocity")
                        angular_velocity = command_data.get("angular_velocity")
                        try:
                            if obj_name in _env.scene.keys():
                                obj = _env.scene[obj_name]
                                vel_tensor = torch.tensor(
                                    [linear_velocity + angular_velocity],
                                    device=_env.device,
                                    dtype=torch.float32,
                                )
                                obj.write_root_velocity_to_sim(
                                    vel_tensor,
                                    env_ids=torch.tensor([env_id], device=_env.device),
                                )
                                send_message(client_sock, {"success": True})
                            else:
                                send_message(
                                    client_sock,
                                    {
                                        "success": False,
                                        "error": f"Object '{obj_name}' not found",
                                    },
                                )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "get_object_default_state":
                        # Get object default state
                        obj_name = command_data.get("name")
                        try:
                            if obj_name in _env.scene.keys():
                                obj = _env.scene[obj_name]
                                default_state = (
                                    obj.data.default_root_state.cpu().numpy().tolist()
                                )
                                send_message(
                                    client_sock,
                                    {"success": True, "default_state": default_state},
                                )
                            else:
                                send_message(
                                    client_sock,
                                    {
                                        "success": False,
                                        "error": f"Object '{obj_name}' not found",
                                    },
                                )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "get_robot_joint_defaults":
                        # Get robot default joint positions and velocities
                        try:
                            if "robot" in _env.scene.keys():
                                robot = _env.scene["robot"]
                                default_joint_pos = (
                                    robot.data.default_joint_pos.cpu().numpy().tolist()
                                )
                                default_joint_vel = (
                                    robot.data.default_joint_vel.cpu().numpy().tolist()
                                )
                                send_message(
                                    client_sock,
                                    {
                                        "success": True,
                                        "default_joint_pos": default_joint_pos,
                                        "default_joint_vel": default_joint_vel,
                                    },
                                )
                            else:
                                send_message(
                                    client_sock,
                                    {
                                        "success": False,
                                        "error": "Robot not found in scene",
                                    },
                                )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "set_env_decimation":
                        # Set environment simulation decimation
                        decimation = command_data.get("decimation")
                        try:
                            _env.cfg.decimation = decimation
                            send_message(client_sock, {"success": True})
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "set_robot_joint_targets":
                        # Set robot joint position and velocity targets
                        pos_target = command_data.get("pos_target")
                        vel_target = command_data.get("vel_target")
                        robot_name = command_data.get("robot_name")
                        try:
                            if robot_name in _env.scene.keys() and isinstance(
                                _env.scene[robot_name], Articulation
                            ):
                                robot = _env.scene[robot_name]
                                if pos_target is not None:
                                    pos_tensor = torch.tensor(
                                        pos_target,
                                        device=_env.device,
                                        dtype=torch.float32,
                                    )
                                    robot.set_joint_position_target(pos_tensor)
                                if vel_target is not None:
                                    vel_tensor = torch.tensor(
                                        vel_target,
                                        device=_env.device,
                                        dtype=torch.float32,
                                    )
                                    robot.set_joint_velocity_target(vel_tensor)
                                send_message(client_sock, {"success": True})
                            else:
                                send_message(
                                    client_sock,
                                    {
                                        "success": False,
                                        "error": f"Robot '{robot_name}' not found in scene",
                                    },
                                )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "write_robot_joint_state":
                        # Write robot joint state directly to simulation
                        joint_pos = command_data.get("joint_pos")
                        joint_vel = command_data.get("joint_vel")
                        robot_name = command_data.get("robot_name")
                        try:
                            if robot_name in _env.scene.keys() and isinstance(
                                _env.scene[robot_name], Articulation
                            ):
                                robot = _env.scene[robot_name]
                                pos_tensor = torch.tensor(
                                    joint_pos, device=_env.device, dtype=torch.float32
                                )
                                vel_tensor = torch.tensor(
                                    joint_vel, device=_env.device, dtype=torch.float32
                                )
                                robot.write_joint_state_to_sim(pos_tensor, vel_tensor)
                                send_message(client_sock, {"success": True})
                            else:
                                send_message(
                                    client_sock,
                                    {
                                        "success": False,
                                        "error": f"Robot '{robot_name}' not found in scene",
                                    },
                                )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "get_action_dim":
                        # Get action dimension
                        try:
                            action_dim = _env.action_manager.total_action_dim
                            send_message(
                                client_sock, {"success": True, "action_dim": action_dim}
                            )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "get_env_origins":
                        # Get environment origins
                        try:
                            env_origins = _env.scene.env_origins.cpu().numpy().tolist()
                            send_message(
                                client_sock,
                                {"success": True, "env_origins": env_origins},
                            )
                        except Exception as e:
                            send_message(
                                client_sock, {"success": False, "error": str(e)}
                            )

                    elif cmd == "custom_object_command":
                        name = command_data.get("name")
                        func = command_data.get("func")
                        params = command_data.get("params", {})
                        args = params.pop("args", [])
                        if name not in _env.scene.keys():
                            send_message(
                                client_sock,
                                {
                                    "success": False,
                                    "error": f"Object '{name}' not found",
                                },
                            )
                            continue
                        obj = _env.scene[name]
                        if not hasattr(obj, func):
                            send_message(
                                client_sock,
                                {
                                    "success": False,
                                    "error": f"Object '{name}' has no function '{func}'",
                                },
                            )
                            continue
                        method = getattr(obj, func)
                        try:
                            result = method(*args, **params)
                            # Convert result to serializable form
                            if isinstance(result, torch.Tensor):
                                result = result.cpu().numpy().tolist()
                            send_message(
                                client_sock,
                                {"success": True, "result": result},
                            )
                        except Exception as e:
                            send_message(
                                client_sock,
                                {
                                    "success": False,
                                    "error": f"Error executing '{func}' on '{name}': {e}",
                                },
                            )
                            continue

                    elif cmd == "get_object_attribute":
                        name = command_data.get("name")
                        attribute = command_data.get("attribute")
                        if name not in _env.scene.keys():
                            send_message(
                                client_sock,
                                {
                                    "success": False,
                                    "error": f"Object '{name}' not found",
                                },
                            )
                            continue
                        obj = _env.scene[name]
                        if not hasattr(obj, attribute):
                            send_message(
                                client_sock,
                                {
                                    "success": False,
                                    "error": f"Object '{name}' has no attribute '{attribute}'",
                                },
                            )
                            continue
                        attr_value = getattr(obj, attribute)
                        # Convert to serializable form if needed
                        if isinstance(attr_value, torch.Tensor):
                            attr_value = attr_value.cpu().numpy().tolist()
                        send_message(
                            client_sock,
                            {"success": True, "attribute_value": attr_value},
                        )
                    else:
                        send_message(
                            client_sock,
                            {"error": f"Unknown command: {cmd}", "success": False},
                        )

                except socket.timeout:
                    # No command, continue
                    pass
                time_end = time.time()
                loop_dua.append(time_end - time_start)
                if len(loop_dua) == loop_dua.maxlen:
                    avg_loop_time = sum(loop_dua) / len(loop_dua)
                    # print(
                    #     f"[IsaacLabServer] Average loop time over last {loop_dua.maxlen} iterations: {avg_loop_time * 1000:.2f} ms, hz: {1.0 / avg_loop_time:.2f}"
                    # )
                    loop_dua.clear()
            except (ConnectionError, BrokenPipeError, EOFError) as e:
                print(f"[IsaacLabServer] Connection error: {e}")
                active = False
            except Exception as e:
                print(f"[IsaacLabServer] Error in main loop: {e}")
                traceback.print_exc()
                try:
                    send_message(client_sock, {"error": str(e), "success": False})
                except Exception:
                    pass
                time.sleep(0.1)

    except Exception as e:
        print(f"[IsaacLabServer] Fatal error: {e}")
        traceback.print_exc()
        try:
            send_message(client_sock, {"status": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        print("[IsaacLabServer] Cleaning up...")

        # Clean up shared memory
        shm_manager.cleanup()
        print("[IsaacLabServer] Shared memory cleaned up")

        if env is not None:
            try:
                env.close()
                print("[IsaacLabServer] Environment closed")
            except Exception as e:
                print(f"[IsaacLabServer] Error closing environment: {e}")

        if simulation_app.is_running():
            try:
                simulation_app.close()
                print("[IsaacLabServer] Isaac App closed")
            except Exception as e:
                print(f"[IsaacLabServer] Error closing Isaac App: {e}")

        client_sock.close()
        server_sock.close()

        # Remove socket file
        if os.path.exists(args.socket):
            os.unlink(args.socket)

        print("[IsaacLabServer] Server terminated")


if __name__ == "__main__":
    main()
