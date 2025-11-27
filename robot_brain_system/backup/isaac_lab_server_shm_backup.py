#!/usr/bin/env python3
"""
Isaac Lab Server - Pure Environment Server (Shared Memory + Unix Socket).
This script provides ONLY environment operations - no skill management, no robot_brain_system dependencies.
AppLauncher is imported first before any other packages.
"""
import argparse
import json
import os
import pickle
import socket
import struct
import sys
import time
import traceback
from typing import Any, Dict
from multiprocessing import shared_memory


# Shared Memory Manager
class SharedMemoryManager:
    """Manager for shared memory buffers used for high-performance data transfer."""

    def __init__(self, name_prefix: str):
        self.name_prefix = name_prefix
        self.shm_obs: shared_memory.SharedMemory | None = None
        self.shm_action: shared_memory.SharedMemory | None = None
        self.shm_metadata: shared_memory.SharedMemory | None = None

    def create_buffers(
        self, obs_size: int = 10 * 1024 * 1024, action_size: int = 1024, metadata_size: int = 4096
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
            raise ValueError(f"Observation too large: {size} > {len(self.shm_obs.buf) - 4}")
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
    parser.add_argument("--config", type=str, required=True, help="JSON config file path")
    parser.add_argument("--shm-name", type=str, required=True, help="Shared memory name prefix")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        sim_config = json.load(f)

    print(f"[IsaacLabServer] Starting PURE environment server")
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

    app_launcher = AppLauncher(app_launcher_params)
    simulation_app = app_launcher.app

    print("[IsaacLabServer] AppLauncher initialized successfully")

    # Now import everything else
    import torch
    import numpy as np
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
    from argparse import Namespace
    import gymnasium as gym
    import yaml

    # Initialize shared memory
    shm_manager = SharedMemoryManager(args.shm_name)
    shm_manager.create_buffers()
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
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None

        # Load custom config if provided
        if hasattr(cli_args, "env_config_file") and cli_args.env_config_file:
            with open(cli_args.env_config_file, "r") as f:
                env_new_cfg = yaml.safe_load(f)
                for key, value in env_new_cfg.items():
                    if hasattr(env_cfg, key):
                        setattr(env_cfg, key, value)

        env = gym.make(cli_args.task, cfg=env_cfg)
        _env: DirectRLEnv | ManagerBasedRLEnv = env.unwrapped
        _sim = _env.sim
        _sim.set_camera_view((-2.08, -1.12, 3.95), (0.6, -2.0, 2.818))

        obs, info = env.reset()

        # Warm-up
        print("[IsaacLabServer] Starting environment warm-up...")
        if hasattr(_env, "action_manager") and hasattr(_env.action_manager, "total_action_dim"):
            action_dim = _env.action_manager.total_action_dim
            zero_action = torch.zeros((_env.num_envs, action_dim), device=_env.device, dtype=torch.float32)
            zero_action[..., -1] = 1  # Gripper open

            warmup_steps = getattr(cli_args, 'warmup_steps', 10)
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
                                obs_dict_out[obs_key][key] = val.clone()
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

        # Send ready signal
        action_space_info = {
            "shape": list(env.action_space.shape),
            "dtype": str(env.action_space.dtype),
        }
        obs_space_info = {
            "shape": list(env.observation_space.shape),
            "dtype": str(env.observation_space.dtype),
        }

        send_message(
            client_sock,
            {
                "status": "ready",
                "action_space": action_space_info,
                "observation_space": obs_space_info,
            },
        )
        print("[IsaacLabServer] Sent 'ready' to client - entering pure environment server mode")

        # Main loop - Pure environment server
        active = True
        while active:
            try:
                # Check for commands
                client_sock.settimeout(0.1)
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

                    elif cmd == "step_env":
                        # Read action, execute step
                        action_dict = shm_manager.read_action()
                        if action_dict:
                            action_data_np = np.array(action_dict["data"])
                            action_tensor = torch.from_numpy(action_data_np).to(env.device)

                            # Ensure correct dimensions
                            if len(action_tensor.shape) == 1:
                                action_tensor = action_tensor.unsqueeze(0)

                            current_obs, reward, terminated, truncated, info = env.step(action_tensor)

                            # Write new observation to shared memory
                            obs_payload = _prepare_observation(current_obs)
                            shm_manager.write_observation(obs_payload)

                            send_message(
                                client_sock,
                                {
                                    "success": True,
                                    "reward": float(reward.item()) if hasattr(reward, 'item') else float(reward),
                                    "terminated": bool(terminated.item()) if hasattr(terminated, 'item') else bool(terminated),
                                    "truncated": bool(truncated.item()) if hasattr(truncated, 'item') else bool(truncated),
                                    "info": {},  # Simplified to avoid serialization issues
                                },
                            )
                        else:
                            send_message(client_sock, {"success": False, "error": "No action data"})

                    elif cmd == "reset_env":
                        # Reset environment
                        current_obs, info = env.reset()
                        obs_payload = _prepare_observation(current_obs)
                        shm_manager.write_observation(obs_payload)
                        send_message(client_sock, {"success": True})

                    elif cmd == "get_scene_info":
                        # Provide scene state query
                        scene_info = {}
                        if hasattr(_env, 'scene'):
                            for key in _env.scene.keys():
                                try:
                                    obj = _env.scene[key]
                                    if hasattr(obj, 'data'):
                                        obj_info = {}
                                        if hasattr(obj.data, 'root_pos_w'):
                                            obj_info['position'] = obj.data.root_pos_w.cpu().numpy().tolist()
                                        if hasattr(obj.data, 'root_quat_w'):
                                            obj_info['orientation'] = obj.data.root_quat_w.cpu().numpy().tolist()
                                        if hasattr(obj.data, 'joint_pos'):
                                            obj_info['joint_positions'] = obj.data.joint_pos.cpu().numpy().tolist()
                                        scene_info[key] = obj_info
                                except Exception as e:
                                    print(f"[IsaacLabServer] Error extracting info for {key}: {e}")

                        send_message(client_sock, {"success": True, "scene_info": scene_info})

                    elif cmd == "get_robot_state":
                        # Get robot state
                        robot_state = {}
                        if hasattr(_env, 'scene') and 'robot' in _env.scene:
                            robot = _env.scene['robot']
                            if hasattr(robot.data, 'joint_pos'):
                                robot_state['joint_positions'] = robot.data.joint_pos.cpu().numpy().tolist()
                            if hasattr(robot.data, 'joint_vel'):
                                robot_state['joint_velocities'] = robot.data.joint_vel.cpu().numpy().tolist()
                            if hasattr(robot.data, 'root_pos_w'):
                                robot_state['end_effector_pos'] = robot.data.root_pos_w.cpu().numpy().tolist()

                        send_message(client_sock, {"success": True, "robot_state": robot_state})

                    else:
                        send_message(client_sock, {"error": f"Unknown command: {cmd}", "success": False})

                except socket.timeout:
                    # No command, continue
                    pass

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
