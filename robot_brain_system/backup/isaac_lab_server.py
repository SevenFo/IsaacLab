#!/usr/bin/env python3
"""
Isaac Lab Server - Standalone process for Isaac Sim/Lab.
This script ensures AppLauncher is imported first, before any other packages.
It runs as a completely independent process and communicates via TCP socket.
"""

import sys
import argparse
import json
import socket
import struct
import traceback
import pickle
from typing import Any, Dict


def send_message(sock: socket.socket, message: Dict[str, Any]) -> None:
    """Send a message with length prefix to avoid TCP packet fragmentation."""
    data = pickle.dumps(message)
    length = struct.pack("!I", len(data))
    sock.sendall(length + data)


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    """Receive a message with length prefix."""
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
    parser = argparse.ArgumentParser(description="Isaac Lab Server")
    parser.add_argument("--port", type=int, required=True, help="TCP port to listen on")
    parser.add_argument(
        "--config", type=str, required=True, help="JSON config file path"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        sim_config = json.load(f)

    print(f"[IsaacLabServer] Starting server on port {args.port}")
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
    import torch
    import queue
    import time
    import math

    from robot_brain_system.core.skill_manager import (
        SkillExecutor,
        get_skill_registry,
    )
    from robot_brain_system.utils import dynamic_set_attr
    import robot_brain_system.skills  # noqa: F401
    from robot_brain_system.skills.alice_control_skills import AliceControl

    # Create TCP server socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", args.port))
    server_sock.listen(1)

    print(f"[IsaacLabServer] Listening on 127.0.0.1:{args.port}")

    client_sock, client_addr = server_sock.accept()
    print(f"[IsaacLabServer] Client connected from {client_addr}")

    env = None
    skill_executor = None

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

        # Warm-up
        print("[IsaacLabServer] Starting environment warm-up...")
        if hasattr(_env, "action_manager") and hasattr(
            _env.action_manager, "total_action_dim"
        ):
            action_dim = _env.action_manager.total_action_dim
            zero_action = torch.zeros(
                (_env.num_envs, action_dim),
                device=_env.device,
                dtype=torch.float32,
            )
            zero_action[..., -1] = 1  # Gripper open

            disable_alice = True
            if "alice" in _env.scene.keys():
                disable_alice = False
                init_alice_joint_position_target = torch.zeros_like(
                    _env.scene["alice"].data.joint_pos_target
                )
                init_alice_joint_position_target[:, :9] = torch.tensor(
                    [
                        0.0,
                        math.radians(66.7),
                        math.radians(50.7),
                        0.0,
                        math.radians(25.9),
                        math.radians(-23.2),
                        math.radians(-141.8),
                        math.radians(-11.0),
                        math.radians(-41.7),
                    ],
                    device=_env.device,
                )
                _env.scene["alice"].set_joint_position_target(
                    init_alice_joint_position_target
                )
                _env.scene["alice"].write_joint_state_to_sim(
                    init_alice_joint_position_target,
                    torch.zeros_like(
                        init_alice_joint_position_target, device=_env.device
                    ),
                )
                marker_cfg = POSITION_GOAL_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/MoveTargetVisualizer"
                _env.move_target_visualizer = VisualizationMarkers(marker_cfg)
                # Alice control can be initialized here if needed
                _ = AliceControl()  # noqa: F841

            for i in range(cli_args.warmup_steps):
                obs, reward, terminated, truncated, info = env.step(zero_action)
                print(f"[IsaacLabServer] Warm-up step {i + 1}/{cli_args.warmup_steps}")

            print(
                f"[IsaacLabServer] Warm-up complete after {cli_args.warmup_steps} steps"
            )

        if not disable_alice:
            alice_articulator = _env.scene["alice"]
            if alice_articulator is not None:
                print(
                    f"[IsaacLabServer] Alice articulator joint_names: {alice_articulator.joint_names}"
                )

        obs_payload = {
            "data": obs,
            "metadata": "None",
            "timestamp": time.time(),
        }
        obs_queue.put(obs_payload)
        latest_obs_payload = obs_payload

        # Initialize skill system
        skill_registry = get_skill_registry()
        print(f"[IsaacLabServer] Found {len(skill_registry.list_skills())} skills")

        skill_executor = SkillExecutor(skill_registry, env=env)
        print("[IsaacLabServer] SkillExecutor initialized")

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
        print("[IsaacLabServer] Sent 'ready' to client")

        # Main loop
        obs_dict = obs

        def _get_observation():
            if isinstance(_env, ManagerBasedRLEnv):
                return _env.observation_manager.compute()
            elif isinstance(_env, DirectRLEnv):
                return _env._get_observations()
            else:
                raise TypeError(f"Unsupported unwrapped environment type: {type(_env)}")

        success_item_map = {
            "open_box": {"success_item": TerminationTermCfg(func=box_open)},
            "grasp_spanner": {"success_item": TerminationTermCfg(func=leave_spanner)},
            "move_box_to_suitable_position": {
                "success_item": TerminationTermCfg(func=lift_eef)
            },
        }
        skill_executor.skill_success_item_map = success_item_map

        active = True
        while active:
            try:
                # Check for commands (non-blocking with timeout)
                client_sock.settimeout(0.001)
                try:
                    command_data = receive_message(client_sock)
                    cmd = command_data.get("command")

                    if cmd == "shutdown":
                        print("[IsaacLabServer] Received shutdown command")
                        active = False
                        send_message(client_sock, {"status": "shutdown_ack"})
                        break

                    elif cmd == "get_skill_registry":
                        send_message(
                            client_sock,
                            {"success": True, "skill_registry": skill_registry},
                        )

                    elif cmd == "start_skill_non_blocking":
                        skill_name = command_data["skill_name"]
                        params = command_data["parameters"]
                        print(
                            f"[IsaacLabServer] Starting skill (non-blocking): {skill_name}"
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
                        send_message(
                            client_sock,
                            {
                                "success": success,
                                "status": skill_executor.status.value,
                            },
                        )

                    elif cmd == "get_skill_executor_status":
                        send_message(client_sock, skill_executor.get_status_info())

                    elif cmd == "cleanup_skill":
                        skill_executor.cleanup_skill()
                        send_message(client_sock, {"success": True})

                    elif cmd == "terminate_current_skill":
                        skill_status = command_data["skill_status"]
                        status_info = command_data.get("status_info", "")
                        success = skill_executor.terminate_current_skill(
                            skill_status, status_info=status_info
                        )
                        send_message(client_sock, {"success": success})

                    elif cmd == "change_current_skill_status":
                        skill_status = command_data["status"]
                        success = skill_executor.change_current_skill_status(
                            skill_status=skill_status
                        )
                        send_message(client_sock, {"success": success})

                    elif cmd == "change_current_skill_status_force":
                        skill_status = command_data["status"]
                        success = skill_executor._change_current_skill_status_force(
                            skill_status=skill_status
                        )
                        send_message(client_sock, {"success": success})

                    elif cmd == "get_observation":
                        obss = []
                        try:
                            while True:
                                obss.append(obs_queue.get_nowait())
                                obs_queue.task_done()
                        except queue.Empty:
                            pass
                        send_message(
                            client_sock, {"success": True, "observation_data": obss}
                        )

                    elif cmd == "get_current_observation":
                        send_message(
                            client_sock,
                            {
                                "success": True,
                                "observation_data": latest_obs_payload,
                                "info": info,
                            },
                        )

                    elif cmd == "reset_env":
                        delattr(env.unwrapped, "switch_heavybox")
                        obs_dict, info = env.reset()
                        send_message(client_sock, {"success": True})

                    else:
                        send_message(
                            client_sock,
                            {
                                "error": f"Unknown command: {cmd}",
                                "success": False,
                            },
                        )

                except socket.timeout:
                    pass  # No message received, continue

                # Step skill if running
                if skill_executor.is_running():
                    skill_exec_result = skill_executor.step(obs_dict)
                    obs_dict, reward, terminated, truncated, info = skill_exec_result

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
                        latest_obs_payload = obs_payload
                    else:
                        obs_dict = latest_obs_payload["data"]
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

                if not skill_executor.is_running():
                    time.sleep(0.001)

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
        print("[IsaacLabServer] Server terminated")


if __name__ == "__main__":
    main()
