# isaac_process.py

# Imports that are safe before AppLauncher:
import torch  # Generally safe, but ensure it doesn't conflict with Isaac's PyTorch if versions differ wildly
import numpy as np
import yaml
import multiprocessing.connection  # For type hinting 'conn' if needed, but not strictly necessary for this file
from isaaclab.app import AppLauncher  # This is the one crucial safe import
import enum


# dynamic_set_attr is a generic utility, safe to define globally
# if it doesn't import any isaac specific modules itself.
def dynamic_set_attr(obj: object, kwargs: dict, path: list):
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
                    if type(current_val) in [int, float, bool, str] and type(
                        current_val
                    ) != type(v):
                        try:
                            v = type(current_val)(v)
                        except ValueError:
                            print(
                                f"Warning: Could not convert value for {'.'.join(path + [k])} from {type(v)} to {type(current_val)}. Using original value."
                            )
                    # print(f"Set {'.'.join(path + [k])} from {getattr(obj, k)} to {v}") # Optional: reduce verbosity
                    setattr(obj, k, v)
                except Exception as e:
                    print(f"Error setting attribute {'.'.join(path + [k])}: {e}")
        else:
            print(f"Warning: Attribute {k} not found in {'.'.join(path)}")


class SimulationMode(enum.Enum):
    MANUAL_STEP = (
        1  # Physics steps only on remote command (paused otherwise, UI responsive)
    )
    AUTO_STEP = 2  # Physics steps continuously (like current, but more explicit)


def isaac_simulation_entry(conn: multiprocessing.connection.Connection, cli_args):
    print("[Isaac Process] Launching Isaac Sim...")
    app_launcher = AppLauncher(cli_args)
    simulation_app = app_launcher.app

    import torch  # Now safe to import torch
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.envs.direct_rl_env import DirectRLEnv
    from isaaclab.sim.simulation_context import SimulationContext

    print("[Isaac Process] Initializing environment...")
    env_cfg = parse_env_cfg(
        cli_args.task,
        device=cli_args.device,
        num_envs=cli_args.num_envs,
        use_fabric=not cli_args.disable_fabric,
    )
    if cli_args.env_config_file:
        print(
            f"[Isaac Process] Loading custom env config from: {cli_args.env_config_file}"
        )
        with open(cli_args.env_config_file, "r") as f:
            env_new_cfg = yaml.safe_load(f)
            dynamic_set_attr(env_cfg, env_new_cfg, path=["env_cfg"])

    env = gym.make(cli_args.task, cfg=env_cfg)
    _env: DirectRLEnv = env.unwrapped  # type: ignore
    _sim: SimulationContext = _env.sim
    print(
        f"[Isaac Process] Environment '{cli_args.task}' created. Device: {_env.device}"
    )

    # Initialize simulation mode (default to MANUAL_STEP or from cli_args if you add it)
    # Convert cli_args.mode to the Enum
    try:
        current_mode = SimulationMode(cli_args.mode)
    except ValueError:
        print(
            f"[Isaac Process] Invalid mode {cli_args.mode} from CLI. Defaulting to MANUAL_STEP."
        )
        current_mode = SimulationMode.MANUAL_STEP

    print(f"[Isaac Process] Initial mode set to: {current_mode.name}")

    # Apply initial mode settings
    if current_mode == SimulationMode.MANUAL_STEP:
        if _sim.is_playing():
            _sim.pause()
        _sim.set_render_mode(
            SimulationContext.RenderMode.FULL_RENDERING
        )  # For debugging
    elif current_mode == SimulationMode.AUTO_STEP:
        if not _sim.is_playing():
            _sim.play()
        _sim.set_render_mode(
            SimulationContext.RenderMode.FULL_RENDERING
        )  # For observation

    last_action_torch_dict = None  # For AUTO_STEP if no new action is received

    try:
        while simulation_app.is_running():
            command_received = False
            # Poll for a short duration to keep the loop responsive
            # In MANUAL_STEP, this loop will mostly be rendering a paused sim
            # In AUTO_STEP, this loop will be stepping the sim
            if conn.poll(0.0001):  # Very short poll
                try:
                    command = conn.recv()
                    command_received = True
                except (EOFError, BrokenPipeError):
                    print("[Isaac Process] Connection closed by parent. Exiting.")
                    break

            if command_received:
                cmd_type = command.get("type")
                print(
                    f"[Isaac Process] Received command: {cmd_type} in mode {current_mode.name}"
                )

                if cmd_type == "reset":
                    # For reset, ensure sim is playing if it was paused (manual mode)
                    was_paused_for_command = False
                    if (
                        current_mode == SimulationMode.MANUAL_STEP
                        and not _sim.is_playing()
                    ):
                        _sim.play()
                        was_paused_for_command = True

                    obs, info = (
                        env.reset()
                    )  # env.reset() internally handles sim.reset()
                    # which should bring sim to a playable state
                    obs_policy = (
                        obs["policy"]
                        if isinstance(obs, dict) and "policy" in obs
                        else obs
                    )
                    conn.send({"obs": obs_policy.cpu().numpy(), "info": info})
                    print("[Isaac Process] Reset complete, result sent.")
                    last_action_torch_dict = None  # Reset last action

                    if was_paused_for_command or (
                        current_mode == SimulationMode.MANUAL_STEP and _sim.is_playing()
                    ):
                        _sim.pause()  # Re-pause if in manual mode

                elif cmd_type == "step":
                    action_numpy_dict = command["action"]
                    action_torch_dict = {
                        k: torch.tensor(v, device=_env.device, dtype=torch.float32)
                        for k, v in action_numpy_dict.items()
                    }
                    last_action_torch_dict = action_torch_dict  # Store for AUTO_STEP

                    was_paused_for_command = False
                    if (
                        current_mode == SimulationMode.MANUAL_STEP
                        and not _sim.is_playing()
                    ):
                        _sim.play()  # Ensure sim is playing for the step
                        was_paused_for_command = True

                    obs, reward, done, trunc, info = env.step(action_torch_dict)
                    obs_policy = (
                        obs["policy"]
                        if isinstance(obs, dict) and "policy" in obs
                        else obs
                    )
                    conn.send(
                        {
                            "obs": obs_policy.cpu().numpy(),
                            "reward": reward.cpu().numpy(),
                            "done": done.cpu().numpy(),
                            "trunc": trunc.cpu().numpy(),
                            "info": info,
                        }
                    )
                    print("[Isaac Process] Step complete, result sent.")

                    if was_paused_for_command or (
                        current_mode == SimulationMode.MANUAL_STEP and _sim.is_playing()
                    ):
                        _sim.pause()  # Re-pause if in manual mode

                elif cmd_type == "set_mode":
                    new_mode_val = command.get("mode")
                    try:
                        new_mode_enum = SimulationMode(new_mode_val)
                        if new_mode_enum != current_mode:
                            old_mode_name = current_mode.name
                            current_mode = new_mode_enum
                            print(
                                f"[Isaac Process] Switching from {old_mode_name} to mode: {current_mode.name}"
                            )

                            if current_mode == SimulationMode.MANUAL_STEP:
                                print(
                                    f"[Isaac Process] Setting MANUAL_STEP. Current play state: {_sim.is_playing()}"
                                )
                                if _sim.is_playing():
                                    print("[Isaac Process] Calling _sim.pause()...")
                                    _sim.pause()
                                    print("[Isaac Process] _sim.pause() completed.")
                                else:
                                    print("[Isaac Process] Sim was already paused.")
                                _sim.set_render_mode(
                                    SimulationContext.RenderMode.FULL_RENDERING
                                )
                                print(
                                    "[Isaac Process] Render mode set for MANUAL_STEP."
                                )

                            elif current_mode == SimulationMode.AUTO_STEP:
                                print(
                                    f"[Isaac Process] Setting AUTO_STEP. Current play state: {_sim.is_playing()}"
                                )
                                if not _sim.is_playing():
                                    print("[Isaac Process] Calling _sim.play()...")
                                    _sim.play()  # This calls app.update() internally once
                                    print("[Isaac Process] _sim.play() completed.")
                                else:
                                    print("[Isaac Process] Sim was already playing.")
                                _sim.set_render_mode(
                                    SimulationContext.RenderMode.FULL_RENDERING
                                )
                                print("[Isaac Process] Render mode set for AUTO_STEP.")

                            print(
                                "[Isaac Process] Sending 'mode_set' response to pipe..."
                            )
                            conn.send(
                                {"status": "mode_set", "new_mode": current_mode.name}
                            )
                            print(
                                f"[Isaac Process] Response sent to pipe for mode {current_mode.name}."
                            )
                        else:
                            print(
                                f"[Isaac Process] Mode already {current_mode.name}. Sending 'mode_already_set' response..."
                            )
                            conn.send(
                                {
                                    "status": "mode_already_set",
                                    "current_mode": current_mode.name,
                                }
                            )
                            print("[Isaac Process] Response sent.")
                    except ValueError:
                        conn.send({"error": f"Invalid mode value: {new_mode_val}"})
                        print(
                            f"[Isaac Process] Invalid mode value received: {new_mode_val}"
                        )

                elif cmd_type == "close":
                    print("[Isaac Process] Received close command.")
                    break
                else:
                    conn.send({"error": "Unknown command"})
            # End of command processing

            # Main loop behavior based on mode
            if current_mode == SimulationMode.MANUAL_STEP:
                if not _sim.is_playing():
                    # In manual mode, if sim is paused, we just render to keep UI alive
                    # _sim.render() will use the current render_mode (FULL_RENDERING for debug)
                    _sim.render()
                else:
                    # This case should ideally not happen if logic is correct,
                    # as steps are command-driven and sim should be paused after.
                    # If it is playing, it means a command just finished, and it will be paused above.
                    # For safety, we can call update, but it might step physics if playSimulations is true.
                    # _sim.render() is safer if we intend no physics step.
                    _sim.render()  # or simulation_app.update() if absolutely necessary.
                    # _sim.render() is preferred to control render mode.

            elif current_mode == SimulationMode.AUTO_STEP:
                if _sim.is_playing():
                    # In AUTO_STEP, we want the simulation to keep running.
                    # If no new command, it steps with internal logic or last action.
                    # For now, env.step() is only called on remote command.
                    # So, we just step the simulation.
                    # If you want to re-apply last_action, you'd do:
                    # if last_action_torch_dict:
                    #     env.step(last_action_torch_dict) # This would send data back via conn
                    # else:
                    #     _sim.step(render=True) # Default step if no action
                    # For simplicity now, just keep sim stepping:
                    _sim.step(render=True)  # render=True to see what's happening
                # uses FULL_RENDERING set for this mode
                else:
                    # Should not be paused in AUTO_STEP unless transitioning or error
                    _sim.play()
                    _sim.step(render=True)

            # A very small sleep to prevent this loop from pegging CPU if conn.poll is too short
            # and no sim.step or sim.render with internal update is called.
            # This is more critical if the sim calls are not blocking enough.
            # time.sleep(0.0001) # e.g. 0.1 ms - use with caution, sim.render/step often suffice

    except Exception as e:
        print(f"[Isaac Process] Error in simulation loop: {e}")
        import traceback

        traceback.print_exc()
        try:
            if conn and not conn.closed:
                conn.send({"error": f"Critical error in Isaac Process: {e}"})
        except Exception as send_e:
            print(f"[Isaac Process] Error sending critical error to parent: {send_e}")
    finally:
        print("[Isaac Process] Cleaning up...")
        if "env" in locals() and env is not None:
            env.close()
            print("[Isaac Process] Environment closed.")
        if "app_launcher" in locals() and app_launcher:
            app_launcher.close()
            print("[Isaac Process] AppLauncher and Simulation app closed.")
        if conn:
            conn.close()
        print("[Isaac Process] Exiting.")
