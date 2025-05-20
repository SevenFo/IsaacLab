# rpyc_server.py (虽然文件名是 rpyc_server.py，但实际用的是 aiohttp)

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Async HTTP Server for Isaac Sim Environment using aiohttp"
)
parser.add_argument(
    "--mode",
    type=int,
    default=2,
    choices=[1, 2],
    help="Running mode: 1=Auto (step with last action or None), 2=Manual (step on request) (default: Manual)",
)
parser.add_argument(
    "--task", type=str, required=True, help="Name of the task/environment to create"
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--env_config_file",
    type=str,
    default=None,
    help="Env Config YAML file Path, use to update default env config",
)
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="Hostname to bind the server.",
)
parser.add_argument("--port", type=int, default=18861, help="Port to bind the server.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments for the server's env instance.",
)
parser.add_argument(
    "--max_episode_length",  # 这个参数在当前代码中没有被直接使用，但保留了
    type=int,
    default=250,
    help="Max steps per episode on server.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()  # Renamed to args_cli to avoid conflict

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import numpy as np
import gymnasium as gym
import yaml
from functools import partial
from typing import Optional, Any, Dict, Tuple, List  # Added List
from isaaclab_tasks.utils import parse_env_cfg
from aiohttp import web
import torch

# import threading # threading 不再需要，我们将使用omni.kit.async_engine
# from queue import Queue # queue.Queue 不再需要，使用 asyncio.Queue
import json
import time  # time 可能仍用于调试或特定逻辑，但核心循环不依赖它
import pickle

# Import Omniverse specific modules
from omni.kit.app import get_app  # For get_app().next_update_async()


def dynamic_set_attr(obj: object, kwargs: dict, path: List[str]):
    if kwargs is None:
        return
    # ... (dynamic_set_attr 实现保持不变) ...
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
                    print(f"Set {'.'.join(path + [k])} from {getattr(obj, k)} to {v}")
                    setattr(obj, k, v)
                except Exception as e:
                    print(f"Error setting attribute {'.'.join(path + [k])}: {e}")
        else:
            print(f"Warning: Attribute {k} not found in {'.'.join(path)}")


class SimulationManager:
    # ... (SimulationManager 实现保持不变，包括 _init_env 和 simulation_worker) ...
    def __init__(self, sim_args):
        self.sim_args = sim_args
        self.env: Optional[gym.Env] = None
        self.mode = sim_args.mode
        self.requests_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        self.running = True
        self.current_action: Optional[torch.Tensor] = None
        self.is_processing_request = False
        self._init_env()

    def _init_env(self):
        print("Initializing environment...")

        print(f"Determined device for env_cfg: {self.sim_args.device}")

        env_cfg = parse_env_cfg(
            self.sim_args.task,
            device=self.sim_args.device,
            num_envs=self.sim_args.num_envs,
            use_fabric=not self.sim_args.disable_fabric,
        )
        if self.sim_args.env_config_file:
            print(f"Loading custom env config from: {self.sim_args.env_config_file}")
            with open(self.sim_args.env_config_file, "r") as f:
                env_new_cfg = yaml.safe_load(f)
                dynamic_set_attr(env_cfg, env_new_cfg, path=["env_cfg"])

        self.env = gym.make(self.sim_args.task, cfg=env_cfg)
        print(
            f"Environment '{self.sim_args.task}' created. Device: {self.env.unwrapped.device}"
        )

    async def simulation_worker(self):
        print("Simulation worker started.")
        app_interface = get_app()
        while self.running and simulation_app.is_running():
            request_processed_this_cycle = False
            if not self.requests_queue.empty():
                if (
                    self.mode == 2 and not self.is_processing_request
                ) or self.mode == 1:
                    try:
                        request = self.requests_queue.get_nowait()
                        self.is_processing_request = True
                        request_processed_this_cycle = True
                        print(f"Worker processing request: {request['type']}")
                        if request["type"] == "reset":
                            obs, info = self.env.reset()
                            obs_policy = (
                                obs["policy"]
                                if isinstance(obs, dict) and "policy" in obs
                                else obs
                            )
                            await self.results_queue.put(
                                {"obs": obs_policy.cpu().numpy(), "info": info}
                            )
                            print("Worker: Reset complete.")
                        elif request["type"] == "step":
                            action = request["action"]
                            obs, reward, done, trunc, info = self.env.step(action)
                            obs_policy = (
                                obs["policy"]
                                if isinstance(obs, dict) and "policy" in obs
                                else obs
                            )
                            await self.results_queue.put(
                                {
                                    "obs": obs_policy.cpu().numpy(),
                                    "reward": reward.cpu().numpy(),
                                    "done": done.cpu().numpy(),
                                    "trunc": trunc.cpu().numpy(),
                                    "info": info,
                                }
                            )
                            self.current_action = None
                            print(
                                f"Worker: Step with action complete. Done: {done.any().item()}"
                            )
                        self.requests_queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                    except Exception as e:
                        print(f"Error in simulation_worker request processing: {e}")
                        if self.is_processing_request:
                            await self.results_queue.put({"error": str(e)})
                    finally:
                        self.is_processing_request = False

            if self.mode == 1 and not request_processed_this_cycle:
                action_to_take = self.current_action
                _ = self.env.step(action_to_take)
                self.current_action = None

            await app_interface.next_update_async()
        print("Simulation worker stopped.")

    def close(self):
        print("Closing SimulationManager...")
        self.running = False
        if self.env:
            self.env.close()
            print("Environment closed.")


# --- aiohttp Handlers ---
# ... (aiohttp handler (handle_reset, handle_step) 实现保持不变) ...
async def handle_reset(request: web.Request) -> web.Response:
    sim_manager: SimulationManager = request.app["sim_manager"]
    print("HTTP: Received /reset request.")
    if sim_manager.mode == 2 and sim_manager.is_processing_request:
        return web.json_response(
            {"error": "Server busy, previous request in progress."}, status=503
        )

    await sim_manager.requests_queue.put({"type": "reset"})
    result = await sim_manager.results_queue.get()
    sim_manager.results_queue.task_done()
    if "error" in result:
        return web.json_response({"error": result["error"]}, status=500)
    print("HTTP: Sending reset response.")
    return web.json_response({"obs": result["obs"].tolist(), "info": result["info"]})


async def handle_step(request: web.Request) -> web.Response:
    sim_manager: SimulationManager = request.app["sim_manager"]
    print("HTTP: Received /step request.")

    if sim_manager.mode == 2 and sim_manager.is_processing_request:
        return web.json_response(
            {"error": "Server busy, previous request in progress."}, status=503
        )

    try:
        action_bytes = await request.read()
        # action_dict_numpy will be a dictionary of NumPy arrays
        action_dict_numpy = pickle.loads(action_bytes)
        print(
            f"HTTP: Action dictionary (NumPy) deserialized. Keys: {list(action_dict_numpy.keys())}"
        )

        if not isinstance(action_dict_numpy, dict):
            raise TypeError("Deserialized action is not a dictionary.")

        action_tensors_dict = {}
        # Convert each NumPy array in the dictionary to a PyTorch tensor
        for key, numpy_array in action_dict_numpy.items():
            if not isinstance(numpy_array, np.ndarray):
                raise TypeError(f"Value for key '{key}' is not a NumPy array.")

            action_tensors_dict[key] = torch.tensor(
                numpy_array,
                device=sim_manager.env.unwrapped.device,
                dtype=torch.float32,
            )
            print(
                f"  Converted '{key}': shape: {action_tensors_dict[key].shape}, device: {action_tensors_dict[key].device}"
            )
    except Exception as e:
        print(f"HTTP: Error deserializing action: {e}")
        return web.json_response(
            {"error": f"Failed to deserialize action: {e}"}, status=400
        )

    if sim_manager.mode == 1:
        sim_manager.current_action = action_tensors_dict
        print("HTTP (Auto Mode): Action queued for next auto-step.")
        await sim_manager.requests_queue.put(
            {"type": "step", "action": action_tensors_dict}
        )
        result = await sim_manager.results_queue.get()
        sim_manager.results_queue.task_done()
        if "error" in result:
            return web.json_response({"error": result["error"]}, status=500)
        print("HTTP (Auto Mode): Sending step response for explicit request.")
        return web.Response(
            body=pickle.dumps(result), content_type="application/python-pickle"
        )
    else:  # Manual mode
        await sim_manager.requests_queue.put(
            {"type": "step", "action": action_tensors_dict}
        )
        result = await sim_manager.results_queue.get()
        sim_manager.results_queue.task_done()
        if "error" in result:
            return web.json_response({"error": result["error"]}, status=500)
        print("HTTP (Manual Mode): Sending step response.")
        return web.Response(
            body=pickle.dumps(result), content_type="application/python-pickle"
        )


async def on_shutdown_web_app(app: web.Application) -> None:
    sim_manager: SimulationManager = app["sim_manager"]
    print("HTTP server on_shutdown triggered...")
    if sim_manager:
        sim_manager.close()


async def setup_server_and_simulation(
    cli_args, sim_app_instance, sim_manager_instance: SimulationManager
):  # Changed last param
    """
    Sets up the aiohttp server and schedules the simulation worker.
    This function itself will be run as an asyncio task on the main Kit loop.
    """
    print("setup_server_and_simulation started.")

    web_app = web.Application()
    web_app["sim_manager"] = sim_manager_instance  # Use passed instance
    web_app.add_routes(
        [web.post("/reset", handle_reset), web.post("/step", handle_step)]
    )
    web_app.on_shutdown.append(on_shutdown_web_app)

    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, cli_args.host, cli_args.port)

    # Get the current event loop (which should be the one Kit is using)
    loop = asyncio.get_event_loop()
    kit_app_interface = get_app()  # 获取Kit应用接口

    try:
        await site.start()
        print(f"Web server started on http://{cli_args.host}:{cli_args.port}")

        # Schedule the simulation worker using standard asyncio on Kit's loop
        print("Scheduling simulation_worker task...")
        loop.create_task(sim_manager_instance.simulation_worker())  # MODIFIED HERE
        print("Simulation worker task created and scheduled.")

        while sim_manager_instance.running and sim_app_instance.is_running():
            await (
                kit_app_interface.next_update_async()
            )  # <<< 修改点：使用 Kit 的方式让出控制权

    except asyncio.CancelledError:
        print("setup_server_and_simulation task was cancelled.")
    except Exception as e:
        print(f"Error in setup_server_and_simulation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("setup_server_and_simulation finishing up...")
        if "site" in locals() and hasattr(site, "_server") and site._server is not None:
            print("Stopping web server...")
            await site.stop()
            print("Web server stopped.")
        if "runner" in locals():
            await runner.cleanup()
            print("Web app runner cleaned up.")
        print("setup_server_and_simulation cleanup complete.")


if __name__ == "__main__":
    g_sim_manager = SimulationManager(args_cli)
    main_setup_task_future = None  # To store the future/task object

    try:
        print("Getting current event loop for main task scheduling...")
        # Omniverse/Kit should have already set up and started an asyncio event loop.
        # We get a reference to it.
        event_loop = asyncio.get_event_loop()
        if not event_loop.is_running():
            # This case should ideally not happen if AppLauncher correctly starts Kit's loop.
            # If it's not running, starting it here might conflict later.
            # For now, we assume AppLauncher has it running.
            print(
                "Warning: Event loop obtained from asyncio.get_event_loop() is not running. Kit might not be fully initialized for asyncio."
            )

        print("Scheduling setup_server_and_simulation task on the event loop...")
        # Use asyncio.ensure_future or loop.create_task to schedule the main setup coroutine.
        # This returns a Task/Future object.
        main_setup_task_future = asyncio.ensure_future(  # MODIFIED HERE
            setup_server_and_simulation(args_cli, simulation_app, g_sim_manager),
            loop=event_loop,
        )
        print("setup_server_and_simulation task has been scheduled.")

        print("Starting Isaac Sim main update loop...")
        while simulation_app.is_running():
            if not g_sim_manager.running:
                print("SimulationManager is no longer running. Exiting main loop.")
                if main_setup_task_future and not main_setup_task_future.done():
                    print("Cancelling main_setup_task_future as sim_manager stopped.")
                    main_setup_task_future.cancel()
                break

            simulation_app.update()  # This drives Kit's event loop and our scheduled tasks.

            # Check if the main_setup_task_future encountered an unhandled exception
            if (
                main_setup_task_future
                and main_setup_task_future.done()
                and not main_setup_task_future.cancelled()
            ):
                if main_setup_task_future.exception():
                    print("Main setup task ended with an exception. Raising it.")
                    # Raise the exception to stop the main loop and trigger finally block
                    raise main_setup_task_future.exception()

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Shutting down...")
        if main_setup_task_future and not main_setup_task_future.done():
            main_setup_task_future.cancel()
    except Exception as e:
        print(f"Unhandled exception in __main__ loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Main loop finished or interrupted. Starting final cleanup...")

        if g_sim_manager and g_sim_manager.running:
            print("Closing SimulationManager from main finally block...")
            g_sim_manager.close()

        # Attempt to await the main_setup_task_future if it was created and needs cleanup,
        # but this is tricky outside an async def.
        # The cancellation and its internal finally block should handle most cleanup.
        if main_setup_task_future and not main_setup_task_future.done():
            print("Main setup task was still running, ensuring it's cancelled.")
            main_setup_task_future.cancel()
            # Give a moment for cancellation to propagate if possible
            if simulation_app.is_running():
                for _ in range(5):
                    simulation_app.update()
                    time.sleep(0.01)

        print("Closing Isaac Sim application...")
        if simulation_app:
            simulation_app.close()

        print("Application shutdown complete. Exiting.")
