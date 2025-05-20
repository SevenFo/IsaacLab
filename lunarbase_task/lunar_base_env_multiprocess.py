from __future__ import annotations
import argparse
import asyncio
import numpy as np
import pickle
from aiohttp import web
import multiprocessing
import time

from isaac_process import isaac_simulation_entry, SimulationMode  # Import Enum too

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Async HTTP Server with Multi-Process Isaac Sim"
)
parser.add_argument(
    "--mode",
    type=int,
    default=SimulationMode.MANUAL_STEP.value,  # Default to MANUAL_STEP
    choices=[mode.value for mode in SimulationMode],
    help=(
        "Initial running mode: "
        f"{SimulationMode.MANUAL_STEP.value}=Manual (step on request), "
        f"{SimulationMode.AUTO_STEP.value}=Auto (step continuously). "
        f"(default: {SimulationMode.MANUAL_STEP.value})"
    ),
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

# Add AppLauncher args (needed for the isaac_process)
# You might need to manually add relevant ones or find a way to pass them if AppLauncher isn't directly used here
# For simplicity, let's assume the isaac_process.py will also parse its own AppLauncher args
# or we pass the full args_cli to it.
# Let's assume AppLauncher.add_app_launcher_args(parser) is done before parsing if needed by isaac_process
# For now, we'll pass the parsed args_cli to the child process.
# The AppLauncher specific args will be parsed within isaac_process.py when AppLauncher is initialized.

# It's better if isaac_process.py also defines its own AppLauncher args if it's a separate script.
# For now, we'll pass the full args_cli.
# from isaaclab.app import AppLauncher # This line might be problematic if AppLauncher tries to init early
# AppLauncher.add_app_launcher_args(parser) # If isaac_process directly uses cli_args for AppLauncher

from isaaclab.app import AppLauncher  # Import for adding args

AppLauncher.add_app_launcher_args(parser)  # Add app launcher args to the parser
args_cli = parser.parse_args()  # Re-parse with app launcher args

# Global variable for the parent connection to the Isaac Sim process
# and the process itself
g_isaac_conn = None
g_isaac_process = None


async def handle_reset(request: web.Request) -> web.Response:
    loop = asyncio.get_event_loop()
    print("HTTP: Received /reset request.")
    if g_isaac_conn is None:
        return web.json_response({"error": "Isaac Sim process not ready."}, status=503)
    try:
        g_isaac_conn.send({"type": "reset"})
        result = await loop.run_in_executor(None, g_isaac_conn.recv)
        if "error" in result:
            print(f"HTTP: Error from Isaac/reset: {result['error']}")
            return web.json_response({"error": result["error"]}, status=500)
        print("HTTP: Sending reset response.")
        return web.json_response(
            {
                "obs": result["obs"].tolist(),
                "info": result["info"],
            }  # Assuming obs is numpy
        )
    except Exception as e:
        print(f"HTTP: Exception during /reset: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_step(request: web.Request) -> web.Response:
    loop = asyncio.get_event_loop()
    print("HTTP: Received /step request.")
    if g_isaac_conn is None:
        return web.json_response({"error": "Isaac Sim process not ready."}, status=503)
    try:
        action_bytes = await request.read()
        action_dict_numpy = pickle.loads(action_bytes)
        # Basic validation can be added here for action_dict_numpy
        g_isaac_conn.send({"type": "step", "action": action_dict_numpy})
        result = await loop.run_in_executor(None, g_isaac_conn.recv)
        if "error" in result:
            print(f"HTTP: Error from Isaac/step: {result['error']}")
            return web.json_response({"error": result["error"]}, status=500)
        print("HTTP: Sending step response.")
        return web.Response(
            body=pickle.dumps(result), content_type="application/python-pickle"
        )
    except Exception as e:
        print(f"HTTP: Exception during /step: {e}")
        return web.json_response({"error": f"Failed to process step: {e}"}, status=400)


async def handle_set_mode(request: web.Request) -> web.Response:
    loop = asyncio.get_event_loop()
    print("HTTP: Received /set_mode request.")
    if g_isaac_conn is None:
        return web.json_response({"error": "Isaac Sim process not ready."}, status=503)
    try:
        data = await request.json()
        new_mode_val = data.get("mode")
        if new_mode_val is None:
            return web.json_response(
                {"error": "Missing 'mode' in request body."}, status=400
            )

        try:
            # Validate if new_mode_val is a valid SimulationMode value
            SimulationMode(new_mode_val)
        except ValueError:
            return web.json_response(
                {
                    "error": f"Invalid mode value: {new_mode_val}. "
                    f"Valid modes are {[m.value for m in SimulationMode]}"
                },
                status=400,
            )

        g_isaac_conn.send({"type": "set_mode", "mode": new_mode_val})
        result = await loop.run_in_executor(None, g_isaac_conn.recv)

        if "error" in result:
            print(f"HTTP: Error from Isaac/set_mode: {result['error']}")
            return web.json_response({"error": result["error"]}, status=500)

        print(f"HTTP: Mode set response: {result}")
        return web.json_response(result)  # Send back the status from isaac_process
    except Exception as e:
        print(f"HTTP: Exception during /set_mode: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def on_startup_web_app(app: web.Application):
    global g_isaac_conn, g_isaac_process
    print("Web App Startup: Initializing IPC and Isaac Sim process...")
    parent_conn, child_conn = multiprocessing.Pipe()
    g_isaac_conn = parent_conn
    g_isaac_process = multiprocessing.Process(
        target=isaac_simulation_entry, args=(child_conn, args_cli)
    )
    g_isaac_process.start()
    print(
        f"Isaac Sim process started (PID: {g_isaac_process.pid}). Mode: {SimulationMode(args_cli.mode).name}"
    )


async def on_shutdown_web_app(app: web.Application):
    global g_isaac_conn, g_isaac_process
    print("Web App Shutdown: Signaling Isaac Sim process to close...")
    if g_isaac_conn:
        try:
            g_isaac_conn.send({"type": "close"})
            print("Close command sent to Isaac Sim process.")
        except Exception as e:
            print(f"Error sending close command: {e}")
        finally:
            g_isaac_conn.close()
    if g_isaac_process and g_isaac_process.is_alive():
        print("Waiting for Isaac Sim process to terminate...")
        g_isaac_process.join(timeout=30)
        if g_isaac_process.is_alive():
            print("Isaac Sim process did not terminate gracefully, attempting to kill.")
            g_isaac_process.terminate()
            g_isaac_process.join(timeout=5)
        if not g_isaac_process.is_alive():
            print("Isaac Sim process terminated.")
        else:
            print("Isaac Sim process could not be terminated.")
    elif g_isaac_process:
        print("Isaac Sim process already terminated.")


def main():
    # It's generally good practice to set the start method for multiprocessing,
    # especially if you plan to run on different OS.
    # 'spawn' is often recommended for cross-platform compatibility and safety,
    # as it starts a fresh process rather than forking.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Multiprocessing start method already set or cannot be changed.")
        pass  # Already set or not possible to change (e.g. if already started)

    web_app = web.Application()
    web_app.on_startup.append(on_startup_web_app)
    web_app.on_shutdown.append(on_shutdown_web_app)

    web_app.add_routes(
        [
            web.post("/reset", handle_reset),
            web.post("/step", handle_step),
            web.post("/set_mode", handle_set_mode),  # Add new route
        ]
    )

    print(f"Starting web server on http://{args_cli.host}:{args_cli.port}")
    print(f"Initial Isaac Sim mode from CLI: {SimulationMode(args_cli.mode).name}")
    web.run_app(web_app, host=args_cli.host, port=args_cli.port, access_log=None)
    print("Web server stopped.")


if __name__ == "__main__":
    main()
