from __future__ import annotations
import argparse
import asyncio
import numpy as np
import pickle
from aiohttp import web
import multiprocessing
import time

from isaaclab.app import AppLauncher  # Import for adding args

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

async def handle_get_observation(request: web.Request) -> web.Response:
    """
    Request payload:
    {
        "key": Optional[[List[str]] = None,     # 用点号分隔的路径（如 ["rgb.front_camera",'depth"]）
        "env_id": Optional[Union[int, List[int]]] = None  # 指定环境ID（支持单个或多个）
    }
    """
    loop = asyncio.get_event_loop()
    print("HTTP: Received /get_observation request.")
    
    if g_isaac_conn is None:
        return web.json_response({"error": "Isaac Sim process not ready."}, status=503)
    
    try:
        payload = await request.json()
        key_path = payload.get("key")
        env_id = payload.get("env_id")
        
        # 将参数直接传递给 Isaac 进程处理
        g_isaac_conn.send({
            "type": "get_observation",
            "key": key_path,
            "env_id": env_id
        })
        
        # 接收处理后的数据
        result = await loop.run_in_executor(None, g_isaac_conn.recv)
        
        if "error" in result:
            print(f"HTTP: Error from Isaac/get_observation: {result['error']}")
            return web.json_response({"error": result["error"]},
                status=400  # 或根据错误类型细化状态码
            )
        
        return web.Response(
            body=pickle.dumps(result),
            content_type="application/python-pickle"
        )
    
    except pickle.UnpicklingError as e:
        error_msg = f"Invalid Pickle payload: {str(e)}"
        print(f"HTTP: {error_msg}")
        return web.json_response({"error": error_msg},
            status=400
        )
    
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"HTTP: {error_msg}")
        return web.json_response({"error": error_msg},
            status=500
        )

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
            web.post("/get_observation", handle_get_observation)
        ]
    )

    print(f"Starting web server on http://{args_cli.host}:{args_cli.port}")
    print(f"Initial Isaac Sim mode from CLI: {SimulationMode(args_cli.mode).name}")
    web.run_app(web_app, host=args_cli.host, port=args_cli.port, access_log=None)
    print("Web server stopped.")


if __name__ == "__main__":
    main()
