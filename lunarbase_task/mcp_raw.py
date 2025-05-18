# mcp_server.py
import argparse
import socket
import pickle
import struct
import json  # Alternative serialization for simple data
import threading
import numpy as np
import torch  # For type hinting

# Import the environment class
try:
    from lunar_env import LunarBaseEnv, MultiObjectSceneCfg
except ImportError as e:
    print(f"ERROR: Could not import LunarBaseEnv or MultiObjectSceneCfg: {e}")
    print("Ensure lunar_env.py is in the Python path or current directory.")
    LunarBaseEnv = None

# Global control for server
MCP_SERVER_RUNNING = True
MCP_ENV_INSTANCE: LunarBaseEnv | None = None
MCP_ENV_LOCK = threading.Lock()  # Protects access to MCP_ENV_INSTANCE


# --- MCP Message Protocol Helpers ---
def send_mcp_message(sock, message: dict):
    """Serializes (pickle) and sends a message dictionary with its length prefixed."""
    try:
        payload = pickle.dumps(message)
        header = struct.pack(
            ">I", len(payload)
        )  # 4-byte unsigned int, network byte order
        sock.sendall(header + payload)
    except (BrokenPipeError, ConnectionResetError):
        # print("[MCP Send] Client connection lost.")
        raise  # Re-raise to be handled by caller
    except Exception as e:
        print(f"[MCP Send] Error sending message: {e}")
        raise


def receive_mcp_message(sock) -> dict | None:
    """Receives and deserializes (pickle) a message dictionary."""
    try:
        header_bytes = sock.recv(4)
        if not header_bytes:
            # print("[MCP Recv] Client disconnected (no header).")
            return None
        payload_length = struct.unpack(">I", header_bytes)[0]

        payload = b""
        while len(payload) < payload_length:
            chunk = sock.recv(min(payload_length - len(payload), 4096))
            if not chunk:
                # print("[MCP Recv] Client disconnected (incomplete payload).")
                return None
            payload += chunk

        return pickle.loads(payload)
    except (ConnectionResetError, EOFError, struct.error, pickle.UnpicklingError):
        # print("[MCP Recv] Client connection lost or bad data.")
        return None  # Indicates client disconnected or sent malformed data
    except Exception as e:
        print(f"[MCP Recv] Error receiving message: {e}")
        return None


# --- MCP Tool Definitions & Handlers ---
# These functions would be called based on the 'tool_name' in the client's request.


def tool_get_environment_status(env: LunarBaseEnv | None, params: dict) -> dict:
    """MCP Tool: Returns the status of the simulation environment."""
    if env is None or env._is_closed:
        return {
            "status": "unavailable",
            "is_ready": False,
            "is_closed": True if env else False,
            "num_envs": 0,
        }
    return {
        "status": "ready",
        "is_ready": True,
        "is_closed": env._is_closed,
        "num_envs": env.num_envs,
        "observation_space_shape": env.observation_space.shape,
        "action_space_shape": env.action_space.shape,
        "max_episode_length": env._max_episode_length,
    }


def tool_get_current_observation(env: LunarBaseEnv, params: dict) -> dict:
    """MCP Tool: Fetches the current observation from a specific environment instance (if num_envs > 1)."""
    env_idx = params.get("env_idx", 0)
    if not (0 <= env_idx < env.num_envs):
        return {
            "error": f"Invalid env_idx: {env_idx}. Must be between 0 and {env.num_envs - 1}."
        }

    # _get_observations returns for all envs. We need to get the latest.
    # A direct call to _get_observations might be redundant if called right after a step.
    # We might need a way to get the *cached* latest observation for a specific env.
    # For simplicity, let's call _get_observations and select.
    # This assumes the caller (LLM) doesn't need ultra-low latency for this query.
    all_obs_tensor = (
        env._get_observations()
    )  # This ensures data is current if called independently
    observation_numpy = all_obs_tensor[env_idx].cpu().numpy()
    return {
        "env_idx": env_idx,
        "observation": observation_numpy.tolist(),
    }  # Serialize to list


def tool_get_object_z_position(env: LunarBaseEnv, params: dict) -> dict:
    """MCP Tool: Gets the Z position of the target object in a specific environment."""
    env_idx = params.get("env_idx", 0)
    if not (0 <= env_idx < env.num_envs):
        return {"error": f"Invalid env_idx: {env_idx}."}

    # Ensure latest data (scene.update is called in env.step or env.reset)
    # If this tool is called out of sync, we might need to call scene.update() or get fresh data.
    # For MCP, we assume queries are less frequent than sim steps.
    # The data in env.object.data should be from the last sim step.
    object_z_pos = env.object.data.root_pos_w[
        env_idx, 2
    ].item()  # .item() to get Python float
    return {"env_idx": env_idx, "object_z_position": object_z_pos}


def tool_describe_task(env: LunarBaseEnv, params: dict) -> dict:
    """MCP Tool: Returns a textual description of the current task."""
    # This would be a human-readable description you define.
    description = (
        "The task is to control a UR5-like robotic arm with a gripper in a simulated lunar base environment. "
        "A target object (cone, cuboid, or sphere) is spawned in front of the robot. "
        "The primary goal is to manipulate this object, for example, to lift it to a target height. "
        f"The environment has {env.num_envs} parallel instances. "
        f"Observations include robot joint states and the object's pose and velocity. "
        f"Actions control the robot's arm and gripper joints."
    )
    return {"task_description": description}


# Add more tools as needed, e.g., for executing high-level actions
# def tool_execute_high_level_action(env: LunarBaseEnv, params: dict) -> dict:
#     action_name = params.get("action_name") # e.g., "lift_object", "move_to_object"
#     # ... translate high-level action to low-level env.step() calls ...
#     # This would be complex and task-specific.
#     return {"status": "action_executed", "details": "..."}


MCP_TOOLS_REGISTRY = {
    "get_environment_status": tool_get_environment_status,
    "get_current_observation": tool_get_current_observation,
    "get_object_z_position": tool_get_object_z_position,
    "describe_task": tool_describe_task,
    # "execute_high_level_action": tool_execute_high_level_action,
}


def handle_mcp_client_connection(client_socket: socket.socket, client_address: tuple):
    """Handles a single client connection for the MCP-style server."""
    global MCP_ENV_INSTANCE, MCP_ENV_LOCK
    print(f"[MCP Server] Client connected: {client_address}")

    try:
        while MCP_SERVER_RUNNING:
            request_message = receive_mcp_message(client_socket)
            if request_message is None:
                print(
                    f"[MCP Server] Client {client_address} disconnected or sent malformed message."
                )
                break  # Client disconnected or error in receiving

            # --- Standard MCP Request Structure (Conceptual) ---
            # {
            #   "protocol_version": "0.1-alpha", // Or similar
            #   "request_id": "some_uuid",
            #   "tool_name": "name_of_the_tool_to_call",
            #   "tool_input": { ... parameters for the tool ... }
            # }
            # For our simplified version, we'll use:
            # { "tool_name": "...", "tool_input": { ... } }

            tool_name = request_message.get("tool_name")
            tool_input_params = request_message.get("tool_input", {})

            response_payload = {}
            status_code = 200  # HTTP-like status codes for tools

            with MCP_ENV_LOCK:  # Ensure exclusive access to the shared env instance
                if tool_name in MCP_TOOLS_REGISTRY:
                    tool_function = MCP_TOOLS_REGISTRY[tool_name]
                    try:
                        # Ensure env is available for tools that need it
                        if (
                            MCP_ENV_INSTANCE is None
                            and tool_name != "get_environment_status"
                        ):  # status can run without full env
                            raise RuntimeError(
                                "Environment instance not available on server."
                            )
                        if (
                            MCP_ENV_INSTANCE
                            and MCP_ENV_INSTANCE._is_closed
                            and tool_name != "get_environment_status"
                        ):
                            raise RuntimeError(
                                "Environment instance has been closed on server."
                            )

                        # Call the tool
                        print(
                            f"[MCP Server] Executing tool: {tool_name} with input: {tool_input_params}"
                        )
                        tool_result = tool_function(MCP_ENV_INSTANCE, tool_input_params)
                        response_payload = {
                            "tool_name": tool_name,
                            "tool_output": tool_result,
                        }
                    except Exception as e:
                        print(f"[MCP Server] Error executing tool '{tool_name}': {e}")
                        status_code = 500  # Internal Server Error
                        response_payload = {
                            "tool_name": tool_name,
                            "error": {"type": type(e).__name__, "message": str(e)},
                        }
                elif tool_name == "ping":  # Simple diagnostic tool
                    response_payload = {"tool_name": "ping", "tool_output": "pong"}
                elif tool_name == "shutdown_simulation_environment":  # Special command
                    if MCP_ENV_INSTANCE and not MCP_ENV_INSTANCE._is_closed:
                        print(
                            "[MCP Server] Received request to shut down simulation environment."
                        )
                        MCP_ENV_INSTANCE.close()
                        response_payload = {
                            "tool_name": tool_name,
                            "tool_output": "Simulation environment shutdown initiated.",
                        }
                    else:
                        response_payload = {
                            "tool_name": tool_name,
                            "tool_output": "Simulation environment already closed or not initialized.",
                        }
                else:
                    status_code = 404  # Not Found
                    response_payload = {
                        "error": {"message": f"Tool '{tool_name}' not found."}
                    }

            # --- Standard MCP Response Structure (Conceptual) ---
            # {
            #   "protocol_version": "0.1-alpha",
            #   "request_id": "echo_request_id", // From original request
            #   "status_code": 200, // Or error code
            #   "tool_name": "name_of_the_tool_called",
            #   "tool_output": { ... result from the tool ... }, // If success
            #   "error": { ... error details ... } // If error
            # }
            # Our simplified response:
            mcp_response = {"status_code": status_code, "payload": response_payload}
            send_mcp_message(client_socket, mcp_response)

            if (
                tool_name == "shutdown_simulation_environment"
                and MCP_ENV_INSTANCE
                and MCP_ENV_INSTANCE._is_closed
            ):
                print(
                    "[MCP Server] Environment is now closed, client connection will also close."
                )
                break  # End connection as env is closed.

    except (BrokenPipeError, ConnectionResetError):
        print(f"[MCP Server] Client {client_address} connection lost.")
    except Exception as e:
        print(
            f"[MCP Server] Unhandled error in client connection {client_address}: {e}"
        )
        import traceback

        traceback.print_exc()
    finally:
        print(f"[MCP Server] Closing connection with {client_address}.")
        client_socket.close()


def run_mcp_server(host: str, port: int, env_config: dict):
    global MCP_SERVER_RUNNING, MCP_ENV_INSTANCE, MCP_ENV_LOCK

    if LunarBaseEnv is None:
        print("ERROR: LunarBaseEnv class not available. MCP Server cannot start.")
        return

    # Create the single LunarBaseEnv instance for this server
    print("[MCP Server] Initializing LunarBaseEnv instance...")
    try:
        with MCP_ENV_LOCK:
            scene_cfg = MultiObjectSceneCfg()  # Default or load from config
            mcp_env_args = {
                k: v
                for k, v in env_config.items()
                if k in LunarBaseEnv.__init__.__code__.co_varnames
            }
            MCP_ENV_INSTANCE = LunarBaseEnv(scene_cfg=scene_cfg, **mcp_env_args)
        print("[MCP Server] LunarBaseEnv instance created successfully.")
    except Exception as e:
        print(f"[MCP Server] FATAL: Failed to create LunarBaseEnv for MCP Server: {e}")
        import traceback

        traceback.print_exc()
        return  # Cannot start server without the env

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((host, port))
        server_socket.listen(5)  # Max queued connections
        print(f"[MCP Server] Listening on {host}:{port} for MCP-style connections...")
        print(f"[MCP Server] Environment config: {env_config}")
        print("[MCP Server] Press Ctrl+C to stop.")

        client_threads = []

        while MCP_SERVER_RUNNING:
            try:
                server_socket.settimeout(
                    1.0
                )  # Timeout to allow checking MCP_SERVER_RUNNING
                client_socket, client_address = server_socket.accept()
                server_socket.settimeout(None)  # Clear timeout after accept

                # Handle each client in a new thread
                # Note: All threads will share the same MCP_ENV_INSTANCE. Access must be synchronized.
                # The MCP_ENV_LOCK is used within tool handlers for this.
                thread = threading.Thread(
                    target=handle_mcp_client_connection,
                    args=(client_socket, client_address),
                    daemon=True,
                )
                client_threads.append(thread)
                thread.start()
            except socket.timeout:
                continue  # Loop back to check MCP_SERVER_RUNNING
            except KeyboardInterrupt:  # Should be caught by the outer try/except
                print("[MCP Server] Accept loop interrupted.")
                MCP_SERVER_RUNNING = False
                break
            except Exception as e:  # Other errors in accept loop
                print(f"[MCP Server] Error in accept loop: {e}")
                MCP_SERVER_RUNNING = False  # Stop server on major accept error
                break

    except KeyboardInterrupt:
        print("[MCP Server] Ctrl+C received. Shutting down server...")
    except Exception as e:
        print(f"[MCP Server] An error occurred: {e}")
    finally:
        MCP_SERVER_RUNNING = False
        print("[MCP Server] Stopping client threads...")
        for t in client_threads:
            if t.is_alive():
                # Threads are daemonic, will exit when main thread exits.
                # Or implement a more graceful shutdown for threads if needed.
                pass  # t.join(timeout=1.0) # Optional: wait for threads to finish

        with MCP_ENV_LOCK:
            if MCP_ENV_INSTANCE and not MCP_ENV_INSTANCE._is_closed:
                print("[MCP Server] Closing LunarBaseEnv instance...")
                MCP_ENV_INSTANCE.close()

        server_socket.close()
        print("[MCP Server] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP-style Server for LunarBaseEnv Tools."
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Hostname to bind the server."
    )
    parser.add_argument(
        "--port", type=int, default=18862, help="Port to bind the server."
    )  # Different default from RPyC
    # Environment configuration arguments
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments for the server's env instance (usually 1 for MCP tools).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim in headless mode for the server.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for PyTorch on server."
    )
    parser.add_argument(
        "--max_episode_length",
        type=int,
        default=250,
        help="Max steps per episode on server (less relevant for MCP tools, but part of env init).",
    )

    args = parser.parse_args()

    if LunarBaseEnv is None:
        print("Exiting: LunarBaseEnv class is required to run the MCP-style server.")
    else:
        mcp_env_creation_config = {
            "num_envs": args.num_envs,
            "headless": args.headless,
            "device": args.device,
            "max_episode_length": args.max_episode_length,
        }
        run_mcp_server(args.host, args.port, mcp_env_creation_config)
