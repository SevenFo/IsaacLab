# rpyc_server.py

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Demo on spawning different objects in multiple environments."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import threading  # For graceful shutdown
import numpy as np

# Attempt to import RPyC
try:
    import rpyc
    from rpyc.utils.server import ThreadedServer

    RPYC_AVAILABLE = True
except ImportError:
    RPYC_AVAILABLE = False
    ThreadedServer = None  # Placeholder
    print(
        "WARNING: RPyC library not found. RPyC server functionality will be unavailable. `pip install rpyc`"
    )

# Import the environment class (assuming lunar_env.py is in the same directory or Python path)
from isaaclab_tasks.direct.franka_cabinet.ur5_lunar_base_env import (
    LunarBaseEnv,
    LunarBaseEnvCfg,
)


# Global variable to control server running state for graceful shutdown
SERVER_RUNNING = True
SERVER_INSTANCE = None


class LunarEnvRPyCService(rpyc.Service if RPYC_AVAILABLE else object):
    """
    RPyC service to expose LunarBaseEnv methods remotely.
    This service instantiates and manages ONE LunarBaseEnv instance.
    """

    ALIASES = ["LunarBaseEnv"]  # Name clients can use to lookup the service

    def __init__(self, env_config: dict | None = None):
        super().__init__()
        self._env: LunarBaseEnv | None = None
        self._env_config = env_config if env_config else {}
        self._lock = threading.Lock()  # To protect env operations if needed, though ThreadedServer handles requests in separate threads

        # Defer environment creation until on_connect or a specific init call
        # to ensure it's created in the main thread context if Isaac Sim requires it.
        # For ThreadedServer, each request is in a new thread, but Isaac Sim instance is global.
        # The env will be created when the first relevant method is called.
        print(
            "[RPyC Service] Initialized. Environment will be created on first request."
        )

    def _ensure_env_created(self):
        """Creates the environment if it hasn't been already."""
        if self._env is None:
            with self._lock:  # Ensure thread-safe creation
                if self._env is None:  # Double-check after acquiring lock
                    print("[RPyC Service] Creating LunarBaseEnv instance...")
                    try:
                        # Scene config can be customized here if needed
                        scene_cfg = LunarBaseEnvCfg()
                        # Pass only relevant args from self._env_config to LunarBaseEnv
                        env_args = {
                            k: v
                            for k, v in self._env_config.items()
                            if k in LunarBaseEnv.__init__.__code__.co_varnames
                        }
                        self._env = LunarBaseEnv(cfg=scene_cfg, **env_args)
                        print(
                            "[RPyC Service] LunarBaseEnv instance created successfully."
                        )
                    except Exception as e:
                        print(
                            f"[RPyC Service] FATAL: Failed to create LunarBaseEnv: {e}"
                        )
                        import traceback

                        traceback.print_exc()
                        # Re-raise to prevent server from appearing operational with a broken env
                        raise RuntimeError(
                            f"Failed to create LunarBaseEnv on server: {e}"
                        ) from e

    def on_connect(self, conn):
        # print(f"[RPyC Service] Client connected: {conn}")
        # Potentially initialize the environment here if not already done
        # self._ensure_env_created() # Or do it lazily on first actual call
        pass

    def on_disconnect(self, conn):
        # print(f"[RPyC Service] Client disconnected: {conn}")
        # Note: The environment is typically closed when the server shuts down,
        # not per client connection, as it's a shared resource.
        pass

    def _check_env_status(self):
        """Ensures environment is created and not closed."""
        self._ensure_env_created()  # Create if not exists
        if self._env is None:  # Should not happen if _ensure_env_created works
            raise rpyc.GenericException(
                "Environment not initialized on server.", "RuntimeError"
            )
        if self._env._is_closed:
            raise rpyc.GenericException(
                "Environment has been closed on server.", "RuntimeError"
            )

    # --- Exposed Environment Methods ---
    def exposed_reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Resets the environment and returns the initial observation and info."""
        with self._lock:
            self._check_env_status()
            obs_tensor, info_dict = self._env.reset(seed=seed, options=options)
        # Convert PyTorch tensor to NumPy array for the client
        return obs_tensor.cpu().numpy(), info_dict

    def exposed_step(
        self, action_numpy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Applies an action to the environment and steps the simulation.
        Args:
            action_numpy: A NumPy array representing the action.
        Returns:
            A tuple (observation, reward, terminated, truncated, info).
            All array-like return values are NumPy arrays.
        """
        with self._lock:
            self._check_env_status()
            # The LunarBaseEnv.step method is designed to accept NumPy or Torch tensors
            obs_tensor, reward_tensor, term_tensor, trunc_tensor, info_dict = (
                self._env.step(action_numpy)
            )

        return (
            obs_tensor.cpu().numpy(),
            reward_tensor.cpu().numpy(),
            term_tensor.cpu().numpy(),
            trunc_tensor.cpu().numpy(),
            info_dict,
        )

    def exposed_close_env_instance(self):
        """
        Closes the managed LunarBaseEnv instance.
        The RPyC server itself will continue running until shutdown_server is called.
        """
        with self._lock:
            if self._env and not self._env._is_closed:
                print(
                    "[RPyC Service] Closing LunarBaseEnv instance via remote call..."
                )
                self._env.close()
                print("[RPyC Service] LunarBaseEnv instance closed.")
            elif self._env and self._env._is_closed:
                print(
                    "[RPyC Service] LunarBaseEnv instance was already closed."
                )
            else:
                print("[RPyC Service] No environment instance to close.")
            # Optionally, set self._env to None to allow recreation if needed,
            # or keep it as closed and prevent recreation. For now, keep as closed.

    # --- Exposed Properties for Environment Info ---
    @property
    def exposed_action_space_info(self) -> dict:
        """Returns information about the action space (serializable)."""
        with self._lock:
            self._check_env_status()
            space = self._env.action_space
        return space

    @property
    def exposed_observation_space_info(self) -> dict:
        """Returns information about the observation space (serializable)."""
        with self._lock:
            self._check_env_status()
            space = self._env.observation_space
        return {
            "type": "Box",  # Assuming Box space
            "low": np.where(
                np.isinf(space.low), None, space.low
            ).tolist(),  # Handle -inf by making it None
            "high": np.where(
                np.isinf(space.high), None, space.high
            ).tolist(),  # Handle inf by making it None
            "shape": space.shape,
            "dtype": str(space.dtype),
        }

    @property
    def exposed_num_envs(self) -> int:
        """Returns the number of parallel environments."""
        with self._lock:
            self._check_env_status()
        return self._env.num_envs

    @property
    def exposed_is_env_closed(self) -> bool:
        """Checks if the underlying environment instance is closed."""
        with self._lock:
            if (
                self._env is None
            ):  # If never created, it's not "closed" in the sense of having been used and shut down
                return False  # Or True, depending on interpretation. False = not yet active and shut down.
            return self._env._is_closed


def run_rpyc_server(host: str, port: int, env_config: dict):
    """
    Starts the RPyC server hosting the LunarBaseEnv service.
    """
    global SERVER_RUNNING, SERVER_INSTANCE
    if not RPYC_AVAILABLE or ThreadedServer is None:
        print("RPyC is not available. Cannot start RPyC server.")
        return

    # The service factory now takes the env_config
    service_factory = lambda: LunarEnvRPyCService(env_config=env_config)

    SERVER_INSTANCE = ThreadedServer(
        service_factory,  # Use a factory to create service instance per connection if needed, or a single instance
        hostname=host,
        port=port,
        protocol_config={
            "allow_public_attrs": True,
            "allow_pickle": True,  # Allow pickling for complex objects if necessary (NumPy arrays)
            "sync_request_timeout": 300,  # seconds, for potentially long env steps
        },
    )

    print(f"[RPyC Server] Starting on {host}:{port}...")
    print(f"[RPyC Server] Environment config: {env_config}")
    print("[RPyC Server] Press Ctrl+C to stop.")

    try:
        # Start server in a separate thread so we can handle Ctrl+C in the main thread
        server_thread = threading.Thread(
            target=SERVER_INSTANCE.start, daemon=True
        )
        server_thread.start()
        while SERVER_RUNNING and server_thread.is_alive():
            server_thread.join(
                timeout=0.5
            )  # Keep main thread alive, checking for shutdown
    except KeyboardInterrupt:
        print("[RPyC Server] Ctrl+C received. Shutting down...")
    except Exception as e:
        print(f"[RPyC Server] An error occurred: {e}")
    finally:
        SERVER_RUNNING = False
        if SERVER_INSTANCE:
            print("[RPyC Server] Closing server instance...")
            # Closing the server instance should also trigger on_disconnect for active connections
            # and allow the service's __del__ or a custom shutdown to close the env.
            # The service needs a method to explicitly close its env.
            # Let's ensure the env is closed if the server is shut down.

            # Access the service instance to close its environment
            # This is a bit tricky as ThreadedServer manages service instances.
            # If we used a single service instance, it would be easier.
            # For now, we rely on the service instance itself to handle env closure
            # if its exposed_close_env_instance is called, or if the whole process exits.
            # A better way is to have the service instance explicitly close its env on server shutdown.
            # This is complex with rpyc's default service lifecycle.
            # The most robust way is for the client to call `close_env_instance` or for the server
            # to have a `shutdown_server` method that iterates services.

            # For a simpler model: when server stops, the process will end, and env resources should be released.
            # However, explicit close is better.
            # We expect the service instance to clean up its environment if it has one.
            # The environment (LunarBaseEnv) itself has a close method.
            # The primary issue is calling that close method.
            # The `LunarEnvRPyCService` should ideally be responsible.
            # If the server process exits, __del__ of LunarBaseEnv *might* be called, but it's not guaranteed
            # to be timely or in the right context for Isaac Sim cleanup.

            # Safest: the LunarBaseEnv itself must be robust to process exit.
            # AppLauncher's close() is the key, called by LunarBaseEnv.close().

            # Let's assume that if the server is stopped, the env will eventually be closed by process termination.
            # Proper cleanup of the Isaac Sim app is critical. LunarBaseEnv.close() handles this.
            # The challenge is ensuring LunarBaseEnv.close() is called.
            # LunarEnvRPyCService.exposed_close_env_instance() provides the way.

            SERVER_INSTANCE.close()  # Stop accepting new connections and close existing ones.
        print("[RPyC Server] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RPyC Server for LunarBaseEnv."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Hostname to bind the server.",
    )
    parser.add_argument(
        "--port", type=int, default=18861, help="Port to bind the server."
    )
    # Environment configuration arguments (passed to LunarBaseEnv)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments for the server's env instance.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim in headless mode for the server.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for PyTorch on server (e.g., cuda:0, cpu).",
    )
    parser.add_argument(
        "--max_episode_length",
        type=int,
        default=250,
        help="Max steps per episode on server.",
    )
    # Add other LunarBaseEnv constructor arguments here if needed

    args = parser.parse_args()

    if not RPYC_AVAILABLE:
        print("Exiting: RPyC library is required to run the RPyC server.")
    else:
        env_creation_config = {
            "num_envs": args.num_envs,
            "headless": args.headless,
            "device": args.device,
            "max_episode_length": args.max_episode_length,
            # Add other args from argparse to this dict if they are for LunarBaseEnv
        }
        run_rpyc_server(args.host, args.port, env_creation_config)
