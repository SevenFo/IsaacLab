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
    "--mode", type=int, default=2,
    choices=[1, 2],
    help="Running mode: 1=Auto, 2=Manual (default: Manual)"
)
parser.add_argument(
    "--task", type=str, required=True,
    help="Name of the task/environment to create"
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--env_config_file", type=str, default=None,
    help="Env Config YAML file Path, use to update default env config"
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
    "--max_episode_length",
    type=int,
    default=250,
    help="Max steps per episode on server.",
)
# Add other LunarBaseEnv constructor arguments here if needed

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

from functools import partial
import threading  # For graceful shutdown
import numpy as np
import gymnasium as gym
import torch
import yaml

from isaaclab_tasks.utils import parse_env_cfg

def dynamic_set_attr(object: object, kwargs: dict, path: list[str]):
    for k, v in kwargs.items():
        if k in object.__dict__:
            if isinstance(v, dict):
                next_path = path.copy()
                next_path.append(k)
                dynamic_set_attr(
                    object.__getattribute__(k), v, next_path
                )
            else:
                print(
                    f"set {'.'.join(path + [k])} from {object.__getattribute__(k)} to {v}"
                )
                object.__setattr__(k, v)


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


# Global variable to control server running state for graceful shutdown
SERVER_RUNNING = True
SERVER_INSTANCE = None


class LunarEnvRPyCService(rpyc.Service if RPYC_AVAILABLE else object):
    """
    """
    MODE_AUTO = 1  # 自动运行模式（空载运行，接受外部action）
    MODE_MANUAL = 2  # 手动控制模式（外部控制每次step）

    ALIASES = ["LunarBaseEnv"]  # Name clients can use to lookup the service
    env_config: dict | None = None  # 类属性用于保存配置

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env = None
        self._env_config = self.env_config  # 从类属性获取配置
        self._lock = threading.Lock()
        self._mode = self._env_config.get("mode", self.MODE_MANUAL)
        self._current_action = None
        self._auto_step_thread = None
        self._stop_auto_step = False
        self._action_available = threading.Condition(self._lock)
        self._task_name = self._env_config.get("task", None)
        print(f"[RPyC Service] Initialized with mode: {self._mode}")

    def _start_auto_step(self):
        """Starts the auto-step loop in a background thread."""
        if self._mode != self.MODE_AUTO or not self._env:
            return
            
        def auto_step_loop():
            print("[RPyC Service] Starting auto-step loop...")
            while not self._stop_auto_step and not self._env._is_closed:
                with self._lock:
                    if self._env._is_closed:
                        break
                    
                    # Check for new action
                    action = self._current_action
                    self._current_action = None
                    
                try:
                    if action is not None:
                        self._env.step(action)
                    else:
                        # Use None action for idle running
                        self._env.step(None)
                except Exception as e:
                    print(f"[RPyC Service] Error in auto-step: {e}")
                
                # 控制循环频率（可选）
                time.sleep(0.01)

            print("[RPyC Service] Auto-step loop stopped")

        self._auto_step_thread = threading.Thread(
            target=auto_step_loop, 
            daemon=True
        )
        self._auto_step_thread.start()
        
    def _ensure_env_created(self):
        """Creates the environment using gym.make if not already created."""
        if self._env is None:
            # with self._lock:
            if self._env is None:
                print("[RPyC Service] Creating environment via gym.make...")
                
                # Parse environment configuration
                try:
                    env_cfg = parse_env_cfg(
                        self._task_name,
                        device=self._env_config.get("device", "cuda:0"),
                        num_envs=self._env_config.get("num_envs", 1),
                        use_fabric=not self._env_config.get("disable_fabric", False)
                    )
                    
                    # Apply custom config if provided
                    if self._env_config.get("env_config_file"):
                        with open(self._env_config["env_config_file"], "r") as f:
                            env_new_cfg = yaml.safe_load(f)
                            dynamic_set_attr(env_cfg, env_new_cfg, path=["env_cfg"])
                    
                    # Create environment via gym.make
                    print("[RPyC Service] Making envrioment...")
                    self._env = gym.make(self._task_name, cfg=env_cfg)
                    print("[RPyC Service] Making envrioment done")

                    # Reset environment
                    print("[RPyC Service] Resetting envrioment ...")
                    self._env.reset()
                    print("[RPyC Service] Resetting envrioment done")
                    
                    
                    # Start auto-step if in auto mode
                    if self._mode == self.MODE_AUTO:
                        self._start_auto_step()
                    
                    print("[RPyC Service] Environment created successfully.")
                except Exception as e:
                    print(f"[RPyC Service] FATAL: Failed to create environment: {e}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Failed to create environment on server: {e}") from e

    def _check_env_status(self):
        """Ensures environment is created and not closed."""
        self._ensure_env_created()
        
        if self._env is None:
            raise rpyc.GenericException(
                "Environment not initialized on server.",
                "RuntimeError"
            )
            
        if self._env._is_closed:
            raise rpyc.GenericException(
                "Environment has been closed on server.",
                "RuntimeError"
            )

    def exposed_reset(self, seed: int | None = None, options: dict | None = None):
        """Resets the environment and returns the initial observation and info."""
        with self._lock:
            self._check_env_status()
            
            if self._mode == self.MODE_AUTO and self._auto_step_thread and self._auto_step_thread.is_alive():
                self._stop_auto_step = True
                self._auto_step_thread.join()
                self._start_auto_step()
            
            obs_tensor, info_dict = self._env.reset(seed=seed, options=options)
            return obs_tensor.cpu().numpy(), info_dict

    def exposed_step(self, action_numpy: np.ndarray):
        """Handles step calls based on running mode."""
        with self._lock:
            self._check_env_status()
            
            if self._mode == self.MODE_MANUAL:
                return self._process_step(action_numpy)
            elif self._mode == self.MODE_AUTO:
                self._current_action = action_numpy
                return "Action queued"

    def _process_step(self, action_numpy):
        """Internal method to process step execution."""
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

    def exposed_shutdown(self):
        """Gracefully shutdown the service."""
        print("[RPyC Service] Initiating shutdown...")
        self._stop_auto_step = True
        
        if self._auto_step_thread and self._auto_step_thread.is_alive():
            self._auto_step_thread.join(timeout=2)
        
        if self._env and not self._env._is_closed:
            self._env.close()
        
        global SERVER_RUNNING
        SERVER_RUNNING = False
        return True

    # # --- Exposed Properties for Environment Info ---
    # @property
    # def exposed_action_space_info(self) -> dict:
    #     """Returns information about the action space (serializable)."""
    #     with self._lock:
    #         self._check_env_status()
    #         space = self._env.action_space
    #     return space

    # @property
    # def exposed_observation_space_info(self) -> dict:
    #     """Returns information about the observation space (serializable)."""
    #     with self._lock:
    #         self._check_env_status()
    #         space = self._env.observation_space
    #     return {
    #         "type": "Box",  # Assuming Box space
    #         "low": np.where(
    #             np.isinf(space.low), None, space.low
    #         ).tolist(),  # Handle -inf by making it None
    #         "high": np.where(
    #             np.isinf(space.high), None, space.high
    #         ).tolist(),  # Handle inf by making it None
    #         "shape": space.shape,
    #         "dtype": str(space.dtype),
    #     }

    # @property
    # def exposed_num_envs(self) -> int:
    #     """Returns the number of parallel environments."""
    #     with self._lock:
    #         self._check_env_status()
    #     return self._env.num_envs

    # @property
    # def exposed_is_env_closed(self) -> bool:
    #     """Checks if the underlying environment instance is closed."""
    #     with self._lock:
    #         if (
    #             self._env is None
    #         ):  # If never created, it's not "closed" in the sense of having been used and shut down
    #             return False  # Or True, depending on interpretation. False = not yet active and shut down.
    #         return self._env._is_closed


def run_rpyc_server(host: str, port: int, env_config: dict):
    """
    Starts the RPyC server hosting the LunarBaseEnv service.
    """
    global SERVER_RUNNING, SERVER_INSTANCE
    if not RPYC_AVAILABLE or ThreadedServer is None:
        print("RPyC is not available. Cannot start RPyC server.")
        return

    LunarEnvRPyCService.env_config = env_config

    SERVER_INSTANCE = ThreadedServer(
        service=LunarEnvRPyCService,
        hostname=host,
        port=port,
        protocol_config={
            "allow_public_attrs": True,
            "allow_pickle": True,
            "sync_request_timeout": None,
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
    # parser = argparse.ArgumentParser(
    #     description="RPyC Server for LunarBaseEnv."
    # )
    
    if not RPYC_AVAILABLE:
        print("Exiting: RPyC library is required to run the RPyC server.")
    else:
        env_creation_config = {
            "mode": args.mode,
            "task": args.task,
            "disable_fabric": args.disable_fabric,
            "env_config_file": args.env_config_file,

            "num_envs": args.num_envs,
            "headless": args.headless,
            "device": args.device,
            "max_episode_length": args.max_episode_length,
            # Add other args from argparse to this dict if they are for LunarBaseEnv
        }
        run_rpyc_server(args.host, args.port, env_creation_config)

# # 启动自动模式服务器
# python rpyc_server.py \
#     --mode 1 \
#     --task ur5_lunar_base \
#     --num_envs 1 \
#     --port 18861 \
#     --device cuda:0

# # 启动手动模式服务器
# python rpyc_server.py \
#     --mode 2 \
#     --task ur5_lunar_base \
#     --num_envs 1 \
#     --port 18861