"""
Simulator component that runs gym environment in a subprocess.
"""

import time
import multiprocessing as mp
from multiprocessing import Queue, Process
from typing import Optional, Dict, Any, Tuple
import gymnasium as gym
import numpy as np
from .types import Action, Observation, SystemStatus


class SimulatorProcess:
    """Simulator process that runs the gym environment."""

    def __init__(self, env_name: str, render_mode: Optional[str] = None):
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = None

    def run(self, command_queue: Queue, result_queue: Queue, obs_queue: Queue):
        """Main process loop for the simulator."""
        try:
            # Create environment
            self.env = gym.make(self.env_name, render_mode=self.render_mode)

            # Send initial info
            info = {
                "action_space": self._serialize_space(self.env.action_space),
                "observation_space": self._serialize_space(
                    self.env.observation_space
                ),
                "spec": str(self.env.spec) if self.env.spec else None,
            }
            result_queue.put(("init_success", info))

            # Reset environment
            obs, info = self.env.reset()
            obs_queue.put(("observation", self._serialize_obs(obs), info))

            # Main loop
            running = True
            while running:
                try:
                    if not command_queue.empty():
                        command, data = command_queue.get_nowait()

                        if command == "step":
                            action = self._deserialize_action(data)
                            obs, reward, terminated, truncated, info = (
                                self.env.step(action)
                            )
                            result_queue.put(
                                (
                                    "step_result",
                                    {
                                        "observation": self._serialize_obs(
                                            obs
                                        ),
                                        "reward": reward,
                                        "terminated": terminated,
                                        "truncated": truncated,
                                        "info": info,
                                    },
                                )
                            )
                            obs_queue.put(
                                ("observation", self._serialize_obs(obs), info)
                            )

                        elif command == "reset":
                            obs, info = self.env.reset()
                            result_queue.put(
                                (
                                    "reset_result",
                                    {
                                        "observation": self._serialize_obs(
                                            obs
                                        ),
                                        "info": info,
                                    },
                                )
                            )
                            obs_queue.put(
                                ("observation", self._serialize_obs(obs), info)
                            )

                        elif command == "close":
                            running = False

                        elif command == "render":
                            if hasattr(self.env, "render"):
                                frame = self.env.render()
                                result_queue.put(("render_result", frame))

                    time.sleep(0.001)  # Small sleep to prevent busy waiting

                except Exception as e:
                    result_queue.put(("error", str(e)))

        except Exception as e:
            result_queue.put(("init_error", str(e)))
        finally:
            if self.env:
                self.env.close()

    def _serialize_space(self, space) -> Dict[str, Any]:
        """Serialize gym space for IPC."""
        if hasattr(space, "shape"):
            return {
                "type": type(space).__name__,
                "shape": space.shape,
                "dtype": str(space.dtype) if hasattr(space, "dtype") else None,
                "low": space.low.tolist() if hasattr(space, "low") else None,
                "high": space.high.tolist()
                if hasattr(space, "high")
                else None,
            }
        elif hasattr(space, "spaces"):  # Dict or Tuple space
            return {
                "type": type(space).__name__,
                "spaces": {
                    k: self._serialize_space(v)
                    for k, v in space.spaces.items()
                },
            }
        else:
            return {"type": type(space).__name__, "data": str(space)}

    def _serialize_obs(self, obs) -> Any:
        """Serialize observation for IPC."""
        if isinstance(obs, np.ndarray):
            return obs.tolist()
        elif isinstance(obs, dict):
            return {k: self._serialize_obs(v) for k, v in obs.items()}
        else:
            return obs

    def _deserialize_action(self, action_data) -> Any:
        """Deserialize action from IPC."""
        if isinstance(action_data, list):
            return np.array(action_data)
        elif isinstance(action_data, dict):
            return {
                k: self._deserialize_action(v) for k, v in action_data.items()
            }
        else:
            return action_data


class Simulator:
    """Main simulator interface running environment in subprocess."""

    def __init__(self, env_name: str, render_mode: Optional[str] = None):
        self.env_name = env_name
        self.render_mode = render_mode

        # Process management
        self.process: Optional[Process] = None
        self.command_queue: Optional[Queue] = None
        self.result_queue: Optional[Queue] = None
        self.obs_queue: Optional[Queue] = None

        # Environment info
        self.action_space_info: Optional[Dict[str, Any]] = None
        self.observation_space_info: Optional[Dict[str, Any]] = None
        self.latest_observation: Optional[Observation] = None

        # Status
        self.status = SystemStatus.IDLE
        self.is_initialized = False

    def initialize(self) -> bool:
        """Initialize the simulator subprocess."""
        try:
            # Create queues for IPC
            self.command_queue = Queue()
            self.result_queue = Queue()
            self.obs_queue = Queue()

            # Start simulator process
            sim_process = SimulatorProcess(self.env_name, self.render_mode)
            self.process = Process(
                target=sim_process.run,
                args=(self.command_queue, self.result_queue, self.obs_queue),
            )
            self.process.start()

            # Wait for initialization
            result_type, data = self.result_queue.get(timeout=10)

            if result_type == "init_success":
                self.action_space_info = data["action_space"]
                self.observation_space_info = data["observation_space"]
                self.is_initialized = True
                self.status = SystemStatus.IDLE

                # Get initial observation
                self._update_observation()
                return True
            else:
                print(f"Simulator initialization failed: {data}")
                return False

        except Exception as e:
            print(f"Failed to initialize simulator: {e}")
            return False

    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """Step the environment with given action."""
        if not self.is_initialized:
            raise RuntimeError("Simulator not initialized")

        # Convert action to format expected by env
        action_data = self._convert_action(action)

        # Send step command
        self.command_queue.put(("step", action_data))

        # Get result
        result_type, data = self.result_queue.get(timeout=5)

        if result_type == "step_result":
            obs = Observation(data=data["observation"], timestamp=time.time())
            self.latest_observation = obs
            return (
                obs,
                data["reward"],
                data["terminated"],
                data["truncated"],
                data["info"],
            )
        else:
            raise RuntimeError(f"Step failed: {data}")

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        """Reset the environment."""
        if not self.is_initialized:
            raise RuntimeError("Simulator not initialized")

        self.command_queue.put(("reset", None))
        result_type, data = self.result_queue.get(timeout=5)

        if result_type == "reset_result":
            obs = Observation(data=data["observation"], timestamp=time.time())
            self.latest_observation = obs
            return obs, data["info"]
        else:
            raise RuntimeError(f"Reset failed: {data}")

    def get_observation(self) -> Optional[Observation]:
        """Get the latest observation."""
        self._update_observation()
        return self.latest_observation

    def get_action_space_info(self) -> Optional[Dict[str, Any]]:
        """Get action space information."""
        return self.action_space_info

    def get_observation_space_info(self) -> Optional[Dict[str, Any]]:
        """Get observation space information."""
        return self.observation_space_info

    def create_random_action(self) -> Action:
        """Create a random action based on action space."""
        if not self.action_space_info:
            raise RuntimeError("Action space info not available")

        space_info = self.action_space_info
        if space_info["type"] == "Box":
            shape = space_info["shape"]
            low = np.array(space_info["low"]) if space_info["low"] else -1
            high = np.array(space_info["high"]) if space_info["high"] else 1
            action_data = np.random.uniform(low, high, shape)
        elif space_info["type"] == "Discrete":
            action_data = np.random.randint(0, space_info.get("n", 2))
        else:
            # Default random action
            action_data = np.array([0.0])

        return Action(data=action_data)

    def shutdown(self):
        """Shutdown the simulator subprocess."""
        if self.process and self.process.is_alive():
            self.command_queue.put(("close", None))
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()

        self.is_initialized = False
        self.status = SystemStatus.SHUTDOWN

    def _update_observation(self):
        """Update latest observation from obs queue."""
        try:
            while not self.obs_queue.empty():
                obs_type, obs_data, info = self.obs_queue.get_nowait()
                if obs_type == "observation":
                    self.latest_observation = Observation(
                        data=obs_data, timestamp=time.time(), metadata=info
                    )
        except:
            pass

    def _convert_action(self, action: Action) -> Any:
        """Convert Action object to format expected by gym env."""
        if isinstance(action.data, np.ndarray):
            return action.data.tolist()
        else:
            return action.data
