"""
Isaac Lab simulator interface - Shared Memory + Unix Socket version.
Runs Isaac simulation in a completely independent process with:
- Unix Socket for control commands (small, infrequent)
- Shared Memory for data transfer (large, frequent - observations, actions)

This ensures AppLauncher is imported first and provides high-performance data transfer.
"""

import socket
import struct
import pickle
import threading
import tempfile
import json
import os
import time
import traceback
import subprocess
import signal
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from multiprocessing import shared_memory
from omegaconf import DictConfig, OmegaConf

# robot_brain_system imports
from robot_brain_system.core.types import (
    Action,
    Observation,
)
from robot_brain_system.utils.retry_utils import retry
from robot_brain_system.ui.console import global_console


def log_subprocess(prefix, msg):
    # 使用 info 或 system 类别，或者自定义样式
    global_console.log(prefix, f"[{prefix}] {msg}")


class SharedMemoryManager:
    """Manager for shared memory buffers used for high-performance data transfer."""

    def __init__(self, name_prefix: str):
        self.name_prefix = name_prefix
        self.shm_obs: Optional[shared_memory.SharedMemory] = None
        self.shm_action: Optional[shared_memory.SharedMemory] = None
        self.shm_metadata: Optional[shared_memory.SharedMemory] = None

    def create_buffers(
        self,
        obs_size: int = 10 * 1024 * 1024,
        action_size: int = 16 * 1024,
        metadata_size: int = 4096,
    ):
        """Create shared memory buffers (called by server)."""
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

    def connect_buffers(self):
        """Connect to existing shared memory buffers (called by client)."""

        self.shm_obs = shared_memory.SharedMemory(name=f"{self.name_prefix}_obs")
        self.shm_action = shared_memory.SharedMemory(name=f"{self.name_prefix}_action")
        self.shm_metadata = shared_memory.SharedMemory(
            name=f"{self.name_prefix}_metadata"
        )

    def cleanup(self):
        """Clean up shared memory buffers."""
        for shm in [self.shm_obs, self.shm_action, self.shm_metadata]:
            if shm:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

    def close(self):
        """Close shared memory buffers without unlinking."""
        for shm in [self.shm_obs, self.shm_action, self.shm_metadata]:
            if shm:
                try:
                    shm.close()
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

    def _read_metadata(self) -> Dict[str, Any]:
        """Read metadata from shared memory."""
        if not self.shm_metadata:
            raise RuntimeError("Metadata shared memory not initialized")
        size = struct.unpack("!I", bytes(self.shm_metadata.buf[:4]))[0]
        data = bytes(self.shm_metadata.buf[4 : 4 + size])
        return pickle.loads(data)

    def write_observation(self, obs_data: Any):
        """Write observation to shared memory."""
        if not self.shm_obs:
            raise RuntimeError("Observation shared memory not initialized")
        data = pickle.dumps(obs_data)
        size = len(data)
        if size > len(self.shm_obs.buf) - 4:
            raise ValueError(
                f"Observation too large: {size} > {len(self.shm_obs.buf) - 4}"
            )
        self.shm_obs.buf[:4] = struct.pack("!I", size)
        self.shm_obs.buf[4 : 4 + size] = data
        # Update metadata
        metadata = self._read_metadata()
        metadata["obs_size"] = size
        metadata["obs_timestamp"] = time.time()
        self._write_metadata(metadata)

    def read_observation(self) -> Any:
        """Read observation from shared memory."""
        if not self.shm_obs:
            raise RuntimeError("Observation shared memory not initialized")

        # 读取前4个字节获取大小
        # memoryview 也可以直接切片读取，避免 bytes() 拷贝
        size_data = self.shm_obs.buf[:4]
        size = struct.unpack("!I", size_data)[0]

        if size == 0:
            return None

        # === 优化前 (旧代码) ===
        # data = bytes(self.shm_obs.buf[4 : 4 + size]) # <--- 罪魁祸首！这会创建一个全新的 10MB 字节串
        # return pickle.loads(data)

        # === 优化后 (新代码) ===
        # 直接使用 memoryview 切片，这是一个“引用”，不占用额外内存
        # pickle.loads 支持直接从 buffer/memoryview 读取
        data_view = self.shm_obs.buf[4 : 4 + size]
        return pickle.loads(data_view)

    def write_action(self, action_data: Any):
        """Write action to shared memory."""
        if not self.shm_action:
            raise RuntimeError("Action shared memory not initialized")
        data = pickle.dumps(action_data)
        size = len(data)
        if size > len(self.shm_action.buf) - 4:
            raise ValueError(
                f"Action too large: {size} > {len(self.shm_action.buf) - 4}"
            )
        self.shm_action.buf[:4] = struct.pack("!I", size)
        self.shm_action.buf[4 : 4 + size] = data

    def read_action(self) -> Any:
        """Read action from shared memory."""
        if not self.shm_action:
            raise RuntimeError("Action shared memory not initialized")

        size_data = self.shm_action.buf[:4]
        size = struct.unpack("!I", size_data)[0]

        if size == 0:
            return None

        # === 同样优化这里 ===
        data_view = self.shm_action.buf[4 : 4 + size]
        return pickle.loads(data_view)


class IsaacSimulator:
    """Isaac Lab simulator with independent process execution via shared memory + Unix socket."""

    def __init__(
        self,
        sim_config: Optional[Dict[str, Any]] = None,
    ):
        self.sim_config = sim_config or {}
        self.socket: Optional[socket.socket] = None
        self.socket_path: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.is_initialized = False
        self.device = self.sim_config.get("device", "cpu")
        self.num_envs = self.sim_config.get("num_envs", 1)
        self._command_lock = threading.Lock()
        self._sim_lock = threading.Lock()
        self._config_file: Optional[str] = None

        # Shared memory manager
        self.shm_name = f"isaac_sim_{os.getpid()}_{int(time.time())}"
        self.shm_manager = SharedMemoryManager(self.shm_name)

        # For API compatibility
        self.action_space_info: Optional[Dict] = None
        self.observation_space_info: Optional[Dict] = None

    def _send_message(self, message: Dict[str, Any]) -> None:
        """Send a control message via Unix socket."""
        if not self.socket:
            raise ConnectionError("Socket not connected")
        data = pickle.dumps(message)
        length = struct.pack("!I", len(data))
        self.socket.sendall(length + data)

    def _receive_message(self) -> Dict[str, Any]:
        """Receive a control message via Unix socket."""
        if not self.socket:
            raise ConnectionError("Socket not connected")

        # Read 4-byte length prefix
        raw_length = b""
        while len(raw_length) < 4:
            chunk = self.socket.recv(4 - len(raw_length))
            if not chunk:
                raise ConnectionError("Socket connection closed")
            raw_length += chunk

        length = struct.unpack("!I", raw_length)[0]

        # Read the message data
        data = b""
        while len(data) < length:
            chunk = self.socket.recv(min(length - len(data), 4096))
            if not chunk:
                raise ConnectionError("Socket connection closed")
            data += chunk

        return pickle.loads(data)

    def initialize(self) -> bool:
        """Start the Isaac simulation subprocess and initialize it."""
        if self.is_initialized:
            global_console.log(
                "isaacsim",
                "[IsaacSimulator] Simulator already initialized and running.",
            )
            return True

        try:
            # Create Unix socket path in temp directory
            temp_dir = tempfile.gettempdir()
            self.socket_path = os.path.join(temp_dir, f"isaac_sim_{os.getpid()}.sock")

            # Remove socket file if it exists
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

            global_console.log(
                "isaacsim", f"[IsaacSimulator] Using Unix socket: {self.socket_path}"
            )

            # Write config to a temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                if isinstance(self.sim_config, DictConfig):
                    # Convert DictConfig to regular dict
                    self.sim_config = OmegaConf.to_container(
                        self.sim_config, resolve=True
                    )
                json.dump(self.sim_config, f, indent=2)
                self._config_file = f.name

            global_console.log(
                "isaacsim", f"[IsaacSimulator] Config written to {self._config_file}"
            )

            # Get the path to the server script
            server_script = (
                Path(__file__).parent.parent / "launcher" / "isaac_lab_server_shm.py"
            )
            if not server_script.exists():
                raise FileNotFoundError(f"Server script not found: {server_script}")

            # Get the workspace root to use isaaclab.sh
            workspace_root = Path(__file__).parent.parent.parent
            isaaclab_sh = workspace_root / "isaaclab.sh"

            if not isaaclab_sh.exists():
                raise FileNotFoundError(f"isaaclab.sh not found: {isaaclab_sh}")

            # Launch the server process using isaaclab.sh
            launch_command = (
                f"{isaaclab_sh} -p {server_script} "
                f"--socket {self.socket_path} "
                f"--config {self._config_file} "
                f"--shm-name {self.shm_name}"
            )

            global_console.log(
                "isaacsim", f"[IsaacSimulator] Launching server: {launch_command}"
            )
            server_env = os.environ.copy()
            server_env["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(
                launch_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=server_env,  # 关键：传入环境变量强制子进程无缓冲
                preexec_fn=os.setsid,  # 创建独立进程组，便于整体终止
            )

            # Start threads to read and display server output in real-time
            def stream_output(pipe, prefix):
                """Read from pipe and print with prefix."""
                try:
                    for line in iter(pipe.readline, ""):
                        if line:
                            clean_line = line.rstrip()
                            if clean_line:
                                log_subprocess(prefix, clean_line)
                except Exception as e:
                    log_subprocess(prefix, f"Stream error: {e}")

            stdout_thread = threading.Thread(
                target=stream_output,
                args=(self.process.stdout, "SERVER-OUT"),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=stream_output,
                args=(self.process.stderr, "SERVER-ERR"),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            # Give the server a moment to start and create the socket
            global_console.log(
                "isaacsim", "[IsaacSimulator] Waiting for server to initialize..."
            )
            time.sleep(5)  # Increased wait time for Isaac Lab initialization

            # Connect to the server via Unix socket
            global_console.log(
                "isaacsim",
                f"[IsaacSimulator] Connecting to server via {self.socket_path}...",
            )
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

            # Try to connect with retries
            max_retries = 120  # 120 seconds timeout for Isaac Lab startup
            for i in range(max_retries):
                # Check if process is still alive
                if self.process.poll() is not None:
                    # Process has terminated
                    returncode = self.process.returncode
                    raise RuntimeError(
                        f"Server process terminated unexpectedly with code {returncode}. "
                        f"Check the server output above for errors."
                    )

                try:
                    self.socket.connect(self.socket_path)
                    global_console.log(
                        "isaacsim", "[IsaacSimulator] Connected to server successfully!"
                    )
                    break
                except (ConnectionRefusedError, FileNotFoundError):
                    if i == max_retries - 1:
                        raise RuntimeError(
                            f"Failed to connect to server after {max_retries} retries. "
                            f"Socket file: {self.socket_path}. "
                            f"Process status: {'running' if self.process.poll() is None else 'terminated'}"
                        )
                    if i % 10 == 0:  # Print every 10 seconds
                        global_console.log(
                            "isaacsim",
                            f"[IsaacSimulator] Waiting for server socket... ({i + 1}/{max_retries})",
                        )
                    time.sleep(1)

            # Connect to shared memory buffers
            global_console.log(
                "isaacsim", "[IsaacSimulator] Connecting to shared memory buffers..."
            )

            import sys

            _ui_stdout = sys.stdout
            _ui_stderr = sys.stderr

            try:
                # 还原为系统原始 stdout/stderr (有真实 FD)
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

                self.shm_manager.connect_buffers()

            except Exception as e:
                global_console.log(
                    "error", f"[IsaacSimulator] Shared memory connection failed: {e}"
                )
                raise e
            finally:
                # [关键] 恢复 UI 接管
                sys.stdout = _ui_stdout
                sys.stderr = _ui_stderr

            self.shm_manager.connect_buffers()
            global_console.log(
                "isaacsim", "[IsaacSimulator] Connected to shared memory"
            )

            # Wait for initialization signal from server
            self.socket.settimeout(480)  # Long timeout for Isaac Lab init
            response = self._receive_message()

            if response.get("status") == "ready":
                self.is_running = True
                self.is_initialized = True
                self.action_space_info = response.get("action_space")
                self.observation_space_info = response.get("observation_space")
                global_console.log(
                    "isaacsim",
                    "[IsaacSimulator] Simulation subprocess started and environment ready.",
                )
                return True
            else:
                error_msg = response.get("error", "Unknown initialization error")
                global_console.log(
                    "isaacsim",
                    f"[IsaacSimulator] Subprocess initialization failed: {error_msg}",
                )
                self._cleanup_process()
                return False

        except Exception as e:
            global_console.log(
                "isaacsim",
                f"[IsaacSimulator] Failed to start simulation subprocess: {e}",
            )
            traceback.print_exc()
            self._cleanup_process()
            return False

    def _cleanup_process(self):
        """Clean up all resources."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

        if self.socket_path and os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except Exception:
                pass

        if self.process:
            try:
                global_console.log(
                    "isaacsim",
                    f"[IsaacSimulator] Cleaning up subprocess pid={self.process.pid}, poll={self.process.poll()}",
                )
            except Exception:
                pass

            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGTERM)
                self.process.wait(timeout=5)
            except Exception:
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    global_console.log(
                        "isaacsim", "[IsaacSimulator] Failed to kill subprocess pgid"
                    )

            try:
                global_console.log(
                    "isaacsim",
                    f"[IsaacSimulator] Subprocess exit code after cleanup: {self.process.poll()}",
                )
            except Exception:
                pass
            self.process = None

        # Clean up shared memory
        self.shm_manager.close()

        if self._config_file and os.path.exists(self._config_file):
            try:
                os.unlink(self._config_file)
            except Exception:
                pass

        self.is_running = False
        self.is_initialized = False

    def shutdown(self):
        """Stop the Isaac simulation subprocess."""
        global_console.log("isaacsim", "[IsaacSimulator] Initiating shutdown...")
        if not self.is_running and not self.is_initialized:
            global_console.log(
                "isaacsim", "[IsaacSimulator] Already shutdown or not started."
            )
            return

        try:
            if self.process:
                global_console.log(
                    "isaacsim",
                    f"[IsaacSimulator] Shutdown called with pid={self.process.pid}, poll={self.process.poll()}",
                )
            if self.socket:
                self._send_message({"command": "shutdown"})
                # Wait for acknowledgment
                try:
                    self.socket.settimeout(10)
                    ack = self._receive_message()
                    global_console.log(
                        "isaacsim",
                        f"[IsaacSimulator] Shutdown acknowledged by subprocess: {ack}",
                    )
                except socket.timeout:
                    global_console.log(
                        "isaacsim",
                        "[IsaacSimulator] No shutdown acknowledgment from subprocess.",
                    )
                except Exception as e:
                    global_console.log(
                        "isaacsim",
                        f"[IsaacSimulator] Error receiving shutdown ack: {e}",
                    )

            self._cleanup_process()
            global_console.log(
                "isaacsim", "[IsaacSimulator] Simulation shutdown completed."
            )

        except Exception as e:
            global_console.log(
                "isaacsim", f"[IsaacSimulator] Error during simulation shutdown: {e}"
            )
            traceback.print_exc()
            self._cleanup_process()

    # ==================== Environment Operations (Pure) ====================

    @staticmethod
    def _should_retry_command(response: Optional[Dict[str, Any]]) -> bool:
        """Check if a command should be retried based on the response."""
        if isinstance(response, dict) and response.get("success") is False:
            error_message = response.get("error", "").lower()
            if "timeout" in error_message:
                global_console.log(
                    "isaacsim",
                    f"[IsaacSimulator._should_retry_command] Detected retryable error: {error_message}",
                )
                return True
        return False

    @retry(
        max_attempts=3,
        delay_seconds=0.5,
        backoff_factor=1.5,
        max_delay_seconds=5.0,
        exceptions_to_retry=(pickle.UnpicklingError, TimeoutError),
        retry_on_return_value_condition=_should_retry_command,
    )
    def _send_command_and_recv(
        self, command: Dict[str, Any], timeout: float = 60
    ) -> Optional[Dict[str, Any]]:
        """Send a command and receive response with retries."""
        acquired = self._command_lock.acquire(timeout=60)
        if not acquired:
            global_console.log(
                "isaacsim",
                "[IsaacSimulator] Failed to acquire command lock within timeout.",
            )
            return {"success": False, "error": "Command lock acquisition failed"}

        try:
            if not self.is_initialized or not self.socket:
                global_console.log(
                    "isaacsim",
                    "[IsaacSimulator] Simulator not initialized or connection lost.",
                )
                return {"success": False, "error": "Simulator not initialized"}

            self._send_message(command)
            self.socket.settimeout(timeout)
            response = self._receive_message()

            if "error" in response:
                global_console.log(
                    "isaacsim",
                    f"[IsaacSimulator] Error from subprocess for command {command.get('command')}: {response['error']}",
                )
            return response

        except socket.timeout:
            global_console.log(
                "isaacsim",
                f"[IsaacSimulator] Timeout waiting for response to command: {command.get('command')}",
            )
            return {"success": False, "error": "Timeout"}
        except (EOFError, BrokenPipeError, ConnectionError) as e:
            global_console.log(
                "isaacsim",
                f"[IsaacSimulator] Communication pipe broken: {e}. Shutting down simulator.",
            )
            self.shutdown()
            return {"success": False, "error": f"Pipe broken: {e}"}
        except Exception as e:
            global_console.log(
                "isaacsim",
                f"[IsaacSimulator] Error sending/receiving command {command.get('command')}: {e}",
            )
            return {"success": False, "error": str(e)}
        finally:
            self._command_lock.release()

    def set_env_decimation(self, decimation: int) -> bool:
        """Set the environment decimation factor."""
        response = self._send_command_and_recv(
            {"command": "set_env_decimation", "decimation": decimation}
        )
        if response and response.get("success"):
            global_console.log(
                "isaacsim",
                f"[IsaacSimulator] Environment decimation set to {decimation}",
            )
            return True
        return False

    # ==================== Environment Query Operations ====================

    def get_scene_info(self) -> Dict[str, Any]:
        """Get scene state information (object positions, robot state, etc.)."""
        response = self._send_command_and_recv({"command": "get_scene_info"})
        if response and response.get("success"):
            return response.get("scene_info", {})
        return {}

    def get_scene_state(
        self, target_name: str, state_names: List[str]
    ) -> Dict[str, Any]:
        """Get scene state information (object positions, robot state, etc.)."""
        response = self._send_command_and_recv(
            {
                "command": "get_scene_state",
                "target_name": target_name,
                "state_names": state_names,
            }
        )
        if response and response.get("success"):
            failed_state = response.get("failed_state", [])
            # if failed_state:
            #     global_console.log("isaacsim",
            #         f"[IsaacSimulator] Warning: Failed to get states {failed_state} for target {target_name}"
            #     )
            return response.get("state_data", {})
        return {}

    def get_robot_state(self) -> Dict[str, Any]:
        """Get robot state (joint positions, velocities, end effector pose)."""
        response = self._send_command_and_recv({"command": "get_robot_state"})
        if response and response.get("success"):
            return response.get("robot_state", {})
        return {}

    def get_observation(self) -> Optional[List[Observation]]:
        """Requests and retrieves the current observation from the simulator subprocess."""
        # Use shared memory for observation data (high performance)
        response = self._send_command_and_recv({"command": "get_observation"})
        if response and response.get("success"):
            # Read from shared memory - returns a single observation dict
            obs_data = self.shm_manager.read_observation()
            if obs_data:
                # Return as list for compatibility (single observation wrapped in list)
                return [Observation(**obs_data)]
        return None

    def get_current_observation(self) -> Optional[Observation]:
        """Get the current observation from the simulator."""
        response = self._send_command_and_recv({"command": "get_current_observation"})
        if response and response.get("success"):
            # Read from shared memory
            obs_data = self.shm_manager.read_observation()
            if obs_data:
                return Observation(**obs_data)
        return None

    def update(self, return_obs: bool = True) -> Optional[Observation]:
        """this is not a proxy method, it bundles multiple simulator calls into one for efficiency. useful for stepping env without actions."""
        response = self._send_command_and_recv(
            {"command": "step_sim", "return_obs": return_obs}
        )
        if response and response.get("success") and return_obs:
            obs_data = self.shm_manager.read_observation()
            obs = Observation(**obs_data) if obs_data else None
            return obs
        return None

    def step_env(
        self, action: Action
    ) -> Optional[Tuple[Observation, float, bool, bool, Dict[str, Any]]]:
        """Manually step the environment with a given action. For testing/direct control."""
        # Write action to shared memory
        action_dict = {
            "data": action.data.tolist()
            if hasattr(action.data, "tolist")
            else action.data,
            "metadata": action.metadata,
        }
        self.shm_manager.write_action(action_dict)

        # Send command via socket (control only)
        response = self._send_command_and_recv({"command": "step_env"})

        if response and response.get("success"):
            # Read observation from shared memory
            obs_data = self.shm_manager.read_observation()
            obs = Observation(**obs_data) if obs_data else None
            return (
                obs,
                response.get("reward"),
                response.get("terminated"),
                response.get("truncated"),
                response.get("info"),
            )
        return None

    def reset_env(self) -> bool:
        """Reset the simulation environment(s)."""
        response = self._send_command_and_recv({"command": "reset_env"})
        if response and response.get("success"):
            return True
        return False


if __name__ == "__main__":
    """
    Comprehensive API test for IsaacSimulator (Pure Environment Client).
    Tests environment operations only - no skill execution (moved to robot_brain_system).
    """
    import time
    from pathlib import Path
    from robot_brain_system.utils.config_utils import load_config

    print("=" * 80)
    print("IsaacSimulator API Test Suite (Pure Environment Client)")
    print("=" * 80)

    # Load default configuration from robot_brain_system/conf
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    print(f"Loading configuration from: {config_path}")
    test_config = load_config(config_path=config_path)
    print(f"Using task: {test_config.simulator.task}")
    print(f"Device: {test_config.simulator.device}")

    # Convert simulator config to dict for IsaacSimulator
    from omegaconf import OmegaConf
    from typing import cast, Dict

    sim_config_dict = cast(
        Dict[str, Any], OmegaConf.to_container(test_config.simulator, resolve=True)
    )

    test_results = {
        "passed": [],
        "failed": [],
    }

    def test_step(name: str, test_func):
        """Run a test step and record results."""
        print(f"\n[TEST] {name}...")
        try:
            test_func()
            print("  ✅ PASSED")
            test_results["passed"].append(name)
            return True
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            test_results["failed"].append(name)
            return False
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback

            traceback.print_exc()
            test_results["failed"].append(name)
            return False

    simulator = None

    try:
        # Test 1: Initialization
        def test_initialization():
            global simulator
            simulator = IsaacSimulator(sim_config=sim_config_dict)
            assert simulator is not None, "Failed to create simulator instance"
            assert not simulator.is_initialized, (
                "Simulator should not be initialized yet"
            )
            assert not simulator.is_running, "Simulator should not be running yet"

        test_step("1. Simulator instantiation", test_initialization)

        # Test 2: Initialize (start simulator)
        def test_initialize():
            result = simulator.initialize()
            assert result, "Simulator initialization failed"
            assert simulator.is_initialized, "Simulator should be marked as initialized"
            assert simulator.is_running, "Simulator should be marked as running"
            assert simulator.socket is not None, "Socket should be connected"
            assert simulator.shm_manager.shm_obs is not None, (
                "Observation shared memory should be connected"
            )
            assert simulator.action_space_info is not None, (
                "Action space info should be retrieved"
            )
            assert simulator.observation_space_info is not None, (
                "Observation space info should be retrieved"
            )

        if not test_step("2. Initialize simulator", test_initialize):
            print("\n⚠️  Cannot continue tests without successful initialization")
            exit(1)

        # Test 3: Reset environment
        def test_reset_env():
            result = simulator.reset_env()
            assert result, "Environment reset failed"

        test_step("3. Reset environment", test_reset_env)

        # Test 4: Get current observation (via shared memory)
        def test_get_current_observation():
            obs = simulator.get_current_observation()
            assert obs is not None, "Failed to get current observation"
            assert hasattr(obs, "data"), "Observation should have 'data' attribute"
            assert hasattr(obs, "timestamp"), (
                "Observation should have 'timestamp' attribute"
            )
            print(f"     Observation timestamp: {obs.timestamp}")

        test_step(
            "4. Get current observation (shared memory)", test_get_current_observation
        )

        # Test 5: Get observation list
        def test_get_observation():
            time.sleep(0.5)  # Let some observations accumulate
            obss = simulator.get_observation()
            print(f"     Retrieved {len(obss) if obss else 0} observations from queue")

        test_step("5. Get observation list", test_get_observation)

        # Test 6: Get scene info
        def test_get_scene_info():
            scene_info = simulator.get_scene_info()
            assert isinstance(scene_info, dict), "Scene info should be a dictionary"
            print(f"     Scene info keys: {list(scene_info.keys())[:5]}...")

        test_step("6. Get scene info", test_get_scene_info)

        # Test 7: Get robot state
        def test_get_robot_state():
            robot_state = simulator.get_robot_state()
            assert isinstance(robot_state, dict), "Robot state should be a dictionary"
            print(f"     Robot state keys: {list(robot_state.keys())[:5]}...")

        test_step("7. Get robot state", test_get_robot_state)

        # Test 8: Step environment (requires action via shared memory)
        def test_step_env():
            # Create a dummy action based on action space info
            action_dim = simulator.action_space_info.get("shape", [1])[-1]
            import numpy as np

            dummy_action = Action(
                data=np.zeros(action_dim, dtype=np.float32),
                metadata={"info": "test action"},
            )
            result = simulator.step_env(dummy_action)
            assert result is not None, "Step environment returned None"
            obs, reward, terminated, truncated, info = result
            breakpoint()
            assert isinstance(obs, Observation), "Returned observation is not valid"
            assert isinstance(reward, float), "Reward should be a float"
            assert isinstance(terminated, bool), "Terminated flag should be bool"
            assert isinstance(truncated, bool), "Truncated flag should be bool"
            assert isinstance(info, dict), "Info should be a dictionary"
            print(
                f"     Step result - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
            )

        test_step("8. Step environment", test_step_env)

        # Test 9: Double initialization (should be idempotent)
        def test_double_initialization():
            result = simulator.initialize()
            assert result, "Re-initialization should return True (idempotent)"
            assert simulator.is_initialized, "Simulator should still be initialized"

        test_step("9. Re-initialization (idempotent)", test_double_initialization)

    finally:
        # Test 10: Shutdown
        def test_shutdown():
            if simulator:
                simulator.shutdown()
                assert not simulator.is_initialized, (
                    "Simulator should be marked as not initialized"
                )
                assert not simulator.is_running, (
                    "Simulator should be marked as not running"
                )

        print("\n" + "=" * 80)
        test_step("10. Shutdown simulator", test_shutdown)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(
        f"✅ Passed: {len(test_results['passed'])}/{len(test_results['passed']) + len(test_results['failed'])}"
    )
    print(
        f"❌ Failed: {len(test_results['failed'])}/{len(test_results['passed']) + len(test_results['failed'])}"
    )

    if test_results["passed"]:
        print("\nPassed tests:")
        for test in test_results["passed"]:
            print(f"  ✅ {test}")

    if test_results["failed"]:
        print("\nFailed tests:")
        for test in test_results["failed"]:
            print(f"  ❌ {test}")

    print("\n" + "=" * 80)

    if len(test_results["failed"]) == 0:
        print("🎉 ALL TESTS PASSED! Pure environment client API is working.")
        exit(0)
    else:
        print(f"⚠️  {len(test_results['failed'])} test(s) failed.")
        exit(1)
