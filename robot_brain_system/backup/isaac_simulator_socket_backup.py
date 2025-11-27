"""
Isaac Lab simulator interface - Refactored for independent process communication.
Runs Isaac simulation in a completely separate process launched via terminal,
ensuring AppLauncher is imported first with a clean context.
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
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# robot_brain_system imports
from robot_brain_system.core.types import (
    Action,
    Observation,
    SkillStatus,
)
from robot_brain_system.utils.retry_utils import retry


class IsaacSimulator:
    """Isaac Lab simulator with independent process execution via socket communication."""

    def __init__(
        self,
        sim_config: Optional[Dict[str, Any]] = None,
    ):
        self.sim_config = sim_config or {}
        self.socket: Optional[socket.socket] = None
        self.port: Optional[int] = None
        self.terminal_id: Optional[str] = None
        self.is_running = False
        self.is_initialized = False
        self.device = self.sim_config.get("device", "cpu")
        self.num_envs = self.sim_config.get("num_envs", 1)
        self._command_lock = threading.Lock()
        self._config_file: Optional[str] = None

    def _find_free_port(self) -> int:
        """Find a free port to use for socket communication."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _send_message(self, message: Dict[str, Any]) -> None:
        """Send a message with length prefix to avoid TCP packet fragmentation."""
        if not self.socket:
            raise ConnectionError("Socket not connected")
        data = pickle.dumps(message)
        length = struct.pack("!I", len(data))
        self.socket.sendall(length + data)

    def _receive_message(self) -> Dict[str, Any]:
        """Receive a message with length prefix."""
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
        """Start the Isaac simulation in an independent process and connect via socket."""
        if self.is_initialized:
            print("[IsaacSimulator] Simulator already initialized and running.")
            return True

        try:
            # Find a free port
            self.port = self._find_free_port()
            print(f"[IsaacSimulator] Using port {self.port} for communication")

            # Write config to a temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            ) as f:
                json.dump(self.sim_config, f, indent=2)
                self._config_file = f.name
            
            print(f"[IsaacSimulator] Config written to {self._config_file}")

            # Get the path to the server script
            server_script = Path(__file__).parent.parent / "launcher" / "isaac_lab_server.py"
            if not server_script.exists():
                raise FileNotFoundError(f"Server script not found: {server_script}")

            # Get the workspace root to use isaaclab.sh
            workspace_root = Path(__file__).parent.parent.parent
            isaaclab_sh = workspace_root / "isaaclab.sh"
            
            if not isaaclab_sh.exists():
                raise FileNotFoundError(f"isaaclab.sh not found: {isaaclab_sh}")

            # Launch the server process in a terminal using isaaclab.sh
            launch_command = (
                f"{isaaclab_sh} -p {server_script} "
                f"--port {self.port} --config {self._config_file}"
            )
            
            print(f"[IsaacSimulator] Launching server: {launch_command}")
            
            # Use subprocess instead of terminal for better control
            import subprocess
            self.process = subprocess.Popen(
                launch_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            
            # Give the server a moment to start
            time.sleep(2)

            # Connect to the server via socket
            print(f"[IsaacSimulator] Connecting to server on port {self.port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Try to connect with retries
            max_retries = 60  # 60 seconds timeout
            for i in range(max_retries):
                try:
                    self.socket.connect(("127.0.0.1", self.port))
                    print("[IsaacSimulator] Connected to server")
                    break
                except ConnectionRefusedError:
                    if i == max_retries - 1:
                        raise
                    time.sleep(1)
                    print(f"[IsaacSimulator] Waiting for server to start... ({i+1}/{max_retries})")

            # Wait for initialization signal from server
            self.socket.settimeout(480)  # Long timeout for Isaac Lab init
            response = self._receive_message()
            
            if response.get("status") == "ready":
                self.is_running = True
                self.is_initialized = True
                self.action_space_info = response.get("action_space")
                self.observation_space_info = response.get("observation_space")
                print("[IsaacSimulator] Server initialized and ready")
                return True
            else:
                error_msg = response.get("error", "Unknown initialization error")
                print(f"[IsaacSimulator] Server initialization failed: {error_msg}")
                self._cleanup()
                return False

        except Exception as e:
            print(f"[IsaacSimulator] Failed to initialize: {e}")
            traceback.print_exc()
            self._cleanup()
            return False

    def _cleanup(self):
        """Clean up resources."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
        
        if hasattr(self, 'process') and self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        
        if self._config_file and os.path.exists(self._config_file):
            try:
                os.unlink(self._config_file)
            except Exception:
                pass
        
        self.is_running = False
        self.is_initialized = False

    def shutdown(self):
        """Stop the Isaac simulation server."""
        print("[IsaacSimulator] Initiating shutdown...")
        if not self.is_running and not self.is_initialized:
            print("[IsaacSimulator] Already shutdown or not started.")
            return

        try:
            if self.socket:
                self._send_message({"command": "shutdown"})
                # Wait for acknowledgment
                try:
                    self.socket.settimeout(10)
                    ack = self._receive_message()
                    print(f"[IsaacSimulator] Shutdown acknowledged: {ack}")
                except socket.timeout:
                    print("[IsaacSimulator] No shutdown acknowledgment (timeout)")
                except Exception as e:
                    print(f"[IsaacSimulator] Error receiving shutdown ack: {e}")

            self._cleanup()
            print("[IsaacSimulator] Shutdown completed")

        except Exception as e:
            print(f"[IsaacSimulator] Error during shutdown: {e}")
            traceback.print_exc()
            self._cleanup()

    def get_skill_registry_from_sim(self):
        """Get the skill registry from the server."""
        response = self._send_command_and_recv({"command": "get_skill_registry"})
        if response and response.get("success"):
            print(
                f"[IsaacSimulator] Retrieved skill registry with "
                f"{len(response.get('skill_registry').list_skills())} skills."
            )
            return response.get("skill_registry")
        return None

    @staticmethod
    def _should_retry_command(response: Optional[Dict[str, Any]]) -> bool:
        """Check if a command should be retried based on the response."""
        if isinstance(response, dict) and response.get("success") is False:
            error_message = response.get("error", "").lower()
            if "timeout" in error_message:
                print(
                    f"[IsaacSimulator._should_retry_command] "
                    f"Detected retryable error: {error_message}"
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
            print("[IsaacSimulator] Failed to acquire command lock")
            return {"success": False, "error": "Command lock acquisition failed"}

        try:
            if not self.is_initialized or not self.socket:
                print("[IsaacSimulator] Simulator not initialized or connection lost")
                return {"success": False, "error": "Simulator not initialized"}

            self._send_message(command)
            self.socket.settimeout(timeout)
            response = self._receive_message()
            
            if "error" in response:
                print(
                    f"[IsaacSimulator] Error from server for command "
                    f"{command.get('command')}: {response['error']}"
                )
            return response

        except socket.timeout:
            print(
                f"[IsaacSimulator] Timeout waiting for response to command: "
                f"{command.get('command')}"
            )
            return {"success": False, "error": "Timeout"}
        except (EOFError, BrokenPipeError, ConnectionError) as e:
            print(f"[IsaacSimulator] Connection error: {e}. Shutting down simulator.")
            self.shutdown()
            return {"success": False, "error": f"Connection error: {e}"}
        except Exception as e:
            print(
                f"[IsaacSimulator] Error sending/receiving command "
                f"{command.get('command')}: {e}"
            )
            return {"success": False, "error": str(e)}
        finally:
            self._command_lock.release()

    def execute_skill_blocking(
        self, skill_name: str, parameters: Dict[str, Any]
    ) -> bool:
        """Execute a skill in the simulation (blocks until skill completion)."""
        response = self._send_command_and_recv(
            {
                "command": "execute_skill_blocking",
                "skill_name": skill_name,
                "parameters": parameters,
            },
            timeout=parameters.get("timeout", 60.0) + 5.0,
        )
        return response.get("success", False) if response else False

    def start_skill_non_blocking(
        self, skill_name: str, parameters: Dict[str, Any], timeout: float = 60.0
    ) -> bool:
        """Start a skill execution (non-blocking)."""
        response = self._send_command_and_recv(
            {
                "command": "start_skill_non_blocking",
                "skill_name": skill_name,
                "parameters": parameters,
            },
            timeout=timeout,
        )
        return response.get("success", False) if response else False

    def get_skill_executor_status(self) -> Dict[str, Any]:
        """Get current skill executor status from the server."""
        response = self._send_command_and_recv({"command": "get_skill_executor_status"})
        return response or {"status": "error", "error": "No response"}

    def terminate_current_skill(
        self, skill_status: SkillStatus = SkillStatus.INTERRUPTED, status_info: str = ""
    ) -> bool:
        """Terminate current skill execution."""
        response = self._send_command_and_recv(
            {
                "command": "terminate_current_skill",
                "skill_status": skill_status,
                "status_info": status_info,
            }
        )
        return response.get("success", False) if response else False

    def pause_current_skill(self) -> bool:
        """Pause current skill execution."""
        response = self._send_command_and_recv(
            {"command": "change_current_skill_status", "status": SkillStatus.PAUSED}
        )
        return response.get("success", False) if response else False

    def recovery_current_skill(self) -> bool:
        """Resume current skill execution."""
        response = self._send_command_and_recv(
            {"command": "change_current_skill_status", "status": SkillStatus.RUNNING}
        )
        return response.get("success", False) if response else False

    def change_current_skill_status(self, status: SkillStatus) -> bool:
        """Change current skill execution status."""
        response = self._send_command_and_recv(
            {"command": "change_current_skill_status_force", "status": status}
        )
        return response.get("success", False) if response else False

    def get_observation(self) -> Optional[list[Observation]]:
        """Get observations from the simulator."""
        response = self._send_command_and_recv({"command": "get_observation"})
        obss = []
        if response and response.get("success"):
            obs_data = response.get("observation_data")
            for obs in obs_data:
                obss.append(Observation(**obs))
            return obss
        return None

    def get_current_observation(self) -> Optional[Observation]:
        """Get the current observation."""
        response = self._send_command_and_recv({"command": "get_current_observation"})
        if response and response.get("success"):
            obs_data = response.get("observation_data")
            return Observation(**obs_data)
        return None

    def step_env(
        self, action: Action
    ) -> Optional[Tuple[Observation, float, bool, bool, Dict[str, Any]]]:
        """Manually step the environment with a given action."""
        response = self._send_command_and_recv(
            {
                "command": "step_env",
                "action_data": action.data.tolist()
                if hasattr(action.data, "tolist")
                else action.data,
                "action_metadata": action.metadata,
            }
        )
        if response and response.get("success"):
            obs = (
                Observation(**response["observation_data"])
                if response.get("observation_data")
                else None
            )
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

    def cleanup_skill(self) -> bool:
        """Clean up the current skill."""
        response = self._send_command_and_recv({"command": "cleanup_skill"})
        if response and response.get("success"):
            return True
        return False
