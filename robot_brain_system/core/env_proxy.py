"""
Environment Proxy for Remote Skill Execution.
This proxy allows skills to access remote environment through IsaacSimulator,
replacing direct env.unwrapped access with RPC calls.
"""

import math
from typing import Any, Dict, List, Optional
from collections import deque
import threading
import time
import torch
import numpy as np
from copy import deepcopy
from PIL import Image

import isaaclab.utils.math as math_utils


class DynamicProxyMixin:
    """
    Mixin class to add dynamic fallback for undefined methods and properties.

    - For ObjectProxy instances: Undefined methods fallback to self.remote_execute(func=name, **kwargs).
    - For ObjectDataProxy instances: Undefined properties fallback to self._simulator.get_scene_state(self._object_name, [name]),
      with the result tensorized and cached if possible.

    This mixin uses __getattr__ to intercept accesses to undefined attributes.
    It distinguishes between the two classes based on the instance type.
    """

    def __getattr__(self, name):
        """Intercept access to undefined attributes and provide fallback based on class type."""
        if name.startswith("__") and name.endswith("__"):
            # Don't interfere with special methods
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

        if isinstance(self, ObjectProxy):
            # For ObjectProxy: Assume it's a method, return a wrapper that calls remote_execute
            def method_wrapper(*args, **kwargs):
                # Convert args/kwargs to a params dict if needed; here we assume kwargs are params
                # If args are present, we can pack them into 'args' key or handle accordingly
                params = {"args": args, **kwargs} if args else kwargs
                return self.remote_execute(func=name, **params)

            return method_wrapper

        elif isinstance(self, ObjectDataProxy):
            # For ObjectDataProxy: Assume it's a property, fetch from get_scene_state
            state_data = self._simulator.get_scene_state(self._object_name, [name])
            value = state_data.get(name)
            if value is None:
                raise AttributeError(
                    f"No data found for '{name}' in scene state for {self._object_name}"
                )

            # Tensorize the value (assuming it's list-like, as in existing code)
            tensor_value = torch.tensor(value, device=self._env_proxy.device)

            # Cache it for future access
            self._cache[name] = tensor_value
            return tensor_value

        else:
            raise TypeError(
                f"{type(self).__name__} is not supported by DynamicProxyMixin"
            )


class EnvProxy:
    """
    Proxy for remote environment access.
    Skills use this instead of direct env.unwrapped access.

    Key Features:
    - Intercepts env.unwrapped attribute access
    - Forwards environment queries to remote server via IsaacSimulator
    - Maintains local cache for scene structure
    - Provides compatible interface with Isaac Lab's RLEnv
    - Buffers observation history for monitoring (thread-safe)
    """

    def __init__(
        self,
        isaac_simulator,
        buffer_size: int = 500,
        store_images: bool = True,
        scene_mode: str = "default",
    ):
        """
        Initialize environment proxy.

        Args:
            isaac_simulator: IsaacSimulator instance (client to remote env server)
            buffer_size: Maximum number of observations to buffer (default: 500 frames = 10s @ 50Hz)
            store_images: Whether to store image data in buffer (default: True)
            scene_mode: Scene mode for asset initialization (default: "default")
        """
        self._simulator = isaac_simulator
        self._device = torch.device(isaac_simulator.device)
        self._num_envs = isaac_simulator.num_envs
        self._env_lock = threading.Lock()

        # Cache for scene structure
        self._scene_cache: Dict[str, Any] = {}
        self._scene_cache_valid = False

        # Fake 'unwrapped' to make skills think they have direct env access
        self.unwrapped = self

        # Create proxies for complex attributes
        self._action_manager_proxy = None
        self._scene_proxy_enhanced = None

        # Observation history buffer (thread-safe)
        self._obs_buffer = deque(maxlen=buffer_size)
        self._obs_buffer_lock = threading.Lock()
        self._last_buffer_clear_time = time.time()
        self._store_images = store_images

        # 新增：场景模式与基准位姿缓存
        self.scene_mode = scene_mode
        self._box_init_pose = None  # tuple(pos, quat)
        self._spanner_init_pose = None  # tuple(pos, quat)
        self._heavy_init_pose = None  # tuple(pos, quat)
        self._box_far_pose = None
        self._spanner_far_pose = None
        self._roi_default = {
            "x": (1.05, 1.36),
            "y": (-3.53, -3.35),
            "z": (2.6, 3.2),
        }

        self._ensure_init_poses()

    def _ensure_init_poses(self):
        """Capture canonical poses for box/spanner/heavy_box once using default_root_state."""
        scene = self.scene
        env_ids = torch.tensor([0], device=self.device)

        box = scene["box"]
        spanner = scene["spanner"]
        heavy_box = scene["heavy_box"] if "heavy_box" in scene.keys() else None

        if self._box_init_pose is None:
            self._box_init_pose = (
                box.data.root_pos_w[env_ids].clone()
                + torch.tensor([[0.0, 0.1, 0.0]], device=self.device),
                box.data.root_quat_w[env_ids].clone(),
            )

        if self._spanner_init_pose is None:
            # Place spanner relative to box so that it sits inside the lid as skills expect
            rel_pos = torch.tensor([0.0, -0.04, 0.001], device=self.device)
            rel_rot = math_utils.quat_from_euler_xyz(
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
                torch.tensor([math.pi / 2], device=self.device),
            )
            box_pos, box_quat = self._box_init_pose
            spanner_pos = box_pos - rel_pos.unsqueeze(0)
            spanner_quat = math_utils.quat_mul(box_quat, rel_rot)
            self._spanner_init_pose = (spanner_pos, spanner_quat)

        # if self._heavy_init_pose is None and heavy_box is not None:
        #     self._heavy_init_pose = (
        #         heavy_box.data.root_pos_w[env_ids].clone(),
        #         heavy_box.data.root_quat_w[env_ids].clone(),
        #     )

        if self._box_far_pose is None:
            offset = torch.tensor([[0.0, 2.0, 0.0]], device=self.device)
            self._box_far_pose = (
                self._box_init_pose[0] + offset,
                self._box_init_pose[1],
            )
        # if self._spanner_far_pose is None:
        #     offset = torch.tensor([[0.0, 2.0, 0.0]], device=self.device)
        #     self._spanner_far_pose = (
        #         self._spanner_init_pose[0] + offset,
        #         self._spanner_init_pose[1],
        #     )

    def reset_spanner_position(self):
        """Reset spanner to initial position inside box."""
        scene = self.scene
        box = scene["heavy_box"]
        spanner = scene["spanner"]
        env_ids = torch.tensor([0], device=self.device)
        box_pos = box.data.root_pos_w[env_ids]
        box_quat = box.data.root_quat_w[env_ids]

        # 设置扳手的位置
        spanner_rel_pos = torch.tensor(
            [0, -0.04, 0.001], device=self.device
        )  # asset2 相对于 asset1 的位置偏移
        spanner_rel_rot = math_utils.quat_from_euler_xyz(  # asset2 相对于 asset1 的旋转 (绕 x 轴 90 度)
            torch.tensor([0.0], device=self.device),
            torch.tensor([0.0], device=self.device),
            torch.tensor([math.pi / 2], device=self.device),
        )
        spanner_pos = box_pos[:, :3] - spanner_rel_pos.unsqueeze(0)
        # 旋转 = asset1 的旋转 * 相对旋转
        spanner_orient = math_utils.quat_mul(box_quat, spanner_rel_rot)
        spanner.write_root_pose_to_sim(
            torch.cat([spanner_pos, spanner_orient], dim=-1),
            env_ids=torch.tensor([0], device=self.device),
        )
        spanner.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=self.device),
            env_ids=torch.tensor([0], device=self.device),
        )

        # 步进环境，确保状态刷新
        for _ in range(5):
            self.update(return_obs=True)

        self._scene_cache_valid = False
        return True

    def reset_box_and_spanner(
        self, mode: str = "normal", snapshot_path: str | None = None
    ):
        """Reset box and spanner placement.

        mode:
            - "normal": place box at canonical pose, spanner inside box; heavy box back to its default (far) pose.
            - "far": move box+spanner far away to emulate missing box scenario; heavy box remains at its default.
        """

        scene = self.scene
        box = scene["box"]
        spanner = scene["spanner"]
        heavy_box = scene["heavy_box"] if "heavy_box" in scene.keys() else None
        env_ids = torch.tensor([0], device=self.device)

        if mode == "normal":
            # Only reset box/spanner poses (do NOT full reset_env to avoid disturbing other state)
            box.write_root_pose_to_sim(torch.cat(self._box_init_pose, dim=-1), env_ids)
            box.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=self.device), env_ids
            )

            # if heavy_box is not None and self._heavy_init_pose is not None:
            #     heavy_box.write_root_pose_to_sim(
            #         torch.cat(self._heavy_init_pose, dim=-1), env_ids
            #     )
            #     heavy_box.write_root_velocity_to_sim(
            #         torch.zeros(1, 6, device=self.device), env_ids
            #     )

            # spanner.write_root_pose_to_sim(
            #     torch.cat(self._spanner_init_pose, dim=-1), env_ids
            # )
            # spanner.write_root_velocity_to_sim(
            #     torch.zeros(1, 6, device=self.device), env_ids
            # )

        elif mode == "far":
            box.write_root_pose_to_sim(torch.cat(self._box_far_pose, dim=-1), env_ids)
            box.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=self.device), env_ids
            )

            # if heavy_box is not None and self._heavy_init_pose is not None:
            #     heavy_box.write_root_pose_to_sim(
            #         torch.cat(self._heavy_init_pose, dim=-1), env_ids
            #     )
            #     heavy_box.write_root_velocity_to_sim(
            #         torch.zeros(1, 6, device=self.device), env_ids
            #     )

            # spanner.write_root_pose_to_sim(
            #     torch.cat(self._spanner_far_pose, dim=-1), env_ids
            # )
            # spanner.write_root_velocity_to_sim(
            #     torch.zeros(1, 6, device=self.device), env_ids
            # )
        else:
            raise ValueError(f"Unknown reset mode: {mode}")

        # 步进环境，确保状态刷新
        for _ in range(5):
            self.update(return_obs=True)

        self._scene_cache_valid = False

        # Snapshot after placement (normal mode) if requested
        if mode == "normal" and snapshot_path:
            obs = self._simulator.get_current_observation()
            if obs is not None:
                try:
                    inspector_rgb = (
                        (
                            obs.data.get("policy", {}).get(
                                "inspector_side", torch.empty(1)
                            )
                        )[0]
                        .cpu()
                        .numpy()
                    )
                    Image.fromarray(inspector_rgb).save(snapshot_path)
                except Exception:
                    pass

        return True

    def check_box_presence(self, roi: dict | None = None) -> dict:
        """Check whether the light box is inside the expected ROI.

        Returns dict: {"ok": bool, "reason": str}
        """

        roi = roi or self._roi_default
        try:
            box_pos = self.scene["box"].data.root_pos_w[0]  # (3,)
            env_origin = (
                self.scene.env_origins[0]
                if hasattr(self.scene, "env_origins")
                else torch.zeros_like(box_pos)
            )
            box_pos_rel = box_pos - env_origin
            in_roi = (
                roi["x"][0] <= box_pos_rel[0] <= roi["x"][1]
                and roi["y"][0] <= box_pos_rel[1] <= roi["y"][1]
                and roi["z"][0] <= box_pos_rel[2] <= roi["z"][1]
            )
            if in_roi:
                return {"ok": True, "reason": "box within ROI"}
            return {
                "ok": False,
                "reason": "box out of ROI: (%.2f, %.2f, %.2f)"
                % (
                    float(box_pos_rel[0].item()),
                    float(box_pos_rel[1].item()),
                    float(box_pos_rel[2].item()),
                ),
            }
        except Exception as e:
            return {"ok": False, "reason": f"roi check failed: {e}"}

        print(
            f"[EnvProxy] Initialized with buffer_size={buffer_size}, store_images={store_images}"
        )

    @property
    def device(self):
        """Get device (cpu/cuda)."""
        return self._device

    @property
    def num_envs(self):
        """Get number of parallel environments."""
        return self._num_envs

    @property
    def scene(self):
        """Get scene proxy for accessing scene objects."""
        if not self._scene_cache_valid:
            self._refresh_scene_cache()

        # Create enhanced scene proxy with env_origins
        if self._scene_proxy_enhanced is None:
            base_scene = SceneProxy(self)
            self._scene_proxy_enhanced = base_scene
            # Add env_origins attribute
            origins_response = self._simulator._send_command_and_recv(
                {"command": "get_env_origins"}
            )
            if origins_response and origins_response.get("success"):
                self._scene_proxy_enhanced.env_origins = torch.tensor(
                    origins_response.get("env_origins"), device=self._device
                )

        return self._scene_proxy_enhanced

    @property
    def action_manager(self):
        """Get action manager proxy."""
        if self._action_manager_proxy is None:
            self._action_manager_proxy = ActionManagerProxy(self)
        return self._action_manager_proxy

    def acquire_env_lock(self):
        """Acquire simulator lock for thread-safe operations."""
        self._env_lock.acquire()

    def release_env_lock(self):
        """Release simulator lock after thread-safe operations."""
        try:
            self._env_lock.release()
        # haven't acquired yet
        except RuntimeError:
            pass

    def __enter__(self):
        """Enter context manager for env lock."""
        self.acquire_env_lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager for env lock."""
        self.release_env_lock()

    def _refresh_scene_cache(self):
        """Refresh scene structure cache from remote server."""
        scene_info = self._simulator.get_scene_info()
        self._scene_cache = scene_info
        self._scene_cache_valid = True

    def update(self, return_obs: bool = True):
        """this is not a proxy method, it bundles multiple simulator calls into one for efficiency. useful for stepping env without actions."""
        return self._simulator.update(return_obs=return_obs)

    def step(self, action):
        """Execute one or more environment steps via the remote server."""
        from robot_brain_system.core.types import Action as ActionType

        empty_result = (None, None, None, None, {})

        if isinstance(action, torch.Tensor):
            action = ActionType(data=action, metadata={})

        metadata = action.metadata or {}
        raw_data = action.data

        action_tensor = self._coerce_action_tensor(raw_data)

        if action_tensor is None or action_tensor.numel() == 0:
            return empty_result

        # Ensure CPU tensor for serialization
        action_tensor = action_tensor.detach().cpu()

        # Normalize shape to (num_envs, chunk, action_dim)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        elif action_tensor.ndim != 2:
            raise ValueError(
                f"Unsupported action shape {action_tensor.shape}, expect 1D/2D tensor"
            )

        sanitized_metadata = self._sanitize_metadata(metadata)
        last_result = empty_result

        single_action = action_tensor
        action_payload = ActionType(
            data=single_action.numpy().tolist(),
            metadata=sanitized_metadata,
        )

        last_result = self._simulator.step_env(action_payload)
        if last_result is None:
            return empty_result

        return last_result

    def _coerce_action_tensor(self, raw_data: Any) -> Optional[torch.Tensor]:
        """Convert the incoming action payload into a torch tensor for chunk handling."""
        if raw_data is None:
            return None

        if isinstance(raw_data, torch.Tensor):
            return raw_data.detach()
        if isinstance(raw_data, np.ndarray):
            return torch.from_numpy(raw_data)
        if isinstance(raw_data, list):
            if len(raw_data) == 0:
                return torch.empty(0)
            return torch.tensor(raw_data)
        if isinstance(raw_data, tuple):
            if len(raw_data) == 0:
                return torch.empty(0)
            return torch.tensor(list(raw_data))

        # Fallback for scalar values
        return torch.tensor(raw_data)

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Strip non-serializable metadata fields before sending across processes."""
        if not isinstance(metadata, dict):
            return {}

        sanitized: Dict[str, Any] = {}
        for key in ("info", "reason", "skill_name"):
            if key not in metadata:
                continue
            value = metadata[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            else:
                sanitized[key] = str(value)

        return sanitized

    def reset(self, **kwargs):
        """Reset environment via remote server."""
        success = self._simulator.reset_env()
        if success:
            # Invalidate cache on reset
            self._scene_cache_valid = False
            self.switch_heavybox = None  # reset heavy box switch state
            # 如果是缺箱子测试场景，重置后直接把箱子移走
            if self.scene_mode == "missing_box_human_intervention_test":
                self.reset_box_and_spanner("far")

            obs = self._simulator.get_current_observation()
            return (obs.data if obs else None), {}
        return None, {}

    def get_latest_observation(self):
        """
        Get the latest observation from SERVER and automatically buffer it.

        This is the primary method for getting observations. It:
        1. Fetches the latest observation from the remote server
        2. Automatically adds it to the history buffer (with timestamp)
        3. Returns the observation for immediate use

        Used by:
        - SkillExecutorClient (50Hz execution loop)
        - System.execute_task (initial planning)
        - System._handle_skill_completion (final summary/replanning)

        Returns:
            Observation or None: the latest Observation instance from server
        """
        obs = self._simulator.get_current_observation()

        def move_to_cpu(data):
            if isinstance(data, torch.Tensor):
                return data.detach().cpu()
            elif isinstance(data, dict):
                return {k: move_to_cpu(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [move_to_cpu(v) for v in data]
            return data

        if obs is not None:
            # Automatically buffer observation (thread-safe)
            with self._obs_buffer_lock:
                # 注意：不要修改返回给 Skill 的原始 obs，因为 Skill 可能需要 GPU 张量进行计算
                obs_for_buffer = deepcopy(obs)
                if hasattr(obs_for_buffer, "data"):
                    obs_for_buffer.data = move_to_cpu(obs_for_buffer.data)

                buffer_entry = {"time": time.time(), "obs": obs}
                # Optionally remove image data to save memory
                if (
                    not self._store_images
                    and hasattr(obs, "data")
                    and isinstance(obs.data, dict)
                ):
                    # Create a shallow copy and remove large image keys
                    filtered_data = {
                        k: v
                        for k, v in obs.data.items()
                        if not (
                            k.startswith("rgb")
                            or k.startswith("depth")
                            or k.startswith("image")
                        )
                    }
                    # Note: This doesn't modify the returned obs, only the buffered copy
                    buffer_entry["obs_data_filtered"] = filtered_data

                self._obs_buffer.append(buffer_entry)

        return obs

    def get_and_clear_observation_buffer(self) -> List:
        """
        Get all buffered observations and clear the buffer.

        Used exclusively by System._main_loop for Brain monitoring.
        Returns observations collected since last clear (typically ~100 frames @ 50Hz over 2 seconds).

        Returns:
            List[Observation]: all buffered observations in chronological order
        """
        with self._obs_buffer_lock:
            buffer_copy = [item["obs"] for item in self._obs_buffer]
            self._obs_buffer.clear()
            self._last_buffer_clear_time = time.time()

        return buffer_copy

    def clear_observation_buffer(self):
        """
        Clear the observation buffer.

        Called by System._start_next_skill before starting each new skill
        to ensure buffer only contains observations from the current skill.
        """
        with self._obs_buffer_lock:
            self._obs_buffer.clear()
            self._last_buffer_clear_time = time.time()
        # print("[EnvProxy] Observation buffer cleared")

    def peek_observation_buffer(self, n: Optional[int] = None) -> List:
        """
        View buffered observations without clearing (for debugging).

        Args:
            n: Number of most recent observations to return (None = all)

        Returns:
            List[Observation]: requested observations (does not modify buffer)
        """
        with self._obs_buffer_lock:
            if n is None:
                return [item["obs"] for item in self._obs_buffer]
            else:
                return [item["obs"] for item in list(self._obs_buffer)[-n:]]

    def get_buffer_info(self) -> Dict[str, Any]:
        """
        Get information about the observation buffer (for debugging).

        Returns:
            Dict with buffer size, age, etc.
        """
        with self._obs_buffer_lock:
            return {
                "buffer_size": len(self._obs_buffer),
                "buffer_maxlen": self._obs_buffer.maxlen,
                "seconds_since_clear": time.time() - self._last_buffer_clear_time,
                "store_images": self._store_images,
            }

    # === Backward compatibility aliases ===

    def get_current_observation(self):
        """Backward compatibility alias for get_latest_observation()."""
        return self.get_latest_observation()

    def get_observation(self):
        """Get the list of observations (compatibility with get_observation on simulator).

        Returns:
            Optional[List[Observation]]: list of Observation objects or None.
        """
        return self._simulator.get_observation()

    def get_observations(self):
        """Compatibility alias returning the raw obs.data dict (older callers expect dict/data).

        This preserves the previous behaviour while newer code should call
        `get_latest_observation()`.
        """
        obs = self.get_latest_observation()
        return obs.data if obs else {}


class SceneProxy:
    """
    Proxy for scene object access.
    Intercepts scene["object_name"] and provides object proxies.
    """

    def __init__(self, env_proxy: EnvProxy):
        self._env_proxy = env_proxy
        self._object_cache: Dict[str, "ObjectProxy"] = {}

    def __getitem__(self, key: str):
        """Get object proxy by name."""
        if key not in self._object_cache:
            self._object_cache[key] = ObjectProxy(self._env_proxy, key)
        return self._object_cache[key]

    def __contains__(self, key: str):
        """Check if object exists in scene."""
        return key in self._env_proxy._scene_cache

    def keys(self):
        """Get all object names in scene."""
        return self._env_proxy._scene_cache.keys()

    @property
    def articulations(self):
        """Get articulations dict (for compatibility)."""
        return {k: self[k] for k in self.keys() if "robot" in k.lower()}


class ObjectProxy(DynamicProxyMixin):
    """
    Proxy for individual scene objects.
    Provides .data attribute and methods that forward to remote server.
    """

    def __init__(self, env_proxy: EnvProxy, object_name: str):
        self._env_proxy = env_proxy
        self._object_name = object_name
        self._simulator = env_proxy._simulator
        self._data_proxy = None

    @property
    def data(self):
        """Get object data proxy (provides access to object state)."""
        if self._data_proxy is None:
            self._data_proxy = ObjectDataProxy(self._env_proxy, self._object_name)
        return self._data_proxy

    @property
    def joint_names(self):
        """Get robot joint names (for robot objects)."""
        response = self._simulator._send_command_and_recv(
            {
                "command": "get_object_attribute",
                "name": self._object_name,
                "attribute": "joint_names",
            }
        )
        if response and response.get("success"):
            return response.get("attribute_value", [])
        else:
            print(
                f"[ObjectProxy] Failed to get joint names for {self._object_name}: {response}"
            )
            return []

    def write_root_pose_to_sim(self, pose: torch.Tensor, env_ids: torch.Tensor):
        """
        Write object root pose to simulation.

        Args:
            pose: Tensor of shape (N, 7) with [x, y, z, qw, qx, qy, qz]
            env_ids: Environment IDs to apply to
        """
        pose_list = pose.cpu().numpy().tolist()
        env_ids_list = env_ids.cpu().numpy().tolist()

        for env_id, single_pose in zip(env_ids_list, pose_list):
            position = single_pose[:3]
            quaternion = single_pose[3:7]
            response = self._simulator._send_command_and_recv(
                {
                    "command": "set_object_pose",
                    "name": self._object_name,
                    "env_id": env_id,
                    "position": position,
                    "quaternion": quaternion,
                }
            )
            if not response or not response.get("success"):
                print(
                    f"[ObjectProxy] Failed to set pose for {self._object_name}: {response}"
                )

    def write_root_velocity_to_sim(self, velocity: torch.Tensor, env_ids: torch.Tensor):
        """
        Write object root velocity to simulation.

        Args:
            velocity: Tensor of shape (N, 6) with [vx, vy, vz, wx, wy, wz]
            env_ids: Environment IDs to apply to
        """
        velocity_list = velocity.cpu().numpy().tolist()
        env_ids_list = env_ids.cpu().numpy().tolist()

        for env_id, single_vel in zip(env_ids_list, velocity_list):
            linear_velocity = single_vel[:3]
            angular_velocity = single_vel[3:6]
            response = self._simulator._send_command_and_recv(
                {
                    "command": "set_object_velocity",
                    "name": self._object_name,
                    "env_id": env_id,
                    "linear_velocity": linear_velocity,
                    "angular_velocity": angular_velocity,
                }
            )
            if not response or not response.get("success"):
                print(
                    f"[ObjectProxy] Failed to set velocity for {self._object_name}: {response}"
                )

    def write_root_state_to_sim(
        self, root_state: torch.Tensor, env_ids: Optional[torch.Tensor] = None
    ):
        """Write combined root pose and velocity to simulation.

        Args:
            root_state: Tensor of shape (N, 13) = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            env_ids: Optional tensor of environment IDs (defaults to zeros)
        """

        if root_state is None:
            raise ValueError("root_state tensor must be provided")

        if not isinstance(root_state, torch.Tensor):
            root_state = torch.tensor(root_state, device=self._env_proxy.device)

        if root_state.ndim == 1:
            root_state = root_state.unsqueeze(0)

        root_state = root_state.to(device=self._env_proxy.device, dtype=torch.float32)

        num_envs = root_state.shape[0]
        if root_state.shape[-1] < 13:
            raise ValueError(
                f"root_state must have at least 13 values (got {root_state.shape[-1]})"
            )

        if env_ids is None:
            env_ids_tensor = torch.zeros(
                num_envs, device=self._env_proxy.device, dtype=torch.long
            )
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self._env_proxy.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(
                env_ids, device=self._env_proxy.device, dtype=torch.long
            )

        pose = root_state[:, :7]
        velocity = root_state[:, 7:13]

        self.write_root_pose_to_sim(pose, env_ids_tensor)
        self.write_root_velocity_to_sim(velocity, env_ids_tensor)

    def set_joint_position_target(self, target: torch.Tensor):
        """Set robot joint position target (for robot objects)."""
        target_list = target.cpu().numpy().tolist()
        response = self._simulator._send_command_and_recv(
            {
                "command": "set_robot_joint_targets",
                "robot_name": self._object_name,
                "pos_target": target_list,
                "vel_target": None,
            }
        )
        if not response or not response.get("success"):
            print(f"[ObjectProxy] Failed to set joint position target: {response}")

    def set_joint_velocity_target(self, target: torch.Tensor):
        """Set robot joint velocity target (for robot objects)."""
        target_list = target.cpu().numpy().tolist()
        response = self._simulator._send_command_and_recv(
            {
                "command": "set_robot_joint_targets",
                "robot_name": self._object_name,
                "pos_target": None,
                "vel_target": target_list,
            }
        )
        if not response or not response.get("success"):
            print(f"[ObjectProxy] Failed to set joint velocity target: {response}")

    def write_joint_state_to_sim(self, position: torch.Tensor, velocity: torch.Tensor):
        """Write robot joint state directly to simulation (for robot objects)."""
        pos_list = position.cpu().numpy().tolist()
        vel_list = velocity.cpu().numpy().tolist()
        response = self._simulator._send_command_and_recv(
            {
                "command": "write_robot_joint_state",
                "robot_name": self._object_name,
                "joint_pos": pos_list,
                "joint_vel": vel_list,
            }
        )
        if not response or not response.get("success"):
            print(f"[ObjectProxy] Failed to write joint state: {response}")

    def remote_execute(self, func, **kwargs) -> Any:
        """
        Execute a custom command on the remote server for this object.

        Args:
            command: Command dictionary to send
        Returns:
            Response from remote server
        """
        response = self._simulator._send_command_and_recv(
            {
                "command": "custom_object_command",
                "name": self._object_name,
                "func": func,
                "params": kwargs,
            }
        )
        if not response or not response.get("success"):
            print(
                f"[ObjectProxy] Failed to execute remote command for {self._object_name}: {response}"
            )
        return response


class ObjectDataProxy(DynamicProxyMixin):
    """
    Proxy for object data attributes.
    Queries remote server for object state when attributes are accessed.
    """

    def __init__(self, env_proxy: EnvProxy, object_name: str):
        self._env_proxy = env_proxy
        self._object_name = object_name
        self._simulator = env_proxy._simulator
        self._cache = {}
        self._cache_timestamp = 0

    def _ensure_fresh_cache(self, max_age: float = 0.05):
        """Ensure cache is fresh (within max_age seconds)."""
        import time

        current_time = time.time()
        if current_time - self._cache_timestamp > max_age:
            self._refresh_cache()

    def _refresh_cache(self, mode="by_share_memory"):
        """Refresh object state from remote server."""
        import time

        if mode == "by_share_memory":
            keys = [
                "root_pos_w",
                "root_quat_w",
                "root_lin_vel_w",
                "root_ang_vel_w",
                "default_root_state",
                "default_joint_pos",
                "default_joint_vel",
                "joint_pos",
                "joint_vel",
                "joint_pos_target",
                "joint_vel_target",
            ]
            state_data = self._simulator.get_scene_state(self._object_name, keys)
            for key, value in state_data.items():
                if value is not None:
                    self._cache[key] = torch.tensor(
                        value, device=self._env_proxy.device
                    )
        else:
            # Get pose
            response = self._simulator._send_command_and_recv(
                {
                    "command": "get_object_pose",
                    "name": self._object_name,
                    "env_id": 0,
                }
            )
            if response and response.get("success"):
                self._cache["root_pos_w"] = torch.tensor(
                    response.get("position"), device=self._env_proxy.device
                )
                self._cache["root_quat_w"] = torch.tensor(
                    response.get("quaternion"), device=self._env_proxy.device
                )

            # Get velocity
            response = self._simulator._send_command_and_recv(
                {
                    "command": "get_object_velocity",
                    "name": self._object_name,
                    "env_id": 0,
                }
            )
            if response and response.get("success"):
                self._cache["root_lin_vel_w"] = torch.tensor(
                    response.get("linear_velocity"), device=self._env_proxy.device
                )
                self._cache["root_ang_vel_w"] = torch.tensor(
                    response.get("angular_velocity"), device=self._env_proxy.device
                )

            # Get default state
            response = self._simulator._send_command_and_recv(
                {
                    "command": "get_object_default_state",
                    "name": self._object_name,
                }
            )
            if response and response.get("success"):
                self._cache["default_root_state"] = torch.tensor(
                    response.get("default_state"), device=self._env_proxy.device
                )

            # Get robot-specific data if this is a robot
            if "robot" in self._object_name.lower():
                response = self._simulator._send_command_and_recv(
                    {
                        "command": "get_robot_joint_defaults",
                    }
                )
                if response and response.get("success"):
                    self._cache["default_joint_pos"] = torch.tensor(
                        response.get("default_joint_pos"), device=self._env_proxy.device
                    )
                    self._cache["default_joint_vel"] = torch.tensor(
                        response.get("default_joint_vel"), device=self._env_proxy.device
                    )

                robot_state = self._simulator.get_robot_state()
                if robot_state:
                    if "joint_positions" in robot_state:
                        self._cache["joint_pos"] = torch.tensor(
                            robot_state["joint_positions"],
                            device=self._env_proxy.device,
                        )
                    if "joint_velocities" in robot_state:
                        self._cache["joint_vel"] = torch.tensor(
                            robot_state["joint_velocities"],
                            device=self._env_proxy.device,
                        )

        self._cache_timestamp = time.time()

    @property
    def root_pos_w(self):
        """Get object root position in world frame."""
        self._ensure_fresh_cache()
        return self._cache.get("root_pos_w")

    @property
    def root_quat_w(self):
        """Get object root quaternion in world frame."""
        self._ensure_fresh_cache()
        return self._cache.get("root_quat_w")

    @property
    def root_lin_vel_w(self):
        """Get object root linear velocity in world frame."""
        self._ensure_fresh_cache()
        return self._cache.get("root_lin_vel_w")

    @property
    def root_ang_vel_w(self):
        """Get object root angular velocity in world frame."""
        self._ensure_fresh_cache()
        return self._cache.get("root_ang_vel_w")

    @property
    def default_root_state(self):
        """Get object default root state."""
        self._ensure_fresh_cache()
        return self._cache.get("default_root_state")

    @property
    def default_joint_pos(self):
        """Get robot default joint positions (for robot objects)."""
        self._ensure_fresh_cache()
        return self._cache.get("default_joint_pos")

    @property
    def default_joint_vel(self):
        """Get robot default joint velocities (for robot objects)."""
        self._ensure_fresh_cache()
        return self._cache.get("default_joint_vel")

    @property
    def joint_pos(self):
        """Get robot current joint positions (for robot objects)."""
        self._ensure_fresh_cache()
        return self._cache.get("joint_pos")

    @property
    def joint_vel(self):
        """Get robot current joint velocities (for robot objects)."""
        self._ensure_fresh_cache()
        return self._cache.get("joint_vel")

    @property
    def root_state_w(self):
        """Get combined root state (pos + quat for compatibility)."""
        self._ensure_fresh_cache()
        pos = self._cache.get("root_pos_w")
        quat = self._cache.get("root_quat_w")
        if pos is not None and quat is not None:
            return torch.cat([pos, quat], dim=-1)
        return None

    @property
    def joint_pos_target(self, latest: bool = True):
        """Get robot joint position target (for robot objects)."""
        # For compatibility, we assume joint_pos_target is same as joint_pos

        self._ensure_fresh_cache()
        return self._cache.get("joint_pos_target")


class SceneProxyEnhanced:
    """
    Enhanced scene proxy with env_origins support.
    """

    def __init__(self, env_proxy: EnvProxy):
        self._env_proxy = env_proxy
        self._env_origins = None

    @property
    def env_origins(self):
        """Get environment origins."""
        if self._env_origins is None:
            response = self._env_proxy._simulator._send_command_and_recv(
                {"command": "get_env_origins"}
            )
            if response and response.get("success"):
                self._env_origins = torch.tensor(
                    response.get("env_origins"), device=self._env_proxy.device
                )
        return self._env_origins


class ActionManagerProxy:
    """Proxy for action manager to provide total_action_dim."""

    def __init__(self, env_proxy: EnvProxy):
        self._env_proxy = env_proxy
        self._total_action_dim = None

    @property
    def total_action_dim(self):
        """Get total action dimension."""
        if self._total_action_dim is None:
            response = self._env_proxy._simulator._send_command_and_recv(
                {"command": "get_action_dim"}
            )
            if response and response.get("success"):
                self._total_action_dim = response.get("action_dim")
        return self._total_action_dim


def create_env_proxy(isaac_simulator, scene_mode: str = "default") -> EnvProxy:
    """
    Factory function to create environment proxy.

    Args:
        isaac_simulator: IsaacSimulator instance

    Returns:
        EnvProxy instance
    """
    return EnvProxy(isaac_simulator, scene_mode=scene_mode)
