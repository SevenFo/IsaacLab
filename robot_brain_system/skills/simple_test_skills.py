"""
Simple test skills for Phase 3 integration testing.
These skills are designed to work with EnvProxy (remote environment).
"""

import torch

from ..core.types import BaseSkill, SkillType, ExecutionMode, Action
from ..core.skill_manager import skill_register


@skill_register(
    name="test_simple_move",
    skill_type=SkillType.POLICY,
    execution_mode=ExecutionMode.STEPACTION,
    timeout=10.0,
    enable_monitoring=False,
    requires_env=True,
    criterion={
        "successed": "Robot moved successfully",
        "progress": "Robot is moving",
    },
    enable=False,
)
class TestSimpleMove(BaseSkill):
    """
    Simple test skill that executes a few steps with zero actions.
    Used for testing the Phase 3 architecture.

    Expected params: None
    """

    def __init__(self, policy_device: str = "cuda", **running_params):
        super().__init__()
        self.policy_device = policy_device
        self.num_steps = 0
        self.max_steps = 5
        print("[TestSimpleMove] Skill created")

    def initialize(self, env):
        """Initialize the skill."""
        print("[TestSimpleMove] Initializing...")
        super().initialize(env)

        # Get action dimension from env proxy
        # Note: env is now an EnvProxy, not the real environment
        print(f"[TestSimpleMove] Env device: {env.device}")
        print(f"[TestSimpleMove] Num envs: {env.num_envs}")

        # Query scene info through proxy
        scene_info = env._simulator.get_scene_info()
        print(f"[TestSimpleMove] Scene objects: {list(scene_info.keys())}")

        # Get robot state through proxy
        robot_state = env._simulator.get_robot_state()
        print(f"[TestSimpleMove] Robot state keys: {list(robot_state.keys())}")

        self.num_steps = 0
        print("[TestSimpleMove] Initialized successfully")

        # Get initial observation
        obs = env.get_observations()
        return obs

    def select_action(self, obs_dict: dict) -> Action:
        """Select action based on observation."""
        self.num_steps += 1

        print(f"[TestSimpleMove] Step {self.num_steps}/{self.max_steps}")

        if self.num_steps >= self.max_steps:
            print("[TestSimpleMove] Completed!")
            return Action(
                data=[], metadata={"info": "finished", "reason": "max_steps_reached"}
            )

        # Return zero action (safe no-op)
        # Action dimension should match env's action space
        # For now, return a 7D action (common for arm + gripper)
        action_data = torch.zeros(1, 7, device=self.env.device)
        action_data[..., -1] = 1.0  # Keep gripper open

        return Action(data=action_data, metadata={"info": "success", "reason": "none"})


@skill_register(
    name="test_query_scene",
    skill_type=SkillType.POLICY,
    execution_mode=ExecutionMode.STEPACTION,
    timeout=10.0,
    enable_monitoring=False,
    requires_env=True,
    criterion={
        "successed": "Scene queried successfully",
        "progress": "Querying scene",
    },
    enable=False,
)
class TestQueryScene(BaseSkill):
    """
    Test skill that queries scene information through EnvProxy.
    Demonstrates how to access remote scene data.

    Expected params: None
    """

    def __init__(self, policy_device: str = "cuda", **running_params):
        super().__init__()
        self.policy_device = policy_device
        self.queried = False
        print("[TestQueryScene] Skill created")

    def initialize(self, env):
        """Initialize the skill."""
        print("[TestQueryScene] Initializing...")
        super().initialize(env)

        # Query all available scene information
        print("\n" + "=" * 60)
        print("SCENE INFORMATION QUERY")
        print("=" * 60)

        # Get scene structure
        scene_info = env._simulator.get_scene_info()
        print(f"\nScene objects ({len(scene_info)}):")
        for obj_name, obj_info in scene_info.items():
            print(f"  - {obj_name}:")
            if "position" in obj_info:
                pos = obj_info["position"][0]  # First env
                print(f"      Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            if "orientation" in obj_info:
                ori = obj_info["orientation"][0]
                print(
                    f"      Orientation: [{ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f}]"
                )
            if "joint_positions" in obj_info:
                joints = obj_info["joint_positions"][0]
                print(f"      Joints: {len(joints)} DOFs")

        # Get robot-specific state
        robot_state = env._simulator.get_robot_state()
        print("\nRobot state:")
        for key, value in robot_state.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list):
                    print(f"  - {key}: {len(value[0])} values")
                else:
                    print(f"  - {key}: {value}")

        print("=" * 60 + "\n")

        obs = env.get_observations()
        return obs

    def select_action(self, obs_dict: dict) -> Action:
        """Select action based on observation."""
        if not self.queried:
            print("[TestQueryScene] Scene queried, finishing skill")
            self.queried = True

        # Finish immediately after one query
        return Action(
            data=[], metadata={"info": "finished", "reason": "query_complete"}
        )
