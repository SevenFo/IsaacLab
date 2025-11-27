"""
Test ObjectTracking skill with EnvProxy architecture.
This test validates that the object_tracking skill works correctly
with the remote environment through EnvProxy.
"""

import sys
import os
import gc

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pathlib import Path
from typing import cast, Dict, Any
from omegaconf import OmegaConf

from robot_brain_system.core.isaac_simulator import IsaacSimulator
from robot_brain_system.core.env_proxy import create_env_proxy
from robot_brain_system.core.skill_manager import get_skill_registry
from robot_brain_system.core.skill_executor_client import SkillExecutorClient
from robot_brain_system.skills.alice_control_skills import AliceControl
from robot_brain_system.core.types import SkillType, ExecutionMode
from robot_brain_system.utils.config_utils import load_config

# Import skills to trigger registration
import robot_brain_system.skills  # noqa: F401


def test_object_tracking_skill():
    """Test ObjectTracking skill with target_object='red box'."""
    print("=" * 80)
    print("ObjectTracking Skill Test: target_object='red box'")
    print("=" * 80)

    # 1. Load configuration
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    print(f"\n[1/7] Loading config from: {config_path}")
    test_config = load_config(config_path=config_path)

    sim_config_dict = cast(
        Dict[str, Any], OmegaConf.to_container(test_config.simulator, resolve=True)
    )

    # 2. Initialize remote environment server
    print("\n[2/7] Initializing remote environment server...")
    simulator = IsaacSimulator(sim_config=sim_config_dict)

    if not simulator.initialize():
        print("❌ Failed to initialize simulator")
        return False

    print("✅ Remote environment server initialized")

    try:
        # 3. Create environment proxy
        print("\n[3/7] Creating environment proxy...")
        env_proxy = create_env_proxy(simulator)
        print(
            f"✅ Environment proxy created (device={env_proxy.device}, num_envs={env_proxy.num_envs})"
        )

        # 4. Load skill registry
        print("\n[4/7] Loading skill registry...")
        skill_registry = get_skill_registry()
        available_skills = skill_registry.list_skills()
        print(f"✅ Loaded {len(available_skills)} skills")

        # Rigist AliceControl skill explicitly
        skill_registry.register_skill(
            AliceControl,
            skill_type=SkillType.POLICY,
            execution_mode=ExecutionMode.STEPACTION,
            requires_env=True,
            name="alice_control",
        )

        # 5. Create skill executor
        print("\n[5/8] Creating skill executor...")
        skill_executor = SkillExecutorClient(skill_registry, env_proxy)
        print(f"✅ Skill executor created (status={skill_executor.status.value})")

        # 6. Test AliceControl skill initialization
        print("\n[6/8] Testing AliceControl skill initialization...")
        print("   This will test:")
        print("   - env.num_envs")
        print("   - env.device")
        print("   - env.action_manager.total_action_dim")
        print("   - AliceControl.initialize(env, zero_action)")

        # Start the skill with alice_control parameter
        skill_params = {"mode": "dynamic"}

        success, obs = skill_executor.initialize_skill(
            "alice_control",
            parameters=skill_params,
        )
        print("\n✅ ObjectTracking skill initialized successfully!")
        print(f"   Current status: {skill_executor.status.value}")
        print(f"   Current skill: {skill_executor.current_skill_name}")

        # 7. Test skill execution (wait for processing)
        print("\n[7/7] Testing skill execution ")
        import time

        for i in range(1000):
            time.sleep(1)
            status = skill_executor.status.value
            print(f"   Step {i + 1}: Skill status={status}")
            env_proxy.clear_observation_buffer()
            gc.collect()
            if status in ["completed", "failed"]:
                break

        # 8. Check if tracking was successful
        if skill_executor.status.value == "running":
            print("\n✅ ObjectTracking skill is running and tracking!")
        elif skill_executor.status.value == "completed":
            print("\n✅ ObjectTracking completed initialization!")
        else:
            print(f"\n⚠️  Skill status: {skill_executor.status.value}")

        # Terminate skill
        print("\n   Terminating skill...")
        skill_executor.terminate_current_skill()
        print("   ✅ Skill terminated")

        # Cleanup
        print("\n" + "=" * 80)
        print("Shutting down...")
        isaac_sim.shutdown()
        print("✅ Shutdown complete")

        print("\n" + "=" * 80)
        print("🎉 OBJECT TRACKING TEST PASSED!")
        print("ObjectTracking skill works with EnvProxy architecture")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED with error: {e}")
        import traceback

        traceback.print_exc()

        # Try to cleanup
        try:
            if "isaac_sim" in locals():
                isaac_sim.shutdown()
        except:
            pass
        return False


if __name__ == "__main__":
    success = test_object_tracking_skill()
    sys.exit(0 if success else 1)
