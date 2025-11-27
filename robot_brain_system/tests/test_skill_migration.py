"""
Test script for migrated skills with EnvProxy.
Tests that old skills can work with new remote environment architecture.
"""

import sys
from pathlib import Path

# Add workspace to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from robot_brain_system.core.isaac_simulator import IsaacSimulator
from robot_brain_system.core.env_proxy import create_env_proxy
from robot_brain_system.core.skill_executor_client import create_skill_executor_client
from robot_brain_system.core.skill_manager import get_skill_registry
from robot_brain_system.utils.config_utils import load_config
from omegaconf import OmegaConf
from typing import cast, Dict, Any

# Import skills to trigger registration
import robot_brain_system.skills  # noqa: F401


def test_press_button_skill():
    """Test PressButton (open_box) skill with EnvProxy."""

    print("=" * 80)
    print("Skill Migration Test: PressButton (open_box)")
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

        # Check if open_box skill is available
        if "open_box" not in available_skills:
            print("❌ 'open_box' skill not found in registry")
            print(f"   Available skills: {available_skills}")
            return False

        print("✅ Found 'open_box' skill")

        # 5. Create skill executor
        print("\n[5/7] Creating skill executor...")
        skill_executor = create_skill_executor_client(skill_registry, env_proxy)
        print(f"✅ Skill executor created (status={skill_executor.status.value})")

        # 6. Initialize PressButton skill
        print("\n[6/7] Testing PressButton skill initialization...")
        print("   This will test:")
        print("   - env.scene['box'], env.scene['heavy_box'], env.scene['spanner']")
        print("   - obj.data.root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w")
        print("   - obj.write_root_pose_to_sim(), write_root_velocity_to_sim()")
        print("   - robot.data.default_joint_pos, default_joint_vel")
        print("   - robot.set_joint_position_target(), write_joint_state_to_sim()")
        print("   - env.action_manager.total_action_dim")
        print("   - env.scene.env_origins")

        success = skill_executor.initialize_skill(
            "open_box", parameters={}, success_item=None, timeout_item=None
        )

        if success:
            print("\n✅ PressButton skill initialized successfully!")
            print(f"   Current status: {skill_executor.status.value}")
            print(f"   Current skill: {skill_executor.current_skill_name}")

            # 7. Test a few steps - just wait for skill to execute
            print("\n[7/7] Testing skill execution (5 steps)...")
            import time

            for i in range(5):
                time.sleep(0.1)  # Let skill execute in background
                status = skill_executor.status.value
                print(f"   Step {i + 1}: Skill status={status}")
                if status in ["completed", "failed"]:
                    break

            # Cleanup
            print("\n   Terminating skill...")
            skill_executor.terminate_current_skill()
            print("   ✅ Skill terminated")

            return True
        else:
            print("\n❌ PressButton skill initialization failed")
            print(f"   Status: {skill_executor.status.value}")
            return False

    finally:
        print("\n" + "=" * 80)
        print("Shutting down...")
        simulator.shutdown()
        print("✅ Shutdown complete")


if __name__ == "__main__":
    success = test_press_button_skill()

    print("\n" + "=" * 80)
    if success:
        print("🎉 SKILL MIGRATION TEST PASSED!")
        print("PressButton skill works with EnvProxy architecture")
    else:
        print("❌ SKILL MIGRATION TEST FAILED")
        print("PressButton skill needs further fixes")
    print("=" * 80)

    sys.exit(0 if success else 1)
