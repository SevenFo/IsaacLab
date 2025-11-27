"""
Test a list of skills
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
from robot_brain_system.utils.config_utils import load_config

# Import skills to trigger registration
import robot_brain_system.skills  # noqa: F401


def test_skills():
    """Test a list of skill"""
    print("=" * 80)
    print("Skills Test")
    print("=" * 80)

    # 1. Load configuration
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    print(f"\n[1/7] Loading config from: {config_path}")
    test_config = load_config(config_path=config_path)

    config = cast(Dict[str, Any], OmegaConf.to_container(test_config, resolve=True))
    sim_config_dict = config["simulator"]
    sim_config_dict["skills"] = config["skills"]

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

        # 5. Create skill executor
        print("\n[5/8] Creating skill executor...")
        skill_executor = SkillExecutorClient(skill_registry, env_proxy)
        print(f"✅ Skill executor created (status={skill_executor.status.value})")

        # Start the skill with alice_control parameter
        skill_params = {"mode": "dynamic"}
        skill_plan = [
            {"skill_name": "move_box_to_suitable_position", "parameters": {}},
            # {"skill_name": "open_box", "parameters": {}},
            # {"skill_name": "grasp_spanner", "parameters": {}},
            {
                "skill_name": "move_to_target_object",
                "parameters": {"target_object": "human hand"},
            },
        ]
        for step in skill_plan:
            print(f"initializing skill: {step['skill_name']}")
            if step["skill_name"] == "move_to_target_object":
                skill_executor.move_alice_to_operation_position()
            success, obs = skill_executor.initialize_skill(
                step["skill_name"], parameters=step["parameters"]
            )
            print(f"\n✅ {step['skill_name']} skill initialized successfully!")
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
                print(f"\n✅ {step['skill_name']} skill is running")
            elif skill_executor.status.value == "completed":
                print(f"\n✅ {step['skill_name']} completed initialization!")
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
        print("🎉 TEST PASSED!")
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
    success = test_skills()
    sys.exit(0 if success else 1)
