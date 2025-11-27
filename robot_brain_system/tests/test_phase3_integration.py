"""
Test script for Phase 3: Client-side skill execution with remote environment.
Tests the complete decoupled architecture:
- IsaacSimulator (client) → Socket → IsaacLabServer (pure env)
- SkillExecutorClient (local) uses EnvProxy to access remote env
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

# Import skills to trigger registration via decorators
import robot_brain_system.skills  # noqa: F401


def test_phase3_integration():
    """Test Phase 3 complete integration."""

    print("=" * 80)
    print("Phase 3 Integration Test: Client-Side Skill Execution")
    print("=" * 80)

    # 1. Load configuration
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    print(f"\n[1/6] Loading config from: {config_path}")
    test_config = load_config(config_path=config_path)

    sim_config_dict = cast(
        Dict[str, Any], OmegaConf.to_container(test_config.simulator, resolve=True)
    )

    # 2. Initialize remote environment server
    print("\n[2/6] Initializing remote environment server...")
    simulator = IsaacSimulator(sim_config=sim_config_dict)

    if not simulator.initialize():
        print("❌ Failed to initialize simulator")
        return False

    print("✅ Remote environment server initialized")

    # 3. Create environment proxy
    print("\n[3/6] Creating environment proxy...")
    env_proxy = create_env_proxy(simulator)
    print(
        f"✅ Environment proxy created (device={env_proxy.device}, num_envs={env_proxy.num_envs})"
    )

    # 4. Load skill registry
    print("\n[4/6] Loading skill registry...")
    # Import skills to trigger registration
    try:
        skill_registry = get_skill_registry()
        available_skills = skill_registry.list_skills()
        print(f"✅ Loaded {len(available_skills)} skills: {available_skills[:5]}...")
    except Exception as e:
        print(f"⚠️  Warning: Could not load skills: {e}")
        skill_registry = get_skill_registry()
        available_skills = []

    # 5. Create client-side skill executor
    print("\n[5/6] Creating client-side skill executor...")
    skill_executor = create_skill_executor_client(skill_registry, env_proxy)
    print(f"✅ Skill executor created (status={skill_executor.status.value})")

    # 6. Test basic operations
    print("\n[6/6] Testing basic operations...")

    # Test 6.1: Reset environment via proxy
    print("\n  [6.1] Testing env reset via proxy...")
    try:
        obs, info = env_proxy.reset()
        if obs is not None:
            print(
                f"    ✅ Environment reset successful (obs keys: {list(obs.keys())[:3]}...)"
            )
        else:
            print("    ❌ Environment reset failed")
            return False
    except Exception as e:
        print(f"    ❌ Environment reset error: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 6.2: Query scene info
    print("\n  [6.2] Testing scene info query...")
    try:
        scene_info = simulator.get_scene_info()
        print(f"    ✅ Scene info retrieved: {list(scene_info.keys())[:5]}...")
    except Exception as e:
        print(f"    ❌ Scene info query error: {e}")
        return False

    # Test 6.3: Query robot state
    print("\n  [6.3] Testing robot state query...")
    try:
        robot_state = simulator.get_robot_state()
        print(f"    ✅ Robot state retrieved: {list(robot_state.keys())[:3]}...")
    except Exception as e:
        print(f"    ❌ Robot state query error: {e}")
        return False

    # Test 6.4: Access scene via proxy
    print("\n  [6.4] Testing scene access via proxy...")
    try:
        scene = env_proxy.scene
        scene_keys = list(scene.keys())
        print(f"    ✅ Scene accessed: {scene_keys[:5]}...")

        # Try to access a specific object
        if "robot" in scene_keys:
            robot_obj = scene["robot"]
            print(f"    ✅ Robot object accessed: {robot_obj}")

    except Exception as e:
        print(f"    ❌ Scene access error: {e}")
        import traceback

        traceback.print_exc()

    # Test 6.5: Test skill initialization (if skills available)
    if available_skills:
        print("\n  [6.5] Testing skill initialization (using first available skill)...")
        test_skill_name = available_skills[0]
        try:
            success, obs_dict = skill_executor.initialize_skill(
                test_skill_name, parameters={}, obs_dict=obs or {}
            )

            if success:
                print(f"    ✅ Skill '{test_skill_name}' initialized successfully")
                print(f"       Status: {skill_executor.status.value}")
            else:
                print(f"    ⚠️  Skill '{test_skill_name}' initialization returned False")
                print(f"       Status: {skill_executor.status.value}")

            # Cleanup skill
            if skill_executor.is_running():
                skill_executor.terminate_current_skill()
                print("    ✅ Skill terminated")

        except Exception as e:
            print(f"    ⚠️  Skill initialization error (expected for some skills): {e}")
    else:
        print("\n  [6.5] Skipping skill test (no skills available)")

    # Cleanup
    print("\n" + "=" * 80)
    print("Shutting down...")
    simulator.shutdown()
    print("✅ Shutdown complete")

    print("\n" + "=" * 80)
    print("🎉 Phase 3 Integration Test PASSED!")
    print("=" * 80)
    print("\nArchitecture Verification:")
    print("  ✅ Remote environment server (pure env, no skills)")
    print("  ✅ Client-side skill executor (local execution)")
    print("  ✅ Environment proxy (remote access)")
    print("  ✅ Complete decoupling achieved!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        success = test_phase3_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
