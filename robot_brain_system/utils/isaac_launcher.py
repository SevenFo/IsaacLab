# robot_brain_system/isaac_launcher.py
import sys


def _print_imports(context_message):
    """Helper function to print imported modules at a specific point."""
    print(f"\n--- DEBUG: Imported Modules at '{context_message}' ---")
    top_level_modules = {m.split(".")[0] for m in sys.modules if not m.startswith("_")}
    for module_name in sorted(list(top_level_modules)):
        print(f"  - {module_name}")
    print(f"--- END DEBUG ({len(sys.modules)} total modules) ---\n")


def isaac_process_entry(child_conn, sim_config):
    """
    This is the minimal entry point for the Isaac Sim subprocess.
    Its only job is to initialize Isaac Sim and then hand over control
    to the main simulation logic function.
    """
    _print_imports("Start of isaac_process_entry")
    print("[IsaacLauncher] Subprocess started. Initializing Isaac Sim...")

    # 1. Import AppLauncher - a necessary step to start the simulation
    try:
        from isaaclab.app import AppLauncher
    except ImportError as e:
        print(f"[IsaacLauncher] Critical Error: Failed to import isaaclab.app. {e}")
        child_conn.send(
            {
                "status": "error",
                "error": "Failed to import isaaclab.app. Check Isaac Lab installation.",
            }
        )
        child_conn.close()
        return

    app_launcher_params = {
        "task": sim_config.get("env_name", "Isaac-Move-Box-Frank-IK-Rel"),
        "device": sim_config.get(
            "device", "cuda:0"
        ),  # AppLauncher might use this for 'sim_device' default
        "num_envs": sim_config.get("num_envs", 1),
        "disable_fabric": sim_config.get("disable_fabric", False),
        "mode": sim_config.get(
            "mode", 1
        ),  # Make sure this value matches AppLauncher expectations
        "env_config_file": sim_config.get("env_config_file"),
        # Arguments AppLauncher specifically uses/pops (add more as needed):
        "enable_cameras": sim_config.get("enable_cameras", True),
        "headless": sim_config.get("headless", False),
        # "livestream": sim_config.get(
        #     "livestream", False
        # ),  # This was the one causing the error
        "sim_device": sim_config.get(
            "sim_device", sim_config.get("device", "cuda:0")
        ),  # AppLauncher often uses 'sim_device'
        # "cpu": sim_config.get("cpu", False),
        # "physics_gpu": sim_config.get("physics_gpu", -1),
        # "graphics_gpu": sim_config.get("graphics_gpu", -1),
        # "pipeline": sim_config.get("pipeline", "gpu"),
        # "fabric_gpu": sim_config.get("fabric_gpu", -1),
        # "kit_app": sim_config.get("kit_app", None),
        # "enable_ros": sim_config.get("enable_ros", False),
        # "ros_domain_id": sim_config.get("ros_domain_id", 0),
        # "verbosity": sim_config.get("verbosity", "info"),
        # "build_path": sim_config.get("build_path", None),
        # # Add any other arguments AppLauncher defines in its command-line parsing
    }

    # 2. Launch the simulation app
    # We pass the entire sim_config here, AppLauncher will pick what it needs.
    app_launcher = AppLauncher(app_launcher_params)

    # 3. Isaac Sim is now running. Now we can import the rest of our application logic.
    # We do this INSIDE this function to ensure they are only imported in the subprocess
    # AFTER the simulation is initialized.
    from robot_brain_system.core.isaac_simulator import IsaacSimulator

    # 4. Call the main simulation logic function, which now assumes Isaac is running.
    # We re-use the _isaac_simulation_entry logic, but now it's just a regular function call.
    IsaacSimulator._isaac_simulation_entry(
        child_conn, app_launcher, app_launcher_params
    )

    # The _isaac_simulation_entry function's finally block will handle cleanup.
    print(
        "[IsaacLauncher] Main simulation logic function has exited. Subprocess terminating."
    )
