# robot_brain_system/isaac_launcher.py
import sys
from omegaconf import OmegaConf, DictConfig


def _print_imports(context_message):
    """Helper function to print imported modules at a specific point."""
    print(f"\n--- DEBUG: Imported Modules at '{context_message}' ---")
    top_level_modules = {m.split(".")[0] for m in sys.modules if not m.startswith("_")}
    for module_name in sorted(list(top_level_modules)):
        print(f"  - {module_name}")
    print(f"--- END DEBUG ({len(sys.modules)} total modules) ---\n")


def isaac_process_entry(child_conn, sim_config:DictConfig):
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

    app_launcher_params = OmegaConf.to_container(sim_config, resolve=True)
    assert type(app_launcher_params) == dict
    
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
