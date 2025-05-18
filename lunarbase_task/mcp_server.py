# mcp_anthropic_server.py
import argparse
import os
import threading
from typing import Optional, List, Dict, Any

# Pydantic for defining data models (schemas for tools)
from pydantic import BaseModel, Field

# mcp-server-python SDK imports
from mcp_server import (
    MCPApplication,
    Tool,
    ToolCallContext,
    # Resource, # If we were defining resources
    # PromptTemplate, # If we were defining prompt templates
)
import uvicorn  # For running the ASGI application

# Import our LunarBaseEnv (assuming it's runnable and accessible)
# In a real deployment, how this env instance is shared or accessed by tools
# needs careful consideration (e.g., global instance, dependency injection, IPC).
try:
    from lunar_env import (
        LunarBaseEnv,
        MultiObjectSceneCfg,
    )  # Ensure this path is correct
except ImportError as e:
    print(f"ERROR: Could not import LunarBaseEnv: {e}. MCP server relies on it.")
    print(
        "Ensure lunar_env.py is accessible and Isaac Lab environment is set up if running env directly."
    )
    LunarBaseEnv = None  # Make it None to allow script to load for SDK exploration

# --- Global State (Conceptual: for managing the LunarBaseEnv instance) ---
# In a production scenario, this would need robust management.
# For this example, we'll assume a single, globally accessible env instance.
# This instance would need to be initialized *before* MCP tools try to use it.
LUNAR_ENV_INSTANCE: Optional[LunarBaseEnv] = None
LUNAR_ENV_LOCK = (
    threading.Lock()
)  # To protect access if env is not inherently thread-safe for queries


# --- Helper function to initialize LunarBaseEnv (called once) ---
def initialize_lunar_env_globally(env_config: dict):
    """
    Initializes the LunarBaseEnv instance.
    This should ideally be called once when the server process starts,
    before any MCP tools try to access it.
    IMPORTANT: This function assumes it's called in a context where
    Isaac Sim can be initialized (e.g., main thread, or a process
    where ./isaaclab.sh has set up the environment).
    """
    global LUNAR_ENV_INSTANCE
    if LunarBaseEnv is None:
        print("ERROR: LunarBaseEnv class not loaded. Cannot initialize environment.")
        return

    with LUNAR_ENV_LOCK:
        if LUNAR_ENV_INSTANCE is None or LUNAR_ENV_INSTANCE._is_closed:
            print("[MCP Server] Initializing global LunarBaseEnv instance...")
            try:
                # Scene config can be customized here if needed
                scene_cfg = MultiObjectSceneCfg()  # Using default from lunar_env.py
                # Ensure only valid args are passed to LunarBaseEnv
                valid_env_args = {
                    k: v
                    for k, v in env_config.items()
                    if k in LunarBaseEnv.__init__.__code__.co_varnames
                }
                LUNAR_ENV_INSTANCE = LunarBaseEnv(scene_cfg=scene_cfg, **valid_env_args)
                # Note: LunarBaseEnv constructor calls sim.reset()
                print(
                    "[MCP Server] Global LunarBaseEnv instance initialized successfully."
                )
            except Exception as e:
                print(f"[MCP Server] FATAL: Failed to initialize LunarBaseEnv: {e}")
                import traceback

                traceback.print_exc()
                LUNAR_ENV_INSTANCE = None  # Ensure it's None if init fails
                # In a real app, you might want to prevent the MCP server from starting.
        else:
            print("[MCP Server] Global LunarBaseEnv instance already initialized.")


def get_lunar_env() -> LunarBaseEnv:
    """Safely gets the LunarBaseEnv instance, raising an error if not available."""
    with LUNAR_ENV_LOCK:
        if LUNAR_ENV_INSTANCE is None:
            raise RuntimeError(
                "LunarBaseEnv instance is not initialized. "
                "The MCP server might need to be configured to initialize it, "
                "or it needs to be started separately."
            )
        if LUNAR_ENV_INSTANCE._is_closed:
            raise RuntimeError("LunarBaseEnv instance has been closed.")
        return LUNAR_ENV_INSTANCE


# --- Define Pydantic Models for Tool Schemas ---


class GetEnvironmentOverviewParams(BaseModel):
    # No input parameters for this tool
    pass


class EnvironmentOverviewOutput(BaseModel):
    task_description: str = Field(description="A brief description of the RL task.")
    num_simulation_envs: int = Field(
        description="Number of parallel simulation environments."
    )
    max_episode_steps: int = Field(description="Maximum number of steps per episode.")
    observation_space_shape: List[int] = Field(
        description="Shape of the observation space."
    )
    action_space_shape: List[int] = Field(description="Shape of the action space.")
    is_simulation_ready: bool = Field(
        description="Indicates if the simulation environment is initialized and ready."
    )
    is_simulation_closed: bool = Field(
        description="Indicates if the simulation environment has been closed."
    )


class GetObjectStateParams(BaseModel):
    env_idx: int = Field(
        0,
        description="Index of the specific parallel environment to query (if num_envs > 1).",
        ge=0,
    )
    object_name: Optional[str] = Field(
        None,
        description="Optional: Name or identifier of the object if multiple objects exist. Default: 'object'.",
    )


class ObjectState(BaseModel):
    position_world: List[float] = Field(
        description="Object's [x, y, z] position in the world frame."
    )
    orientation_quaternion_world: List[float] = Field(
        description="Object's [w, x, y, z] orientation quaternion in the world frame."
    )
    linear_velocity_world: List[float] = Field(
        description="Object's [vx, vy, vz] linear velocity in the world frame."
    )
    angular_velocity_world: List[float] = Field(
        description="Object's [ax, ay, az] angular velocity in the world frame."
    )


class GetObjectStateOutput(BaseModel):
    env_idx: int
    object_name_queried: str
    state: Optional[ObjectState] = None
    error: Optional[str] = None


# --- Define MCP Tools ---


class GetEnvironmentOverviewTool(
    Tool[GetEnvironmentOverviewParams, EnvironmentOverviewOutput]
):
    name = "get_environment_overview"
    description = "Provides a general overview of the LunarBase simulation environment, including its configuration and current status."
    # No specific parameters class needed if there are no inputs, but Pydantic model is still good practice.
    # Parameters = GetEnvironmentOverviewParams (implicitly if not set, or use type hint)
    # Output = EnvironmentOverviewOutput (implicitly if not set, or use type hint)

    async def call(
        self, params: GetEnvironmentOverviewParams, context: ToolCallContext
    ) -> EnvironmentOverviewOutput:
        print(
            f"[MCP Tool] '{self.name}' called by client (ID: {context.client_id}, Session: {context.session_id})"
        )
        try:
            env = get_lunar_env()  # This will raise if env is not ready
            # Accessing properties of the env instance
            # Ensure these properties exist on your LunarBaseEnv
            obs_space_shape = (
                list(env.observation_space.shape) if env.observation_space else []
            )
            act_space_shape = list(env.action_space.shape) if env.action_space else []

            return EnvironmentOverviewOutput(
                task_description="Control a robotic arm in a lunar base to interact with objects.",
                num_simulation_envs=env.num_envs,
                max_episode_steps=env._max_episode_length,
                observation_space_shape=obs_space_shape,
                action_space_shape=act_space_shape,
                is_simulation_ready=True,  # If get_lunar_env() succeeded
                is_simulation_closed=env._is_closed,  # Should be False if we got here
            )
        except RuntimeError as e:  # From get_lunar_env()
            return EnvironmentOverviewOutput(
                task_description="N/A",
                num_simulation_envs=0,
                max_episode_steps=0,
                observation_space_shape=[],
                action_space_shape=[],
                is_simulation_ready=False,
                is_simulation_closed=LUNAR_ENV_INSTANCE._is_closed
                if LUNAR_ENV_INSTANCE
                else True,
                # You might want a more specific error field in the output schema for this case
            )
        except Exception as e:
            print(f"[MCP Tool] Error in '{self.name}': {e}")
            # For production, map to a proper error structure in EnvironmentOverviewOutput
            # or raise an MCPError from mcp_server.errors
            raise  # Re-raise for now, MCP framework might handle it or convert to 500


class GetObjectStateTool(Tool[GetObjectStateParams, GetObjectStateOutput]):
    name = "get_object_state"
    description = "Retrieves the current kinematic state (position, orientation, velocities) of a specified object in a given environment instance."

    async def call(
        self, params: GetObjectStateParams, context: ToolCallContext
    ) -> GetObjectStateOutput:
        print(f"[MCP Tool] '{self.name}' called with params: {params}")
        try:
            env = get_lunar_env()
            queried_object_name = (
                params.object_name if params.object_name else "object"
            )  # Default to "object"

            if not (0 <= params.env_idx < env.num_envs):
                return GetObjectStateOutput(
                    env_idx=params.env_idx,
                    object_name_queried=queried_object_name,
                    error=f"Invalid env_idx: {params.env_idx}. Must be between 0 and {env.num_envs - 1}.",
                )

            # Accessing data from the environment instance.
            # This assumes env.object.data.root_state_w is updated and available.
            # For thread safety with Isaac Sim, queries should be careful if sim is stepping.
            # Here, we assume data is from the last completed simulation step.
            with LUNAR_ENV_LOCK:  # Protect read if necessary
                # Ensure data buffers are fresh. This is tricky. MCP calls are out-of-band from sim steps.
                # A robust solution might involve the env periodically publishing its state,
                # or these tools triggering a non-stepping 'update_buffers' call if possible.
                # For now, assume env.object.data contains reasonably recent state.
                obj_data = (
                    env.object.data
                )  # This is an ArticulationData or RigidObjectData

                # root_state_w is (num_envs, 13)
                # [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, lin_vel_x, ..., ang_vel_z]
                root_state_tensor = obj_data.root_state_w[params.env_idx].cpu().numpy()

            state = ObjectState(
                position_world=root_state_tensor[0:3].tolist(),
                orientation_quaternion_world=root_state_tensor[3:7].tolist(),
                linear_velocity_world=root_state_tensor[7:10].tolist(),
                angular_velocity_world=root_state_tensor[10:13].tolist(),
            )
            return GetObjectStateOutput(
                env_idx=params.env_idx,
                object_name_queried=queried_object_name,
                state=state,
            )
        except RuntimeError as e:  # From get_lunar_env()
            return GetObjectStateOutput(
                env_idx=params.env_idx,
                object_name_queried=params.object_name or "object",
                error=str(e),
            )
        except Exception as e:
            print(f"[MCP Tool] Error in '{self.name}': {e}")
            return GetObjectStateOutput(
                env_idx=params.env_idx,
                object_name_queried=params.object_name or "object",
                error=f"Internal server error: {type(e).__name__} - {str(e)}",
            )


# --- Create MCP Application and Register Tools ---
# This application object will be served by Uvicorn.
mcp_app = MCPApplication(
    title="LunarBase Environment MCP Server",
    description="Exposes tools to interact with and query a simulated LunarBase environment within Isaac Lab.",
    version="0.1.0",
    # tools_router_prefix="/custom_tools", # Optional: if you want to change the default /mcp/tools prefix
)

# Register the defined tools
mcp_app.register_tool(GetEnvironmentOverviewTool())
mcp_app.register_tool(GetObjectStateTool())
# mcp_app.register_resource(...) # If you had MCP Resources
# mcp_app.register_prompt_template(...) # If you had Prompt Templates


# --- Main block to run the Uvicorn server ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP Server for LunarBaseEnv (using mcp-server-python SDK)."
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Hostname to bind the MCP server."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the MCP server (default for MCP is often 8080).",
    )  # MCP default
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn auto-reload (for development).",
    )

    # Arguments for initializing the LunarBaseEnv instance
    parser.add_argument(
        "--env_num_envs",
        type=int,
        default=1,
        help="Number of environments for the LunarBaseEnv instance.",
    )
    parser.add_argument(
        "--env_headless", action="store_true", help="Run LunarBaseEnv in headless mode."
    )
    parser.add_argument(
        "--env_device",
        type=str,
        default="cuda:0",
        help="Device for LunarBaseEnv (e.g., cuda:0, cpu).",
    )
    parser.add_argument(
        "--env_max_episode_length",
        type=int,
        default=250,
        help="Max episode length for LunarBaseEnv.",
    )
    # Add other relevant LunarBaseEnv constructor arguments here

    cli_args = parser.parse_args()

    # Prepare LunarBaseEnv configuration
    lunar_env_config = {
        "num_envs": cli_args.env_num_envs,
        "headless": cli_args.env_headless,
        "device": cli_args.env_device,
        "max_episode_length": cli_args.env_max_episode_length,
        # pass other env-specific args here
    }

    # --- IMPORTANT ---
    # How and when LunarBaseEnv is initialized is CRITICAL.
    # Isaac Sim typically requires its application to be managed carefully, often in the main thread.
    # Running Uvicorn (ASGI server) and Isaac Sim in the same process can be complex.
    #
    # Option 1 (Simplest for Demo): Initialize LunarBaseEnv synchronously before starting Uvicorn.
    # This means the main thread will run Isaac Sim initialization, then Uvicorn.
    # This is what `initialize_lunar_env_globally` attempts.
    # This script needs to be run via `./isaaclab.sh -p your_script.py ...` for this to work.
    #
    # Option 2 (More Robust for Production):
    #   - Run LunarBaseEnv in a separate dedicated process (e.g., using the RPyC server or a custom one).
    #   - The MCP tools would then communicate with that separate LunarBaseEnv process via IPC (e.g., RPyC calls, sockets).
    #   - This decouples the lifecycles and threading models.
    #
    # For this conceptual script, we proceed with Option 1, assuming it's run via Isaac Lab's launcher.

    print("--- LunarBase MCP Server ---")
    print(
        "Attempting to initialize LunarBaseEnv (if not already done by another part of your app)..."
    )
    # If this script is the *sole entry point* for both Isaac Sim and MCP server:
    initialize_lunar_env_globally(
        lunar_env_config
    )  # This will try to start Isaac Sim via AppLauncher

    if LUNAR_ENV_INSTANCE is None or LUNAR_ENV_INSTANCE._is_closed:
        print(
            "ERROR: LunarBaseEnv could not be initialized or is closed. MCP server functionality will be limited or non-operational."
        )
        print(
            "Ensure Isaac Lab environment is set up and paths in lunar_env.py are correct."
        )
        # Optionally, exit if the env is critical for all tools
        # exit(1)
    else:
        print(
            "LunarBaseEnv appears to be initialized. MCP tools can attempt to use it."
        )

    print(f"Starting MCP server on http://{cli_args.host}:{cli_args.port}")
    print(
        f"  MCP Spec (OpenAPI) will be available at: http://{cli_args.host}:{cli_args.port}{mcp_app.openapi_url}"
    )
    print(
        f"  Registered tools will be under: http://{cli_args.host}:{cli_args.port}{mcp_app.tools_router_prefix}"
    )

    uvicorn.run(
        "mcp_anthropic_server:mcp_app",  # Points to the mcp_app instance in this file
        host=cli_args.host,
        port=cli_args.port,
        reload=cli_args.reload,
        # workers=1, # Uvicorn workers. Usually 1 if sharing a non-thread-safe global like Isaac Sim.
        # log_level="info",
    )

    # Cleanup after Uvicorn finishes (e.g., on Ctrl+C)
    print("MCP Server (Uvicorn) has shut down.")
    with LUNAR_ENV_LOCK:
        if LUNAR_ENV_INSTANCE and not LUNAR_ENV_INSTANCE._is_closed:
            print("Attempting to close LunarBaseEnv instance...")
            LUNAR_ENV_INSTANCE.close()
            print("LunarBaseEnv instance closed.")
