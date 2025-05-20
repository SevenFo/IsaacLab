import requests
import pickle
import numpy as np
import json  # For printing JSON responses nicely

BASE_URL = "http://127.0.0.1:18861"


def print_response(response: requests.Response):
    """Helper function to print response status and content."""
    print(f"Status Code: {response.status_code}")
    try:
        # Try to print as JSON if possible
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        elif "application/python-pickle" in content_type:
            print("Response (Pickled, attempting to unpickle):")
            try:
                data = pickle.loads(response.content)
                print(data)
            except pickle.UnpicklingError as e:
                print(f"  Could not unpickle: {e}")
                print(
                    f"  Raw content: {response.content[:200]}..."
                )  # Print first 200 bytes
        else:
            print("Response Text:")
            print(response.text)
    except requests.exceptions.JSONDecodeError:
        print("Response (Not JSON):")
        print(response.text)
    print("-" * 30)


def reset_env():
    print("Testing /reset...")
    try:
        response = requests.post(f"{BASE_URL}/reset", timeout=10)  # Added timeout
        print_response(response)
        if response.ok:
            return response.json()  # Return obs, info
    except requests.exceptions.RequestException as e:
        print(f"Request failed for /reset: {e}")
    return None


def step_env(action_dict):
    print("Testing /step...")
    try:
        serialized_action = pickle.dumps(action_dict)
        response = requests.post(
            f"{BASE_URL}/step",
            data=serialized_action,
            headers={"Content-Type": "application/python-pickle"},
            timeout=10,  # Added timeout
        )
        # Step response is pickled
        if response.ok:
            try:
                data = pickle.loads(response.content)
                print("Unpickled step response:")
                # Assuming data is a dict like {'obs': ..., 'reward': ..., ...}
                # Print keys and shapes for numpy arrays
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: {value}")
                print("-" * 30)
                return data
            except pickle.UnpicklingError:
                print("Failed to unpickle step response.")
                print_response(response)  # Print raw if unpickle fails
        else:
            print_response(response)

    except requests.exceptions.RequestException as e:
        print(f"Request failed for /step: {e}")
    return None


def set_simulation_mode(mode_value: int):
    """
    Sets the simulation mode.
    mode_value: 1 for MANUAL_STEP, 2 for AUTO_STEP
    """
    print(f"Testing /set_mode with mode_value: {mode_value}...")
    payload = {"mode": mode_value}
    try:
        response = requests.post(
            f"{BASE_URL}/set_mode", json=payload, timeout=20
        )  # Increased timeout
        print_response(response)
        if response.ok:
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed for /set_mode: {e}")
    return None


def get_sample_action(num_envs=1):
    """Generates a sample action dictionary."""
    # Assuming the action space is defined as in your comment:
    # "end_effector": Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    # "gripper": Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    # The server side `isaac_process.py` expects actions for `num_envs`
    # So the shape for end_effector should be (num_envs, 6) and gripper (num_envs, 2)

    action = {
        "end_effector": np.random.uniform(
            low=-0.2, high=0.2, size=(num_envs, 6)
        ).astype(np.float32),
        "gripper": np.random.uniform(low=0.0, high=1.0, size=(num_envs, 2)).astype(
            np.float32
        ),
    }
    # Example: Keep rotation part of end_effector small or zero if preferred for initial tests
    action["end_effector"][:, 3:] = 0.0  # No rotation initially
    # Example: Small forward movement
    action["end_effector"][:, 0] = 0.05  # Move a bit in +x
    return action


if __name__ == "__main__":
    # --- Initial Reset (usually needed first) ---
    print("--- Performing Initial Reset ---")
    reset_data = (
        reset_env()
    )  # reset_data is a dict: {"obs": list_of_lists_or_numbers, "info": ...}
    current_obs_from_server = None  # This will store the actual observation data

    if reset_data and "obs" in reset_data:
        current_obs_from_server = reset_data["obs"]
        # Convert the list back to a NumPy array for consistent handling in the client
        # (optional, but good if you plan to do numpy operations on it)
        current_obs_np = np.array(current_obs_from_server)
        print(
            f"Initial observation received (shape on client after np.array): {current_obs_np.shape}"
        )
        print(f"Initial info: {reset_data.get('info')}")
    else:
        print("Initial reset failed or 'obs' not in response. Exiting client test.")
        exit()

    # --- Test Mode Switching ---
    print("\n--- Testing Mode Switching ---")
    set_simulation_mode(2)  # Switch to AUTO_STEP
    print(
        "Switched to AUTO_STEP. If you want to observe server, add a time.sleep here in client."
    )
    # import time
    # time.sleep(5)
    set_simulation_mode(1)  # Switch back to MANUAL_STEP
    print("Switched back to MANUAL_STEP.")

    # --- Interactive Loop for Stepping ---
    NUM_ENVS_ON_SERVER = 1  # Adjust this if your server's --num_envs is different

    print("\n--- Interactive Stepping (MANUAL_STEP mode assumed) ---")
    print(
        "Enter 's' to step, 'r' to reset, 'm1' for manual mode, 'm2' for auto mode, 'q' to quit."
    )

    while True:
        try:
            command = input("Client command (s, r, m1, m2, q): ").strip().lower()

            if command == "q":
                print("Exiting client.")
                break
            elif command == "r":
                print("--- Resetting Environment ---")
                reset_data = reset_env()
                if reset_data and "obs" in reset_data:
                    current_obs_from_server = reset_data["obs"]
                    current_obs_np = np.array(current_obs_from_server)  # Update obs
                    print(
                        f"Reset observation received (shape on client): {current_obs_np.shape}"
                    )
                else:
                    print("Reset failed or 'obs' not in response.")
            elif command == "s":
                if current_obs_from_server is None:  # Check if we have an observation
                    print("No current observation. Please reset first ('r').")
                    continue
                print("--- Performing a Step ---")
                action_to_send = get_sample_action(num_envs=NUM_ENVS_ON_SERVER)
                print(f"Sending action: {action_to_send}")
                step_result = step_env(
                    action_to_send
                )  # step_result is a dict from pickle.loads
                if step_result and "obs" in step_result:
                    # The 'obs' from step_result is already a NumPy array because
                    # the server sends the pickled dictionary which contains NumPy arrays.
                    current_obs_np = step_result["obs"]
                    current_obs_from_server = (
                        current_obs_np.tolist()
                    )  # if you want to keep it as list for consistency
                    print(
                        f"  Step observation received (shape on client): {current_obs_np.shape}"
                    )
                    print(f"  Reward: {step_result.get('reward')}")
                    print(f"  Done: {step_result.get('done')}")
                elif step_result:
                    print(
                        "Step executed but 'obs' not in pickled response or step_result is None."
                    )
                else:
                    print("Step failed to return a result.")

            elif command == "m1":
                print("--- Setting Mode to MANUAL_STEP (1) ---")
                set_simulation_mode(1)
            elif command == "m2":
                print("--- Setting Mode to AUTO_STEP (2) ---")
                set_simulation_mode(2)
            else:
                print(
                    "Unknown command. Options: 's' (step), 'r' (reset), 'm1' (manual), 'm2' (auto), 'q' (quit)"
                )
        except EOFError:
            print("\nExiting client (EOF).")
            break
        except KeyboardInterrupt:
            print("\nExiting client (KeyboardInterrupt).")
            break
        except Exception as e:
            print(f"An error occurred in the client loop: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging client-side errors
