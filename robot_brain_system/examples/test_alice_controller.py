from ..utils.config_utils import load_config
from ..core.isaac_simulator import IsaacSimulator

if __name__ == "__main__":
    isaac_simulator = IsaacSimulator(
        sim_config=load_config(
            config_path="../conf", config_name="config", return_dict=True
        )["simulator"]
    )
    isaac_simulator.initialize()
