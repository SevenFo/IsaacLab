import hydra
from omegaconf import DictConfig

from robot_brain_system.core.isaac_simulator import IsaacSimulator

if __name__ == "__main__":

    @hydra.main(
        version_base=None,
        config_path="../robot_brain_system/conf",
        config_name="config",
    )
    def main(cfg: DictConfig):
        isim = IsaacSimulator(sim_config=cfg.simulator)
        result = isim.initialize()
        assert result, "Failed to initialize Isaac Simulator"
        print("Isaac Simulator initialized successfully.")

        isim.start_skill_non_blocking(
            "object_tracking",
            {
                "target_object": "red box",
            },
            timeout=1000,  # Set a reasonable timeout for the skill to start
        )
        isim.start_skill_non_blocking(
            "move_to_target_object",
            {
                "target_object": "red box",
            },
            timeout=1000,  # Set a reasonable timeout for the skill to start
        )
        for i in range(1000):
            import time

            time.sleep(1)

        isim.shutdown()

    main()
