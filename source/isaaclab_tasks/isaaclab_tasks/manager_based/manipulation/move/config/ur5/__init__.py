import gymnasium as gym
import os

from . import (
    agents,
    move_ik_rel_env_cfg
)

# 逆运动学 + 相对位姿控制
print("Registering UR5 Move Box IK Rel Env")
gym.register(
    id="Isaac-Move-Box-UR5-IK-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # 提供的配置是类
        "env_cfg_entry_point": move_ik_rel_env_cfg.UR5BoxMoveEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)