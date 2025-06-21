import gymnasium as gym
import os

from . import (
    move_ik_rel_env_cfg
)

# 逆运动学 + 相对位姿控制
print("Registering Frank Move Box IK Rel Env")
gym.register(
    id="Isaac-Move-Box-Frank-IK-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # 提供的配置是类
        "env_cfg_entry_point": move_ik_rel_env_cfg.FrankBoxMoveEnvCfg,
    },
    disable_env_checker=True,
)