import gymnasium as gym
import os

from . import agents, move_ik_rel_env_cfg, move_ik_rel_env_cfg_rgb, move_ik_rel_env_cfg_rgb_nt

# 逆运动学 + 相对位姿控制
print("Registering UR5 Move Box IK Rel Env")
gym.register(
    id="Isaac-Move-Box-UR5-IK-Rel-20250707",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # 提供的配置是类
        "env_cfg_entry_point": move_ik_rel_env_cfg.UR5BoxMoveEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Move-Box-UR5-IK-Rel-RGB-20250707",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # 提供的配置是类
        "env_cfg_entry_point": move_ik_rel_env_cfg_rgb.UR5BoxMoveEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Move-Box-UR5-IK-Rel-RGB-20250707-disable_termination",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # 提供的配置是类
        "env_cfg_entry_point": move_ik_rel_env_cfg_rgb_nt.UR5BoxMoveEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)
