# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Cabinet-Direct-v0",
    entry_point=f"{__name__}.franka_cabinet_env:FrankaCabinetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cabinet_env:FrankaCabinetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-UR5-LunarBase-Direct-v0",
    entry_point=f"{__name__}.ur5_lunar_base_env:LunarBaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_lunar_base_env:LunarBaseEnvCfg",
    },
)
gym.register(
    id="Isaac-Frank-LunarBase-Direct-v0",
    entry_point=f"{__name__}.frank_lunar_base_env:FrankLunarBaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.frank_lunar_base_env:FrankLunarBaseEnvCfg",
    },
)
