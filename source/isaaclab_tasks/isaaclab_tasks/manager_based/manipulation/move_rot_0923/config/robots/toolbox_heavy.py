# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

HEAVY_TOOLBOX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/data/shared_folder/IssacAsserts/Projects/Box/Collected_red_toolbox/red_toolbox.usd",  # "/data/shared_folder/IssacAsserts/Projects/Box/Collected_red_toolbox/red_toolbox.usd",
        scale=(1.2, 1.2, 1.2),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.215 * 10, -3.45 * 10, 2.9),
        rot=(1, 0, 0, 0),
        joint_pos={
            # "boxjoint": 0.0,
            "boxjoint": 0.33,
        },
    ),
    actuators={
        "box_lip": ImplicitActuatorCfg(
            joint_names_expr=["boxjoint"],
            effort_limit=100.0,
            velocity_limit=5.0,
            stiffness=1000.0,
            damping=52.4,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
