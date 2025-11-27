# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import numpy as np

##
# Configuration
##


UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""

UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/data/shared_folder/gripper/ur5_with_robotiq_gripper/ur5_with_gripper_drive_fix.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.24, -4.000, 2.830),
        rot=(1, 0, 0, 0),
        joint_pos={
            "shoulder_pan_joint": np.deg2rad(90.0).item(),
            "shoulder_lift_joint": np.deg2rad(-106.0).item(),
            "elbow_joint": np.deg2rad(40.0).item(),
            "wrist_1_joint": np.deg2rad(-60.0).item(),
            "wrist_2_joint": np.deg2rad(-90.0).item(),
            "wrist_3_joint": np.deg2rad(90.0).item(),
            "finger_joint_.*": np.deg2rad(0.0).item(),
            "left_inner_finger_joint": np.deg2rad(-45.0).item(),
            "right_inner_finger_joint": np.deg2rad(-45.0).item(),
        },
    ),
    actuators={
        "shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=["shoulder.*", "elbow.*"],
            stiffness=1008,
            # damping=20,
            damping=100,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist.*"],
            stiffness=380,
            # damping=10,
            damping=40,
        ),
        "finger_joint": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint.*"],
            stiffness=380,  # 0.45,
            damping=0.0002 * 1000,
        ),
        "inner_finger_joint": ImplicitActuatorCfg(
            joint_names_expr=["left_inner_finger_joint", "right_inner_finger_joint"],
            stiffness=380,  # 0.02,
            damping=0.00001 * 100,
        ),
        "passive_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_outer_finger_joint",
                "left_outer_finger_joint",
                "right_inner_finger_knuckle_joint",
                "left_inner_finger_knuckle_joint",
            ],
            stiffness=0,
            damping=0,
        ),
    },
)
"""Configuration of UR-5 arm using implicit actuator models."""
