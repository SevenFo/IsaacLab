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
ALICE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/data/shared_folder/IssacAsserts/Projects/PN_Stickman_v12_ThumbInward_joint_link.usd",
        # activate_contact_sensors=True,
        scale=(0.01, 0.01, 0.01),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            max_linear_velocity=0,
            max_angular_velocity=0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    ),
    # 所有joint的Damping=90，Stiffness=45
    # /World/Noitom_Hips/Noitom_Spine/Noitom_Spine1/Noitom_Spine2/Noitom_RightShoulder/Noitom_RightArm/SM_RightArm/SM_RightArm/D6Joint：
    # （0，66.7，50.7）
    # /World/Noitom_Hips/Noitom_Spine/Noitom_Spine1/Noitom_Spine2/Noitom_RightShoulder/Noitom_RightArm/SM_RightArm/Noitom_RightForeArm/SM_RightForeArm/SM_RightForeArm/D6Joint：
    # （0，25.9，-23.2)
    # /World/Noitom_Hips/Noitom_Spine/Noitom_Spine1/Noitom_Spine2/Noitom_RightShoulder/Noitom_RightArm/SM_RightArm/Noitom_RightForeArm/Noitom_RightHand/SM_RightHand/SM_RightHand/D6Joint：
    # （-141.8， -11.0，-41.7）
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, -3.1, 1.6),
        pos=(-1.8, -2.5, 2.8),
        # pos=(0.4067, 10, 1.7),
        rot=(0.5, 0.5, 0.5, 0.5),
        # joint_pos={
        #     "D6Joint_1:0": math.radians(0),
        #     "D6Joint_1:1": math.radians(66.7),
        #     "D6Joint_1:2": math.radians(50.7),
        #     "D6Joint_2:0": math.radians(0),
        #     "D6Joint_2:1": math.radians(25.9),
        #     "D6Joint_2:2": math.radians(-23.2),
        #     "D6Joint_3:0": math.radians(-141.8),
        #     "D6Joint_3:1": math.radians(-11.0),
        #     "D6Joint_3:2": math.radians(-41.7),
        #     "RevoluteJoint": math.radians(0.0),
        # },
    ),
    # dict_keys(['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg',
    # 'LeftFoot', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 'Head',
    # 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1',
    # 'RightHandThumb2', 'RightHandThumb3', 'RightInHandIndex', 'RightHandIndex1',
    # 'RightHandIndex2', 'RightHandIndex3', 'RightInHandMiddle', 'RightHandMiddle1',
    # 'RightHandMiddle2', 'RightHandMiddle3', 'RightInHandRing', 'RightHandRing1',
    # 'RightHandRing2', 'RightHandRing3', 'RightInHandPinky', 'RightHandPinky1',
    # 'RightHandPinky2', 'RightHandPinky3', 'LeftShoulder', 'LeftArm', 'LeftForeArm',
    # 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3',
    # 'LeftInHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
    # 'LeftInHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',
    # 'LeftInHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftInHandPinky',
    # 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3'])
    # """
    actuators={
        "D6Joints": ImplicitActuatorCfg(
            joint_names_expr=["D6Joint_.*"],
            effort_limit=100.0,
            velocity_limit=5.0,
            stiffness=0,
            damping=0,
        ),
        "RevoluteJoints": ImplicitActuatorCfg(
            joint_names_expr=["RevoluteJoint_.*"],
            # effort_limit=100.0,
            # velocity_limit=5.0,
            stiffness=0,
            damping=0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
