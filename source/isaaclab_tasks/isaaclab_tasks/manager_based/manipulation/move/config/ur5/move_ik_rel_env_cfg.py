from isaaclab.controllers.differential_ik_cfg import (
    DifferentialIKControllerCfg,
)
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.utils import configclass

from . import move_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import (
    FRANKA_PANDA_HIGH_PD_CFG,
    FRANKA_PANDA_CFG,
)  # isort: skip


@configclass
class UR5BoxMoveEnvCfg(move_joint_pos_env_cfg.UR5BoxMoveEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set UR5 as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        # self.scene.robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # # Set actions for the specific robot type (ur5)
        # self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        #     asset_name="robot",
        #     joint_names=["shoulder.*", "elbow.*", "wrist.*"],
        #     body_name="base_link_gripper",
        #     controller=DifferentialIKControllerCfg(
        #         command_type="pose", use_relative_mode=True, ik_method="dls"
        #     ),
        #     scale=0.5,
        #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
        #         pos=[0.0, 0.0, 0.0]
        #     ),
        # )

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107]
            ),
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                # gravity_compensation=False,
                # load_mass=2.0,
                # gravity_gain=0.1,
            ),
        )
