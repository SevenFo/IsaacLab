# ik_cam_env_cfg.py

from dataclasses import field
from typing import Literal
import numpy as np

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim import PinholeCameraCfg
from omni.isaac.lab.sensors import TiledCameraCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg

from . import ik_abs_env_cfg
from . import ik_rel_env_cfg
import omni.isaac.lab_tasks.manager_based.manipulation.lift.mdp as mdp
from ...lift_env_cfg import ObjectTableSceneCfg, ObservationsCfg
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class ObjectTableRGBSceneCfg(ObjectTableSceneCfg):
    """场景配置，添加RGB相机"""

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.6, 0.0, 0.6),
            rot=(0.5963678, 0.3799282, 0.3799282, 0.5963678),
            convention="opengl",
        ),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=480,
        height=480,
    )


@configclass
class ObservationWithRGBCfg(ObservationsCfg):
    """观察配置，添加RGB图像"""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    rgb: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()


@configclass
class FrankaCubeLiftAbsCameraEnvCfg(ik_abs_env_cfg.FrankaCubeLiftEnvCfg):
    """绝对姿态控制的相机环境配置"""

    scene = ObjectTableRGBSceneCfg(num_envs=1, env_spacing=2.5)
    observations = ObservationWithRGBCfg()

    def __post_init__(self):
        super().__post_init__()
        # pos_offset = (0.62562, -1.29608, 2.95021)
        # rot_offset = (0.707, 0.0, 0, -0.707)  # z轴旋转-90度
        # self.scene.object.init_state.pos = tuple(
        #     np.array(self.scene.object.init_state.pos) + np.array(pos_offset)
        # )
        # self.scene.robot.init_state.pos = tuple(
        #     np.array(self.scene.robot.init_state.pos) + np.array(pos_offset)
        # )
        # object_pos_offset_x = (
        #     self.scene.object.init_state.pos[0] - self.scene.robot.init_state.pos[0]
        # )
        # self.scene.object.init_state.pos = tuple(
        #     np.array(self.scene.object.init_state.pos)
        #     + np.array([-object_pos_offset_x, -object_pos_offset_x, 0])
        #     + np.array([-0.2, 0.0, 0.0])
        # )
        # self.scene.robot.init_state.rot = rot_offset
        # self.scene.tiled_camera.offset.rot = tuple(
        #     np.array(self.scene.tiled_camera.offset.rot) + np.array(rot_offset)
        # )


@configclass
class FrankaCubeLiftRelCameraEnvCfg(ik_rel_env_cfg.FrankaCubeLiftEnvCfg):
    """绝对姿态控制的相机环境配置"""

    scene = ObjectTableRGBSceneCfg(num_envs=1, env_spacing=2.5)
    observations = ObservationWithRGBCfg()

    def __post_init__(self):
        super().__post_init__()
