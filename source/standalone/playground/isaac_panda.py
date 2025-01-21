# -*- coding: utf-8 -*-
# Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2023 PickNik, LLC. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import re
import os

import carb
import numpy as np
from pathlib import Path

# In older versions of Isaac Sim (prior to 4.0), SimulationApp is imported from
# omni.isaac.kit rather than isaacsim.
try:
    from isaacsim import SimulationApp
except:
    from omni.isaac.kit import SimulationApp

FRANKA_STAGE_PATH = "/Franka"
FRANKA_USD_PATH = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
REALSENSE_CAMERA_USD_PATH = "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
CAMERA_PRIM_PATH = f"{FRANKA_STAGE_PATH}/panda_hand/geometry/realsense/realsense_camera"
FRONT_CAMERA_PRIM_PATH = f"{FRANKA_STAGE_PATH}/fixed_cameras/front_camera"
WRIST_CAMERA_PRIM_PATH = f"{FRANKA_STAGE_PATH}/panda_hand/geometry/realsense/wrist_camera"
BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"
GRAPH_PATH = "/ActionGraph"
REALSENSE_VIEWPORT_NAME = "realsense_viewport"

CONFIG = {"renderer": "RayTracedLighting", "headless": False,"max_gpu_count":1}

simulation_app = SimulationApp(CONFIG)

from omni.isaac.version import get_version

# Check the major version number of Isaac Sim to see if it's four digits, corresponding
# to Isaac Sim 2023.1.1 or older.  The version numbering scheme changed with the
# Isaac Sim 4.0 release in 2024.
is_legacy_isaacsim = len(get_version()[2]) == 4

# More imports that need to compare after we create the app
from omni.isaac.core import SimulationContext  # noqa E402
from omni.isaac.core.utils.prims import set_targets
from omni.isaac.core.utils import (  # noqa E402
    extensions,
    nucleus,
    prims,
    rotations,
    stage,
    viewports,
)
from omni.isaac.core_nodes.scripts.utils import set_target_prims  # noqa E402
from omni.isaac.sensor import Camera
from pxr import Gf, UsdGeom  # noqa E402
import omni.graph.core as og  # noqa E402
import omni

# enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")

simulation_context = SimulationContext(stage_units_in_meters=1.0)

# Locate Isaac Sim assets folder to load environment and robot stages
assets_root_path = nucleus.get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

# Preparing stage
viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

# Loading the simple_room environment
stage.add_reference_to_stage(
    assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH
)

# Loading the franka robot USD
prims.create_prim(
    FRANKA_STAGE_PATH,
    "Xform",
    position=np.array([0, -0.64, 0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
    usd_path=assets_root_path + FRANKA_USD_PATH,
)

# add some objects, spread evenly along the X axis
# with a fixed offset from the robot in the Y and Z
prims.create_prim(
    "/cracker_box",
    "Xform",
    position=np.array([-0.2, -0.25, 0.15]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
)
prims.create_prim(
    "/sugar_box",
    "Xform",
    position=np.array([-0.07, -0.25, 0.1]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
)
prims.create_prim(
    "/soup_can",
    "Xform",
    position=np.array([0.1, -0.25, 0.10]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
)
prims.create_prim(
    "/mustard_bottle",
    "Xform",
    position=np.array([0.0, 0.15, 0.12]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
)
front_camera = Camera(
    **{
        "resolution": [640, 640],
        "frequency": None,
        "dt": None,
        "translation": (1.6, 0, 0.6),  # with respect to its parent prim
        "render_product_path": "/default_render_product",
        "prim_path": FRONT_CAMERA_PRIM_PATH,
        "name": "front",
        "position": None,  # with respect to the world frame
        "orientation": (0.5963678, 0.3799282, 0.3799282,0.5963678),  #  (w, x, y, z) # with respect to its parent prim or the world frame dependis if position or traslation is set
    }
)
front_camera.set_focal_length(2.7) # cm
# front_camera.set_focus_distance(400)
front_camera.set_clipping_range(near_distance=0.01)
front_camera.set_local_pose(
    translation=None,
    orientation=(0.5963678, 0.3799282, 0.3799282,0.5963678),
    camera_axes="usd"
)

wrist_camera = Camera(
    **{
        "resolution": [640, 640],
        "frequency": None,
        "dt": None,
        "translation": (0, 0.05, 0.05),  # with respect to its parent prim
        "render_product_path": "/default_render_product",
        "prim_path": WRIST_CAMERA_PRIM_PATH,
        "name": "wrist",
        "position": None,  # with respect to the world frame
        "orientation": (0,0, -1,0),  #  (w, x, y, z) # with respect to its parent prim or the world frame dependis if position or traslation is set
    }
)
wrist_camera.set_focal_length(1.18) # cm
# wrist_camera.set_focus_distance(400)
wrist_camera.set_clipping_range(near_distance=0.01)
wrist_camera.set_local_pose(
    translation=None,
    orientation=(0,0, -1,0),
    camera_axes="usd"
)


simulation_app.update()

try:
    ros_domain_id = int(os.environ["ROS_DOMAIN_ID"])
    print("Using ROS_DOMAIN_ID: ", ros_domain_id)
except ValueError:
    print("Invalid ROS_DOMAIN_ID integer value. Setting value to 0")
    ros_domain_id = 0
except KeyError:
    print("ROS_DOMAIN_ID environment variable is not set. Setting value to 0")
    ros_domain_id = 0

# Create an action graph with ROS component nodes
try:
    og_keys_set_values = [
        ("Context.inputs:domain_id", ros_domain_id),
        # Set the /Franka target prim to Articulation Controller node
        ("ArticulationController.inputs:robotPath", FRANKA_STAGE_PATH),
        ("PublishJointState.inputs:topicName", "isaac_joint_states"),
        ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
        # ("createViewport.inputs:name", REALSENSE_VIEWPORT_NAME),
        # ("createViewport.inputs:viewportId", 1),
        # ("cameraHelperRgb.inputs:frameId", "realsense_camera"),
        # ("cameraHelperRgb.inputs:topicName", "rgb"),
        # ("cameraHelperRgb.inputs:type", "rgb"),
        # ("cameraHelperInfo.inputs:frameId", "realsense_camera"),
        # ("cameraHelperInfo.inputs:topicName", "camera_info"),
        # ("cameraHelperInfo.inputs:type", "camera_info"),
        # ("cameraHelperDepth.inputs:frameId", "realsense_camera"),
        # ("cameraHelperDepth.inputs:topicName", "depth"),
        # ("cameraHelperDepth.inputs:type", "depth_pcl"),
        # front
        # ("createFrontViewport.inputs:name", "front_camera_viewpoint"),
        # ("createFrontViewport.inputs:viewportId", 2),
        ("frontCameraRP.inputs:cameraPrim", FRONT_CAMERA_PRIM_PATH),
        ("frontCameraRP.inputs:height", 640),
        ("frontCameraRP.inputs:width", 640),
        ("FrontcameraHelperRgb.inputs:frameId", "front_camera"),
        ("FrontcameraHelperRgb.inputs:topicName", "front_rgb"),
        ("FrontcameraHelperRgb.inputs:type", "rgb"),
        ("FrontcameraHelperInfo.inputs:frameId", "front_camera"),
        ("FrontcameraHelperInfo.inputs:topicName", "front_camera_info"),
        ("FrontcameraHelperInfo.inputs:type", "camera_info"),
        ("FrontcameraHelperDepth.inputs:frameId", "front_camera"),
        ("FrontcameraHelperDepth.inputs:topicName", "front_depth"),
        ("FrontcameraHelperDepth.inputs:type", "depth_pcl"),
        # ("setFrontCamera.inputs:cameraPrim", FRONT_CAMERA_PRIM_PATH),
        # wrist
        # ("createWristViewport.inputs:name", "wrist_camera_viewpoint"),
        # ("createWristViewport.inputs:viewportId", 2),
        ("wristCameraRP.inputs:cameraPrim", WRIST_CAMERA_PRIM_PATH),
        ("wristCameraRP.inputs:height", 640),
        ("wristCameraRP.inputs:width", 640),
        ("WristcameraHelperRgb.inputs:frameId", "wrist_camera"),
        ("WristcameraHelperRgb.inputs:topicName", "wrist_rgb"),
        ("WristcameraHelperRgb.inputs:type", "rgb"),
        ("WristcameraHelperInfo.inputs:frameId", "wrist_camera"),
        ("WristcameraHelperInfo.inputs:topicName", "wrist_camera_info"),
        ("WristcameraHelperInfo.inputs:type", "camera_info"),
        ("WristcameraHelperDepth.inputs:frameId", "wrist_camera"),
        ("WristcameraHelperDepth.inputs:topicName", "wrist_depth"),
        ("WristcameraHelperDepth.inputs:type", "depth_pcl"),
        # ("setWristCamera.inputs:cameraPrim", WRIST_CAMERA_PRIM_PATH),
        # tf
        ("TFPuber.inputs:parentPrim", FRANKA_STAGE_PATH),
        ("TFPuber.inputs:targetPrims", [FRONT_CAMERA_PRIM_PATH, WRIST_CAMERA_PRIM_PATH]),
        ("StaticTFPuber.inputs:parentFrameId","world"),
        ("StaticTFPuber.inputs:childFrameId","Franka"),
    ]

    # In older versions of Isaac Sim, the articulation controller node contained a
    # "usePath" checkbox input that should be enabled.
    if is_legacy_isaacsim:
        og_keys_set_values.insert(1, ("ArticulationController.inputs:usePath", True))

    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                (
                    "SubscribeJointState",
                    "omni.isaac.ros2_bridge.ROS2SubscribeJointState",
                ),
                (
                    "ArticulationController",
                    "omni.isaac.core_nodes.IsaacArticulationController",
                ),
                ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                ("OnTick", "omni.graph.action.OnTick"),
                # ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                # ("createFrontViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                # ("createWristViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                # (
                #     "getRenderProduct",
                #     "omni.isaac.core_nodes.IsaacGetViewportRenderProduct",
                # ),
                # (
                #     "getFrontRenderProduct",
                #     "omni.isaac.core_nodes.IsaacGetViewportRenderProduct",
                # ),
                # (
                #     "getWristRenderProduct",
                #     "omni.isaac.core_nodes.IsaacGetViewportRenderProduct",
                # ),
                # ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                # ("setFrontCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                # ("setWristCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                ("frontCameraRP","omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                ("wristCameraRP","omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                # ("cameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                # ("cameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                # ("cameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("FrontcameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("FrontcameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("FrontcameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("WristcameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("WristcameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("WristcameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("TFPuber", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                ("StaticTFPuber", "omni.isaac.ros2_bridge.ROS2PublishRawTransformTree"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                (
                    "OnImpulseEvent.outputs:execOut",
                    "ArticulationController.inputs:execIn",
                ),
                (
                    "OnImpulseEvent.outputs:execOut",
                    "TFPuber.inputs:execIn",
                ),
                (
                    "OnImpulseEvent.outputs:execOut",
                    "StaticTFPuber.inputs:execIn",
                ),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "TFPuber.inputs:context"),
                ("Context.outputs:context", "StaticTFPuber.inputs:context"),
                (
                    "ReadSimTime.outputs:simulationTime",
                    "PublishJointState.inputs:timeStamp",
                ),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "TFPuber.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "StaticTFPuber.inputs:timeStamp"),
                (
                    "SubscribeJointState.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "SubscribeJointState.outputs:velocityCommand",
                    "ArticulationController.inputs:velocityCommand",
                ),
                (
                    "SubscribeJointState.outputs:effortCommand",
                    "ArticulationController.inputs:effortCommand",
                ),
                # ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                # ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                # ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                # ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                # (
                #     "getRenderProduct.outputs:renderProductPath",
                #     "setCamera.inputs:renderProductPath",
                # ),
                # ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                # ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                # ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                # ("Context.outputs:context", "cameraHelperRgb.inputs:context"),
                # ("Context.outputs:context", "cameraHelperInfo.inputs:context"),
                # ("Context.outputs:context", "cameraHelperDepth.inputs:context"),
                # (
                #     "getRenderProduct.outputs:renderProductPath",
                #     "cameraHelperRgb.inputs:renderProductPath",
                # ),
                # (
                #     "getRenderProduct.outputs:renderProductPath",
                #     "cameraHelperInfo.inputs:renderProductPath",
                # ),
                # (
                #     "getRenderProduct.outputs:renderProductPath",
                #     "cameraHelperDepth.inputs:renderProductPath",
                # ),
                # front
                ("OnTick.outputs:tick", "frontCameraRP.inputs:execIn"), # front
                ("frontCameraRP.outputs:execOut", "FrontcameraHelperRgb.inputs:execIn"),
                ("frontCameraRP.outputs:execOut", "FrontcameraHelperInfo.inputs:execIn"),
                ("frontCameraRP.outputs:execOut", "FrontcameraHelperDepth.inputs:execIn"),
                ("Context.outputs:context", "FrontcameraHelperRgb.inputs:context"),
                ("Context.outputs:context", "FrontcameraHelperInfo.inputs:context"),
                ("Context.outputs:context", "FrontcameraHelperDepth.inputs:context"),
                (
                    "frontCameraRP.outputs:renderProductPath",
                    "FrontcameraHelperRgb.inputs:renderProductPath",
                ),
                (
                    "frontCameraRP.outputs:renderProductPath",
                    "FrontcameraHelperInfo.inputs:renderProductPath",
                ),
                (
                    "frontCameraRP.outputs:renderProductPath",
                    "FrontcameraHelperDepth.inputs:renderProductPath",
                ),
                # wrist
                ("OnTick.outputs:tick", "wristCameraRP.inputs:execIn"), # wrist
                ("wristCameraRP.outputs:execOut", "WristcameraHelperRgb.inputs:execIn"),
                ("wristCameraRP.outputs:execOut", "WristcameraHelperInfo.inputs:execIn"),
                ("wristCameraRP.outputs:execOut", "WristcameraHelperDepth.inputs:execIn"),
                ("Context.outputs:context", "WristcameraHelperRgb.inputs:context"),
                ("Context.outputs:context", "WristcameraHelperInfo.inputs:context"),
                ("Context.outputs:context", "WristcameraHelperDepth.inputs:context"),
                (
                    "wristCameraRP.outputs:renderProductPath",
                    "WristcameraHelperRgb.inputs:renderProductPath",
                ),
                (
                    "wristCameraRP.outputs:renderProductPath",
                    "WristcameraHelperInfo.inputs:renderProductPath",
                ),
                (
                    "wristCameraRP.outputs:renderProductPath",
                    "WristcameraHelperDepth.inputs:renderProductPath",
                ),
            ],
            og.Controller.Keys.SET_VALUES: og_keys_set_values,
        },
    )
except Exception as e:
    print(e)


# Setting the /Franka target prim to Publish JointState node
set_target_prims(
    primPath="/ActionGraph/PublishJointState", targetPrimPaths=[FRANKA_STAGE_PATH]
)

# Fix camera settings since the defaults in the realsense model are inaccurate
# realsense_prim = camera_prim = UsdGeom.Camera(
#     stage.get_current_stage().GetPrimAtPath(CAMERA_PRIM_PATH)
# )
# realsense_prim.GetHorizontalApertureAttr().Set(20.955)
# realsense_prim.GetVerticalApertureAttr().Set(15.7)
# realsense_prim.GetFocalLengthAttr().Set(18.8)
# realsense_prim.GetFocusDistanceAttr().Set(400)

# set_targets(
#     prim=stage.get_current_stage().GetPrimAtPath(GRAPH_PATH + "/setCamera"),
#     attribute="inputs:cameraPrim",
#     target_prim_paths=[CAMERA_PRIM_PATH],
# )
front_camera.set_clipping_range(near_distance=0.01, far_distance=2)

simulation_app.update()

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()

simulation_context.play()

# Dock the second camera window
# viewport = omni.ui.Workspace.get_window("Viewport")
# viewport = omni.ui.Workspace.get_window(REALSENSE_VIEWPORT_NAME)
# viewport.dock_in(viewport, omni.ui.DockPosition.RIGHT)

simulation_context.stop()
simulation_context.play()

while simulation_app.is_running():

    # Run with a fixed step size
    simulation_context.step(render=True)

    # Tick the Publish/Subscribe JointState, Publish TF and Publish Clock nodes each frame
    og.Controller.set(
        og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True
    )

simulation_context.stop()
simulation_app.close()