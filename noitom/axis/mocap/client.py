from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Usd, UsdGeom, Gf
from noitom.axis.mocap.base_stage_async import BaseStageAsync
import noitom.axis.mocap.mocap_api as mocap_api
import omni.kit.commands
import threading
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


class MocapClient(BaseStageAsync):
    def __init__(self, udp_port=8003, prim_prefix=None) -> None:
        super().__init__()
        self.prim_prefix = prim_prefix
        self.application = mocap_api.MCPApplication()
        settings = mocap_api.MCPSettings()
        settings.set_udp(udp_port)
        self.application.set_settings(settings)
        del settings
        renderSettings = mocap_api.MCPRenderSettings()
        self.application.set_render_settings(renderSettings)
        del renderSettings

        self._world: World = World.instance()
        self.relative_trans = {}

        self.data_updated = False
        self.lock = threading.Lock()  # 用于保护avatar obj
        self.init_bones()
        # self.quat_visualizer = MultithreadVisualizer()

        return

    def init_bones(self):
        # 骨骼名称列表
        self.bone_names = [
            "Hips",
            "RightUpLeg",
            "RightLeg",
            "RightFoot",
            "LeftUpLeg",
            "LeftLeg",
            "LeftFoot",
            "Spine",
            "Spine1",
            "Spine2",
            "Neck",
            "Neck1",
            "Head",
            "RightShoulder",
            "RightArm",
            "RightForeArm",
            "RightHand",
            "RightHandThumb1",
            "RightHandThumb2",
            "RightHandThumb3",
            "RightHandIndex",
            "RightHandIndex1",
            "RightHandIndex2",
            "RightHandIndex3",
            "RightHandMiddle",
            "RightHandMiddle1",
            "RightHandMiddle2",
            "RightHandMiddle3",
            "RightHandRing",
            "RightHandRing1",
            "RightHandRing2",
            "RightHandRing3",
            "RightHandPinky",
            "RightHandPinky1",
            "RightHandPinky2",
            "RightHandPinky3",
            "LeftShoulder",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "LeftHandThumb1",
            "LeftHandThumb2",
            "LeftHandThumb3",
            "LeftHandIndex",
            "LeftHandIndex1",
            "LeftHandIndex2",
            "LeftHandIndex3",
            "LeftHandMiddle",
            "LeftHandMiddle1",
            "LeftHandMiddle2",
            "LeftHandMiddle3",
            "LeftHandRing",
            "LeftHandRing1",
            "LeftHandRing2",
            "LeftHandRing3",
            "LeftHandPinky",
            "LeftHandPinky1",
            "LeftHandPinky2",
            "LeftHandPinky3",
        ]

        # 创建avatar字典
        self.avatar = {
            name: {
                "px": 0,
                "py": 0,
                "pz": 0,
                "rx": 0,
                "ry": 0,
                "rz": 0,
                "rw": 1,
                "world_px": 0,
                "world_py": 0,
                "world_pz": 0,
                "world_rx": 0,
                "world_ry": 0,
                "world_rz": 0,
                "world_rw": 1,
            }
            for name in self.bone_names
        }

    def get_bone_name_dic(self):
        return self.bone_names

    def setup_scene(self, world=None):
        world = World.instance() if world is None else world
        # world.scene.add_default_ground_plane()
        context = omni.usd.get_context()
        stage = context.get_stage()
        if not stage.GetPrimAtPath("/World"):
            raise RuntimeError(f"World prim at '/World' does not exist.")

        # 设置Stage的单位为米
        # stage.SetMetadata("metersPerUnit", 0.01)  # 1厘米 = 0.01米
        # x 0.45417 y -7.59708 z 0.00000
        # x 0.00000 y 0.00000 z 97.00000

        script_dir = os.path.dirname(os.path.abspath(__file__))
        omni.kit.commands.execute(
            "CreatePayload",
            usd_context=omni.usd.get_context(),
            path_to="/World/PN_Stickman_v12_ThumbInward",
            asset_path=script_dir
            + "\\avatar\\PN_Stickman_v12_ThumbInward_RightHandPM.usd",
            instanceable=False,
        )

        # 缩放模型从厘米到米
        root_prim = stage.GetPrimAtPath("/World/PN_Stickman_v12_ThumbInward")
        UsdGeom.XformCommonAPI(root_prim).SetScale((0.01, 0.01, 0.01))  # 缩小100倍

        root_p = "/World/PN_Stickman_v12_ThumbInward" + (
            f"/{self.prim_prefix}" if self.prim_prefix else ""
        )
        world.scene.add(XFormPrim(prim_path=root_p, name="PN_Stickman"))

        hip_p = root_p + "/Noitom_Hips"
        world.scene.add(XFormPrim(prim_path=hip_p, name="Hips"))
        left_up_leg_p = hip_p + "/Noitom_LeftUpLeg"
        world.scene.add(XFormPrim(prim_path=left_up_leg_p, name="LeftUpLeg"))
        left_leg_p = left_up_leg_p + "/Noitom_LeftLeg"
        world.scene.add(XFormPrim(prim_path=left_leg_p, name="LeftLeg"))
        world.scene.add(
            XFormPrim(prim_path=left_leg_p + "/Noitom_LeftFoot", name="LeftFoot")
        )
        right_up_leg_p = hip_p + "/Noitom_RightUpLeg"
        world.scene.add(XFormPrim(prim_path=right_up_leg_p, name="RightUpLeg"))
        right_leg_p = right_up_leg_p + "/Noitom_RightLeg"
        world.scene.add(XFormPrim(prim_path=right_leg_p, name="RightLeg"))
        world.scene.add(
            XFormPrim(prim_path=right_leg_p + "/Noitom_RightFoot", name="RightFoot")
        )

        spine_p = hip_p + "/Noitom_Spine"
        world.scene.add(XFormPrim(prim_path=spine_p, name="Spine"))
        spine1_p = spine_p + "/Noitom_Spine1"
        world.scene.add(XFormPrim(prim_path=spine1_p, name="Spine1"))
        spine2_p = spine1_p + "/Noitom_Spine2"
        world.scene.add(XFormPrim(prim_path=spine2_p, name="Spine2"))
        neck_p = spine2_p + "/Noitom_Neck"
        world.scene.add(XFormPrim(prim_path=neck_p, name="Neck"))
        neck1_p = neck_p + "/Noitom_Neck1"
        world.scene.add(XFormPrim(prim_path=neck1_p, name="Neck1"))
        head_p = neck1_p + "/Noitom_Head"
        world.scene.add(XFormPrim(prim_path=head_p, name="Head"))

        right_shoulder_p = spine2_p + "/Noitom_RightShoulder"
        world.scene.add(XFormPrim(prim_path=right_shoulder_p, name="RightShoulder"))
        right_arm_p = right_shoulder_p + "/Noitom_RightArm"
        world.scene.add(XFormPrim(prim_path=right_arm_p, name="RightArm"))
        right_fore_arm_p = right_arm_p + "/Noitom_RightForeArm"
        world.scene.add(XFormPrim(prim_path=right_fore_arm_p, name="RightForeArm"))
        right_hand_p = right_fore_arm_p + "/Noitom_RightHand"
        world.scene.add(XFormPrim(prim_path=right_hand_p, name="RightHand"))
        right_hand_thumb1_p = right_hand_p + "/Noitom_RightHandThumb1"
        world.scene.add(
            XFormPrim(prim_path=right_hand_thumb1_p, name="RightHandThumb1")
        )
        right_hand_thumb2_p = right_hand_thumb1_p + "/Noitom_RightHandThumb2"
        world.scene.add(
            XFormPrim(prim_path=right_hand_thumb2_p, name="RightHandThumb2")
        )
        right_hand_thumb3_p = right_hand_thumb2_p + "/Noitom_RightHandThumb3"
        world.scene.add(
            XFormPrim(prim_path=right_hand_thumb3_p, name="RightHandThumb3")
        )
        right_hand_index_p = right_hand_p + "/Noitom_RightHandIndex"
        world.scene.add(XFormPrim(prim_path=right_hand_index_p, name="RightHandIndex"))
        right_hand_index1_p = right_hand_index_p + "/Noitom_RightHandIndex1"
        world.scene.add(
            XFormPrim(prim_path=right_hand_index1_p, name="RightHandIndex1")
        )
        right_hand_index2_p = right_hand_index1_p + "/Noitom_RightHandIndex2"
        world.scene.add(
            XFormPrim(prim_path=right_hand_index2_p, name="RightHandIndex2")
        )
        right_hand_index3_p = right_hand_index2_p + "/Noitom_RightHandIndex3"
        world.scene.add(
            XFormPrim(prim_path=right_hand_index3_p, name="RightHandIndex3")
        )
        right_hand_middle_p = right_hand_p + "/Noitom_RightHandMiddle"
        world.scene.add(
            XFormPrim(prim_path=right_hand_middle_p, name="RightHandMiddle")
        )
        right_hand_middle1_p = right_hand_middle_p + "/Noitom_RightHandMiddle1"
        world.scene.add(
            XFormPrim(prim_path=right_hand_middle1_p, name="RightHandMiddle1")
        )
        right_hand_middle2_p = right_hand_middle1_p + "/Noitom_RightHandMiddle2"
        world.scene.add(
            XFormPrim(prim_path=right_hand_middle2_p, name="RightHandMiddle2")
        )
        right_hand_middle3_p = right_hand_middle2_p + "/Noitom_RightHandMiddle3"
        world.scene.add(
            XFormPrim(prim_path=right_hand_middle3_p, name="RightHandMiddle3")
        )
        right_hand_ring_p = right_hand_p + "/Noitom_RightHandRing"
        world.scene.add(XFormPrim(prim_path=right_hand_ring_p, name="RightHandRing"))
        right_hand_ring1_p = right_hand_ring_p + "/Noitom_RightHandRing1"
        world.scene.add(XFormPrim(prim_path=right_hand_ring1_p, name="RightHandRing1"))
        right_hand_ring2_p = right_hand_ring1_p + "/Noitom_RightHandRing2"
        world.scene.add(XFormPrim(prim_path=right_hand_ring2_p, name="RightHandRing2"))
        right_hand_ring3_p = right_hand_ring2_p + "/Noitom_RightHandRing3"
        world.scene.add(XFormPrim(prim_path=right_hand_ring3_p, name="RightHandRing3"))
        right_hand_pinky_p = right_hand_p + "/Noitom_RightHandPinky"
        world.scene.add(XFormPrim(prim_path=right_hand_pinky_p, name="RightHandPinky"))
        right_hand_pinky1_p = right_hand_pinky_p + "/Noitom_RightHandPinky1"
        world.scene.add(
            XFormPrim(prim_path=right_hand_pinky1_p, name="RightHandPinky1")
        )
        right_hand_pinky2_p = right_hand_pinky1_p + "/Noitom_RightHandPinky2"
        world.scene.add(
            XFormPrim(prim_path=right_hand_pinky2_p, name="RightHandPinky2")
        )
        right_hand_pinky3_p = right_hand_pinky2_p + "/Noitom_RightHandPinky3"
        world.scene.add(
            XFormPrim(prim_path=right_hand_pinky3_p, name="RightHandPinky3")
        )

        left_shoulder_p = spine2_p + "/Noitom_LeftShoulder"
        world.scene.add(XFormPrim(prim_path=left_shoulder_p, name="LeftShoulder"))
        left_arm_p = left_shoulder_p + "/Noitom_LeftArm"
        world.scene.add(XFormPrim(prim_path=left_arm_p, name="LeftArm"))
        left_fore_arm_p = left_arm_p + "/Noitom_LeftForeArm"
        world.scene.add(XFormPrim(prim_path=left_fore_arm_p, name="LeftForeArm"))
        left_hand_p = left_fore_arm_p + "/Noitom_LeftHand"
        world.scene.add(XFormPrim(prim_path=left_hand_p, name="LeftHand"))
        left_hand_thumb1_p = left_hand_p + "/Noitom_LeftHandThumb1"
        world.scene.add(XFormPrim(prim_path=left_hand_thumb1_p, name="LeftHandThumb1"))
        left_hand_thumb2_p = left_hand_thumb1_p + "/Noitom_LeftHandThumb2"
        world.scene.add(XFormPrim(prim_path=left_hand_thumb2_p, name="LeftHandThumb2"))
        left_hand_thumb3_p = left_hand_thumb2_p + "/Noitom_LeftHandThumb3"
        world.scene.add(XFormPrim(prim_path=left_hand_thumb3_p, name="LeftHandThumb3"))
        left_hand_index_p = left_hand_p + "/Noitom_LeftHandIndex"
        world.scene.add(XFormPrim(prim_path=left_hand_index_p, name="LeftHandIndex"))
        left_hand_index1_p = left_hand_index_p + "/Noitom_LeftHandIndex1"
        world.scene.add(XFormPrim(prim_path=left_hand_index1_p, name="LeftHandIndex1"))
        left_hand_index2_p = left_hand_index1_p + "/Noitom_LeftHandIndex2"
        world.scene.add(XFormPrim(prim_path=left_hand_index2_p, name="LeftHandIndex2"))
        left_hand_index3_p = left_hand_index2_p + "/Noitom_LeftHandIndex3"
        world.scene.add(XFormPrim(prim_path=left_hand_index3_p, name="LeftHandIndex3"))
        left_hand_middle_p = left_hand_p + "/Noitom_LeftHandMiddle"
        world.scene.add(XFormPrim(prim_path=left_hand_middle_p, name="LeftHandMiddle"))
        left_hand_middle1_p = left_hand_middle_p + "/Noitom_LeftHandMiddle1"
        world.scene.add(
            XFormPrim(prim_path=left_hand_middle1_p, name="LeftHandMiddle1")
        )
        left_hand_middle2_p = left_hand_middle1_p + "/Noitom_LeftHandMiddle2"
        world.scene.add(
            XFormPrim(prim_path=left_hand_middle2_p, name="LeftHandMiddle2")
        )
        left_hand_middle3_p = left_hand_middle2_p + "/Noitom_LeftHandMiddle3"
        world.scene.add(
            XFormPrim(prim_path=left_hand_middle3_p, name="LeftHandMiddle3")
        )
        left_hand_ring_p = left_hand_p + "/Noitom_LeftHandRing"
        world.scene.add(XFormPrim(prim_path=left_hand_ring_p, name="LeftHandRing"))
        left_hand_ring1_p = left_hand_ring_p + "/Noitom_LeftHandRing1"
        world.scene.add(XFormPrim(prim_path=left_hand_ring1_p, name="LeftHandRing1"))
        left_hand_ring2_p = left_hand_ring1_p + "/Noitom_LeftHandRing2"
        world.scene.add(XFormPrim(prim_path=left_hand_ring2_p, name="LeftHandRing2"))
        left_hand_ring3_p = left_hand_ring2_p + "/Noitom_LeftHandRing3"
        world.scene.add(XFormPrim(prim_path=left_hand_ring3_p, name="LeftHandRing3"))
        left_hand_pinky_p = left_hand_p + "/Noitom_LeftHandPinky"
        world.scene.add(XFormPrim(prim_path=left_hand_pinky_p, name="LeftHandPinky"))
        left_hand_pinky1_p = left_hand_pinky_p + "/Noitom_LeftHandPinky1"
        world.scene.add(XFormPrim(prim_path=left_hand_pinky1_p, name="LeftHandPinky1"))
        left_hand_pinky2_p = left_hand_pinky1_p + "/Noitom_LeftHandPinky2"
        world.scene.add(XFormPrim(prim_path=left_hand_pinky2_p, name="LeftHandPinky2"))
        left_hand_pinky3_p = left_hand_pinky2_p + "/Noitom_LeftHandPinky3"
        world.scene.add(XFormPrim(prim_path=left_hand_pinky3_p, name="LeftHandPinky3"))

        self.application.open()
        return

    async def setup_post_load(self):
        self._world.add_physics_callback(
            "sending_actions", callback_fn=self.updateAvatarPosture
        )
        return

    def move_model(self, x, y, z):
        xform_prim = self._world.scene.get_object("PN_Stickman")
        xform_prim.set_local_pose([x, y, z])

    def q_multi(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return x, y, z, w

    # 欧拉角转四元数
    def to_quat(self, rot, ord="yxz"):
        r = R.from_euler(ord, rot, degrees=True)
        # [x, y, z, w]
        q = r.as_quat()
        return q

    # 四元数转欧拉角
    def to_euler(self, quat, degrees=True, ord="yxz"):
        # [x, y, z, w]
        r = R.from_quat(quat)
        # 欧拉角输出为 [yaw, pitch, roll] 对应于 [z, y, x]
        euler = r.as_euler(ord, degrees)
        # [z, y, x] ==> [x, -y, z]
        e = [euler[2], -euler[1], euler[0]]
        return e

    # 欧拉角反转问题
    def eulers_inverse(self, pre_angle, angle):
        angle[0] = self.euler_inverse(pre_angle[0], angle[0])
        angle[1] = self.euler_inverse(pre_angle[1], angle[1])
        angle[2] = self.euler_inverse(pre_angle[2], angle[2])
        return angle

    # 欧拉角反转问题
    @staticmethod
    def euler_inverse(pre_angle, angle):
        # 处理正负180度反转问题
        if abs(angle - pre_angle) > 180:
            if angle > 0:
                angle = angle - 360
            else:
                angle = angle + 360
        # 处理正负360度反转问题
        if abs(angle - pre_angle) > 360:
            if angle > 0:
                angle = angle - 720
            else:
                angle = angle + 720
        return angle

    # 旋转一个矢量
    def rot_vec(self, q, vec):
        # p to qp
        p4 = [vec[0], vec[1], vec[2], 0]
        # q_conj
        q_conjugate = [-q[0], -q[1], -q[2], q[3]]
        # p_new = q * p * q_conjugate
        q_p = self.q_multi(q, p4)
        p_new_quat = self.q_multi(q_p, q_conjugate)
        # Extract new p
        px, py, pz, _ = p_new_quat
        p_new = [px, py, pz]
        return p_new

    def has_bone_data(self):
        with self.lock:
            return self.data_updated

    def updateAvatarPosture(self, step_size):
        context = omni.usd.get_context()
        stage = context.get_stage()

        events = self.application.poll_next_event()
        for ev in events:
            if ev.event_type != mocap_api.MCPEventType.AvatarUpdated:
                continue

            self.data_updated = False
            with self.lock:
                # 更新模型
                avatar = mocap_api.MCPAvatar(ev.event_data.avatar_handle)
                for joint in avatar.get_joints():
                    # XFormPrim obj
                    bone_name = joint.get_name()
                    xform_prim = self._world.scene.get_object(bone_name)
                    if xform_prim is None:
                        continue

                    # update module
                    xform_prim.set_local_pose(
                        joint.get_local_position(),
                        joint.get_local_rotation(),
                        # if bone_name != "RightArm"
                        # else [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
                    )

                # 骨骼对象在世界系下的位姿
                for bone_name in self.avatar:
                    # XFormPrim obj
                    xform_prim = self._world.scene.get_object(bone_name)
                    if xform_prim is None:
                        continue

                    local_pos, local_quat = (
                        xform_prim.get_local_pose()
                    )  # x,y,z; w,x,y,z
                    world_pos, world_quat = xform_prim.get_world_pose()

                    self.avatar[bone_name]["px"] = float(local_pos[0])
                    self.avatar[bone_name]["py"] = float(local_pos[1])
                    self.avatar[bone_name]["pz"] = float(local_pos[2])
                    self.avatar[bone_name]["rx"] = local_quat[1]
                    self.avatar[bone_name]["ry"] = local_quat[2]
                    self.avatar[bone_name]["rz"] = local_quat[3]
                    self.avatar[bone_name]["rw"] = local_quat[0]

                    self.avatar[bone_name]["world_px"] = float(world_pos[0])
                    self.avatar[bone_name]["world_py"] = float(world_pos[1])
                    self.avatar[bone_name]["world_pz"] = float(world_pos[2])
                    self.avatar[bone_name]["world_rx"] = world_quat[1]
                    self.avatar[bone_name]["world_ry"] = world_quat[2]
                    self.avatar[bone_name]["world_rz"] = world_quat[3]
                    self.avatar[bone_name]["world_rw"] = world_quat[0]

                import omni.isaac.core.utils.transformations as isaacsim_tf
                import omni.isaac.core.utils.rotations as isaacsim_rotations

                relative_trans = {}

                right_arm_r2_hips = isaacsim_tf.pose_from_tf_matrix(
                    isaacsim_tf.get_relative_transform(
                        source_prim=self._world.scene.get_object("RightArm").prim,
                        target_prim=self._world.scene.get_object("Hips").prim,
                    )
                )
                hips_world_pose = self._world.scene.get_object("Hips").get_world_pose()
                spine2_r2_hips = isaacsim_tf.pose_from_tf_matrix(
                    isaacsim_tf.get_relative_transform(
                        source_prim=self._world.scene.get_object("Spine2").prim,
                        target_prim=self._world.scene.get_object("Hips").prim,
                    )
                )
                right_fore_arm_r2_hips = isaacsim_tf.pose_from_tf_matrix(
                    isaacsim_tf.get_relative_transform(
                        source_prim=self._world.scene.get_object("RightForeArm").prim,
                        target_prim=self._world.scene.get_object("Hips").prim,
                    )
                )
                right_hand_r2_right_fore_arm = isaacsim_tf.pose_from_tf_matrix(
                    isaacsim_tf.get_relative_transform(
                        source_prim=self._world.scene.get_object("RightHand").prim,
                        target_prim=self._world.scene.get_object("RightForeArm").prim,
                    )
                )
                right_hand_r2_right_arm = isaacsim_tf.pose_from_tf_matrix(
                    isaacsim_tf.get_relative_transform(
                        source_prim=self._world.scene.get_object("RightHand").prim,
                        target_prim=self._world.scene.get_object("RightArm").prim,
                    )
                )
                right_hand_r2_hips = isaacsim_tf.pose_from_tf_matrix(
                    isaacsim_tf.get_relative_transform(
                        source_prim=self._world.scene.get_object("RightHand").prim,
                        target_prim=self._world.scene.get_object("Hips").prim,
                    )
                )

                relative_trans["RightArm_Hips"] = right_arm_r2_hips
                relative_trans["Spine2_Hips"] = spine2_r2_hips
                relative_trans["RightForeArm_Hips"] = right_fore_arm_r2_hips
                relative_trans["RightHand_RightForeArm"] = right_hand_r2_right_fore_arm
                relative_trans["RightHand_RightArm"] = right_hand_r2_right_arm
                relative_trans["RightHand_Hips"] = right_hand_r2_hips
                relative_trans["Hips_World"] = hips_world_pose

                self.relative_trans = relative_trans

                # if self.quat_visualizer.is_running():
                #     self.quat_visualizer.update(quat_wxyz=right_arm_r2_hips[1])
                #     # self.quat_visualizer.update(
                #     #     quat_wxyz=[
                #     #         self.avatar["RightArm"]["rw"],
                #     #         self.avatar["RightArm"]["rx"],
                #     #         self.avatar["RightArm"]["ry"],
                #     #         self.avatar["RightArm"]["rz"],
                #     #     ]
                #     # )

                # isaacsim_tf.get_relative_transform()

            self.data_updated = True

    def get_relative_trans(self):
        with self.lock:
            return self.relative_trans

    # get actor's joints local angle
    def get_avatar_joints_data(self):
        with self.lock:
            return {
                bone_name: self.to_euler(
                    [data["rx"], data["ry"], data["rz"], data["rw"]]
                )
                for bone_name, data in self.avatar.items()
            }

    # get actor's local p(x,y,z), q(x,y,z,w)
    def get_local_avatar_data(self):
        with self.lock:
            return {
                bone_name: {
                    "pos": [data["px"], data["py"], data["pz"]],
                    "quat": [data["rx"], data["ry"], data["rz"], data["rw"]],
                }
                for bone_name, data in self.avatar.items()
            }

    # get actor's world p(x,y,z), q(x,y,z,w)
    def get_world_avatar_data(self):
        with self.lock:
            return {
                bone_name: {
                    "pos": [data["px"], data["py"], data["pz"]],
                    "quat": [data["rx"], data["ry"], data["rz"], data["rw"]],
                }
                for bone_name, data in self.avatar.items()
            }

    # get bone's world p(x,y,z), q(x,y,z,w)
    def get_world_bone_data(self, bone_name):
        with self.lock:
            if bone_name in self.avatar:
                data = self.avatar[bone_name]
                return {
                    "pos": [data["px"], data["py"], data["pz"]],
                    "quat": [data["rx"], data["ry"], data["rz"], data["rw"]],
                }
            return None
