from omni.isaac.core import World
from omni.isaac.core.scenes import Scene
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.prims import RigidPrimView
from pxr import Usd, UsdGeom, Gf, UsdPhysics, Sdf  # Added UsdPhysics, Sdf

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
        context = omni.usd.get_context()
        self.stage = context.get_stage()

        self.bone_prims_paths = []  # To store paths for RigidPrimView
        self.bone_name_to_view_idx = {}  # To map bone names to RigidPrimView indices
        self.character_root_path = "/World/PN_Stickman_v12_ThumbInward"  # Adjust if prim_prefix is used at root
        if self.prim_prefix:
            self.character_root_path = (
                f"/World/PN_Stickman_v12_ThumbInward{self.prim_prefix}"
            )
        self.mocap_character_view = None  # Will be RigidPrimView
        self.num_mocap_bones = 0

        self.data_updated = False
        self.lock = threading.Lock()  # 用于保护avatar obj
        self.init_bones()
        # self.quat_visualizer = MultithreadVisualizer()
        self.my_scene = {}
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

    def setup_scene(self):
        if not self.stage.GetPrimAtPath("/World"):
            raise RuntimeError(f"World prim at '/World' does not exist.")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        omni.kit.commands.execute(
            "CreatePayload",
            usd_context=omni.usd.get_context(),
            path_to="/World/PN_Stickman_v12_ThumbInward",
            asset_path=script_dir
            + "/avatar/PN_Stickman_v12_ThumbInward_RightHandPMKINE.usd",  # _RightHandPM
            instanceable=False,
        )
        # 缩放模型从厘米到米
        asset_root_prim = self.stage.GetPrimAtPath("/World/PN_Stickman_v12_ThumbInward")
        UsdGeom.XformCommonAPI(asset_root_prim).SetScale(
            (0.01, 0.01, 0.01)
        )  # 缩小100倍

        self.bone_prims_paths = []
        self.bone_name_to_view_idx = {}
        temp_bone_xform_prims = {}  # To store your XFormPrim wrappers if still needed

        root_p = self.character_root_path
        self.bone_to_usd_path_map = {}

        def add_and_configure_bone(
            name: str, usd_path: str, is_rigid_body: bool = True
        ):
            prim = self.stage.GetPrimAtPath(usd_path)
            if not prim.IsValid():
                print(f"Warning: Prim at path {usd_path} for bone {name} not found.")
                return

            # Store for your XFormPrim wrapper if you still use it for non-physics things
            temp_bone_xform_prims[name] = XFormPrim(prim_path=usd_path, name=name)

        # self.my_scene.add(XFormPrim(prim_path=root_p, name="PN_Stickman"))

        add_and_configure_bone("PN_Stickman", root_p)

        hip_p = root_p + "/Noitom_Hips"

        # prim = self.stage.GetPrimAtPath(hip_p)
        # rb_api = UsdPhysics.RigidBodyAPI.Get(self.stage, hip_p)
        # if not rb_api:
        #     rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        #     print(f"Applied RigidBodyAPI to {hip_p}")
        # # 2. Set as Kinematic
        # kinematic_attr = rb_api.GetKinematicEnabledAttr()
        # if not kinematic_attr:  # Create if it doesn't exist
        #     kinematic_attr = rb_api.CreateKinematicEnabledAttr()
        # kinematic_attr.Set(True)
        # # print(f"Set {usd_path} to kinematic.")

        add_and_configure_bone("Hips", hip_p)
        left_up_leg_p = hip_p + "/Noitom_LeftUpLeg"
        add_and_configure_bone("LeftUpLeg", left_up_leg_p)
        left_leg_p = left_up_leg_p + "/Noitom_LeftLeg"
        add_and_configure_bone("LeftLeg", left_leg_p)
        add_and_configure_bone("LeftFoot", left_leg_p + "/Noitom_LeftFoot")
        right_up_leg_p = hip_p + "/Noitom_RightUpLeg"
        add_and_configure_bone("RightUpLeg", right_up_leg_p)
        right_leg_p = right_up_leg_p + "/Noitom_RightLeg"
        add_and_configure_bone("RightLeg", right_leg_p)
        add_and_configure_bone("RightFoot", right_leg_p + "/Noitom_RightFoot")

        spine_p = hip_p + "/Noitom_Spine"
        add_and_configure_bone("Spine", spine_p)
        spine1_p = spine_p + "/Noitom_Spine1"
        add_and_configure_bone("Spine1", spine1_p)
        spine2_p = spine1_p + "/Noitom_Spine2"
        add_and_configure_bone("Spine2", spine2_p)
        neck_p = spine2_p + "/Noitom_Neck"
        add_and_configure_bone("Neck", neck_p)
        neck1_p = neck_p + "/Noitom_Neck1"
        add_and_configure_bone("Neck1", neck1_p)
        head_p = neck1_p + "/Noitom_Head"
        add_and_configure_bone("Head", head_p)

        right_shoulder_p = spine2_p + "/Noitom_RightShoulder"
        add_and_configure_bone("RightShoulder", right_shoulder_p)
        right_arm_p = right_shoulder_p + "/Noitom_RightArm"
        add_and_configure_bone("RightArm", right_arm_p)
        right_fore_arm_p = right_arm_p + "/Noitom_RightForeArm"
        add_and_configure_bone("RightForeArm", right_fore_arm_p)
        right_hand_p = right_fore_arm_p + "/Noitom_RightHand"
        add_and_configure_bone("RightHand", right_hand_p)
        right_hand_thumb1_p = right_hand_p + "/Noitom_RightHandThumb1"
        add_and_configure_bone("RightHandThumb1", right_hand_thumb1_p)
        right_hand_thumb2_p = right_hand_thumb1_p + "/Noitom_RightHandThumb2"
        add_and_configure_bone("RightHandThumb2", right_hand_thumb2_p)
        right_hand_thumb3_p = right_hand_thumb2_p + "/Noitom_RightHandThumb3"
        add_and_configure_bone("RightHandThumb3", right_hand_thumb3_p)
        right_hand_index_p = right_hand_p + "/Noitom_RightHandIndex"
        add_and_configure_bone("RightHandIndex", right_hand_index_p)
        right_hand_index1_p = right_hand_index_p + "/Noitom_RightHandIndex1"
        add_and_configure_bone("RightHandIndex1", right_hand_index1_p)
        right_hand_index2_p = right_hand_index1_p + "/Noitom_RightHandIndex2"
        right_hand_index2_p = right_hand_index1_p + "/Noitom_RightHandIndex2"
        right_hand_index3_p = right_hand_index2_p + "/Noitom_RightHandIndex3"
        add_and_configure_bone("RightHandIndex3", right_hand_index3_p)
        right_hand_middle_p = right_hand_p + "/Noitom_RightHandMiddle"
        add_and_configure_bone("RightHandMiddle", right_hand_middle_p)
        right_hand_middle1_p = right_hand_middle_p + "/Noitom_RightHandMiddle1"
        add_and_configure_bone("RightHandMiddle1", right_hand_middle1_p)
        right_hand_middle2_p = right_hand_middle1_p + "/Noitom_RightHandMiddle2"
        add_and_configure_bone("RightHandMiddle2", right_hand_middle2_p)
        right_hand_middle3_p = right_hand_middle2_p + "/Noitom_RightHandMiddle3"
        add_and_configure_bone("RightHandMiddle3", right_hand_middle3_p)
        right_hand_ring_p = right_hand_p + "/Noitom_RightHandRing"
        add_and_configure_bone("RightHandRing", right_hand_ring_p)
        right_hand_ring1_p = right_hand_ring_p + "/Noitom_RightHandRing1"
        add_and_configure_bone("RightHandRing1", right_hand_ring1_p)
        right_hand_ring2_p = right_hand_ring1_p + "/Noitom_RightHandRing2"
        add_and_configure_bone("RightHandRing2", right_hand_ring2_p)
        right_hand_ring3_p = right_hand_ring2_p + "/Noitom_RightHandRing3"
        add_and_configure_bone("RightHandRing3", right_hand_ring3_p)
        right_hand_pinky_p = right_hand_p + "/Noitom_RightHandPinky"
        add_and_configure_bone("RightHandPinky", right_hand_pinky_p)
        right_hand_pinky1_p = right_hand_pinky_p + "/Noitom_RightHandPinky1"
        add_and_configure_bone("RightHandPinky1", right_hand_pinky1_p)
        right_hand_pinky2_p = right_hand_pinky1_p + "/Noitom_RightHandPinky2"
        add_and_configure_bone("RightHandPinky2", right_hand_pinky2_p)
        right_hand_pinky3_p = right_hand_pinky2_p + "/Noitom_RightHandPinky3"
        add_and_configure_bone("RightHandPinky3", right_hand_pinky3_p)

        left_shoulder_p = spine2_p + "/Noitom_LeftShoulder"
        add_and_configure_bone("LeftShoulder", left_shoulder_p)
        left_arm_p = left_shoulder_p + "/Noitom_LeftArm"
        add_and_configure_bone("LeftArm", left_arm_p)
        left_fore_arm_p = left_arm_p + "/Noitom_LeftForeArm"
        add_and_configure_bone("LeftForeArm", left_fore_arm_p)
        left_hand_p = left_fore_arm_p + "/Noitom_LeftHand"
        add_and_configure_bone("LeftHand", left_hand_p)
        left_hand_thumb1_p = left_hand_p + "/Noitom_LeftHandThumb1"
        add_and_configure_bone("LeftHandThumb1", left_hand_thumb1_p)
        left_hand_thumb2_p = left_hand_thumb1_p + "/Noitom_LeftHandThumb2"
        add_and_configure_bone("LeftHandThumb2", left_hand_thumb2_p)
        left_hand_thumb3_p = left_hand_thumb2_p + "/Noitom_LeftHandThumb3"
        add_and_configure_bone("LeftHandThumb3", left_hand_thumb3_p)
        left_hand_index_p = left_hand_p + "/Noitom_LeftHandIndex"
        add_and_configure_bone("LeftHandIndex", left_hand_index_p)
        left_hand_index1_p = left_hand_index_p + "/Noitom_LeftHandIndex1"
        add_and_configure_bone("LeftHandIndex1", left_hand_index1_p)
        left_hand_index2_p = left_hand_index1_p + "/Noitom_LeftHandIndex2"
        add_and_configure_bone("LeftHandIndex2", left_hand_index2_p)
        left_hand_index3_p = left_hand_index2_p + "/Noitom_LeftHandIndex3"
        add_and_configure_bone("LeftHandIndex3", left_hand_index3_p)
        left_hand_middle_p = left_hand_p + "/Noitom_LeftHandMiddle"
        add_and_configure_bone("LeftHandMiddle", left_hand_middle_p)
        left_hand_middle1_p = left_hand_middle_p + "/Noitom_LeftHandMiddle1"
        add_and_configure_bone("LeftHandMiddle1", left_hand_middle1_p)
        left_hand_middle2_p = left_hand_middle1_p + "/Noitom_LeftHandMiddle2"
        add_and_configure_bone("LeftHandMiddle2", left_hand_middle2_p)
        left_hand_middle3_p = left_hand_middle2_p + "/Noitom_LeftHandMiddle3"
        add_and_configure_bone("LeftHandMiddle3", left_hand_middle3_p)
        left_hand_ring_p = left_hand_p + "/Noitom_LeftHandRing"
        add_and_configure_bone("LeftHandRing", left_hand_ring_p)
        left_hand_ring1_p = left_hand_ring_p + "/Noitom_LeftHandRing1"
        add_and_configure_bone("LeftHandRing1", left_hand_ring1_p)
        left_hand_ring2_p = left_hand_ring1_p + "/Noitom_LeftHandRing2"
        add_and_configure_bone("LeftHandRing2", left_hand_ring2_p)
        left_hand_ring3_p = left_hand_ring2_p + "/Noitom_LeftHandRing3"
        add_and_configure_bone("LeftHandRing3", left_hand_ring3_p)
        left_hand_pinky_p = left_hand_p + "/Noitom_LeftHandPinky"
        add_and_configure_bone("LeftHandPinky", left_hand_pinky_p)
        left_hand_pinky1_p = left_hand_pinky_p + "/Noitom_LeftHandPinky1"
        add_and_configure_bone("LeftHandPinky1", left_hand_pinky1_p)
        left_hand_pinky2_p = left_hand_pinky1_p + "/Noitom_LeftHandPinky2"
        add_and_configure_bone("LeftHandPinky2", left_hand_pinky2_p)
        left_hand_pinky3_p = left_hand_pinky2_p + "/Noitom_LeftHandPinky3"
        add_and_configure_bone("LeftHandPinky3", left_hand_pinky3_p)

        self.my_scene = temp_bone_xform_prims

        # Initialize RigidPrimView
        # if self.bone_prims_paths:
        #     self.mocap_character_view = RigidPrimView(
        #         prim_paths_expr=self.bone_prims_paths,  # List of exact prim paths
        #         name="mocap_driven_bones_view",
        #         # positions= # You can set initial positions here if needed
        #         # orientations= # And orientations
        #     )
        #     self.num_mocap_bones = len(self.bone_prims_paths)
        #     print(f"RigidPrimView initialized with {self.num_mocap_bones} bones.")

        #     self.mocap_character_view.initialize(
        #         physics_sim_view=self._world.physics_sim_view
        #     )
        #     print(f"RigidPrimView '{self.mocap_character_view.name}' initialized")

        # else:
        #     print("Warning: No bone prim paths collected for RigidPrimView.")

        self.application.open()
        return

    async def setup_post_load(self):
        self._world.add_physics_callback(
            "sending_actions", callback_fn=self.updateAvatarPosture
        )
        return

    def move_model(self, x=None, y=None, z=None, qw=None, qx=None, qy=None, qz=None):
        xform_prim: XFormPrim = self.my_scene["PN_Stickman"]
        trans = [x, y, z]
        orient = [qw, qx, qy, qz]
        xform_prim.set_local_pose(
            translation=tuple(trans) if (None not in set([x, y, z])) else None,
            orientation=tuple(orient) if (None not in set([qw, qx, qy, qz])) else None,
        )

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
        # if self.mocap_character_view.count == 0:
        #     print("Mocap view has no prims.")
        #     return

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
                    if (
                        bone_name not in self.bone_names
                        or bone_name not in self.my_scene
                    ):
                        continue
                    xform_prim = self.my_scene[bone_name]
                    if xform_prim is None:
                        continue

                    # update module
                    xform_prim.set_local_pose(
                        joint.get_local_position(),
                        joint.get_local_rotation(),
                        # if bone_name != "RightArm"
                        # else [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
                    )

                # # 骨骼对象在世界系下的位姿
                # for bone_name in self.avatar:
                #     # XFormPrim obj
                #     xform_prim = self.my_scene.get_object(bone_name)
                #     if xform_prim is None:
                #         continue

                #     local_pos, local_quat = (
                #         xform_prim.get_local_pose()
                #     )  # x,y,z; w,x,y,z
                #     world_pos, world_quat = xform_prim.get_world_pose()

                #     self.avatar[bone_name]["px"] = float(local_pos[0])
                #     self.avatar[bone_name]["py"] = float(local_pos[1])
                #     self.avatar[bone_name]["pz"] = float(local_pos[2])
                #     self.avatar[bone_name]["rx"] = local_quat[1]
                #     self.avatar[bone_name]["ry"] = local_quat[2]
                #     self.avatar[bone_name]["rz"] = local_quat[3]
                #     self.avatar[bone_name]["rw"] = local_quat[0]

                #     self.avatar[bone_name]["world_px"] = float(world_pos[0])
                #     self.avatar[bone_name]["world_py"] = float(world_pos[1])
                #     self.avatar[bone_name]["world_pz"] = float(world_pos[2])
                #     self.avatar[bone_name]["world_rx"] = world_quat[1]
                #     self.avatar[bone_name]["world_ry"] = world_quat[2]
                #     self.avatar[bone_name]["world_rz"] = world_quat[3]
                #     self.avatar[bone_name]["world_rw"] = world_quat[0]

                # import isaacsim.core.utils.transformations as isaacsim_tf
                # import isaacsim.core.utils.rotations as isaacsim_rotations

                # relative_trans = {}

                # right_arm_r2_hips = isaacsim_tf.pose_from_tf_matrix(
                #     isaacsim_tf.get_relative_transform(
                #         source_prim=self.my_scene.get_object("RightArm").prim,
                #         target_prim=self.my_scene.get_object("Hips").prim,
                #     )
                # )
                # hips_world_pose = self.my_scene.get_object("Hips").get_world_pose()
                # spine2_r2_hips = isaacsim_tf.pose_from_tf_matrix(
                #     isaacsim_tf.get_relative_transform(
                #         source_prim=self.my_scene.get_object("Spine2").prim,
                #         target_prim=self.my_scene.get_object("Hips").prim,
                #     )
                # )
                # right_fore_arm_r2_hips = isaacsim_tf.pose_from_tf_matrix(
                #     isaacsim_tf.get_relative_transform(
                #         source_prim=self.my_scene.get_object("RightForeArm").prim,
                #         target_prim=self.my_scene.get_object("Hips").prim,
                #     )
                # )
                # right_hand_r2_right_fore_arm = isaacsim_tf.pose_from_tf_matrix(
                #     isaacsim_tf.get_relative_transform(
                #         source_prim=self.my_scene.get_object("RightHand").prim,
                #         target_prim=self.my_scene.get_object("RightForeArm").prim,
                #     )
                # )
                # right_hand_r2_right_arm = isaacsim_tf.pose_from_tf_matrix(
                #     isaacsim_tf.get_relative_transform(
                #         source_prim=self.my_scene.get_object("RightHand").prim,
                #         target_prim=self.my_scene.get_object("RightArm").prim,
                #     )
                # )
                # right_hand_r2_hips = isaacsim_tf.pose_from_tf_matrix(
                #     isaacsim_tf.get_relative_transform(
                #         source_prim=self.my_scene.get_object("RightHand").prim,
                #         target_prim=self.my_scene.get_object("Hips").prim,
                #     )
                # )

                # relative_trans["RightArm_Hips"] = right_arm_r2_hips
                # relative_trans["Spine2_Hips"] = spine2_r2_hips
                # relative_trans["RightForeArm_Hips"] = right_fore_arm_r2_hips
                # relative_trans["RightHand_RightForeArm"] = right_hand_r2_right_fore_arm
                # relative_trans["RightHand_RightArm"] = right_hand_r2_right_arm
                # relative_trans["RightHand_Hips"] = right_hand_r2_hips
                # relative_trans["Hips_World"] = hips_world_pose

                # self.relative_trans = relative_trans

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
