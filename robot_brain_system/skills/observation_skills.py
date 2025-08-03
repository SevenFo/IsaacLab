# from thirdparty.ImagePipeline.imagepipeline import imagepipeline
from typing import Any
import torch
import numpy as np
import time
import open3d as o3d
from PIL import Image

from robot_brain_system.core.skill_manager import skill_register
from robot_brain_system.core.types import SkillType, ExecutionMode, BaseSkill
from robot_brain_system.core.model_adapters_v2 import OpenAIAdapter
from robot_brain_system.skills.imagepipleline import ImagePipeline
from robot_brain_system.utils.visualization_utils import visualize_all
from robot_brain_system.utils.config_utils import hydra_context_base
from robot_brain_system.skills.manipulation_skills import AliceControl
from cutie.utils.get_default_model import get_default_model


@skill_register(
    name="object_tracking",
    skill_type=SkillType.OBSERVATION,
    execution_mode=ExecutionMode.PREACTION,
    enable_monitoring=False,  # Disable monitoring for this skill
    requires_env=True,
)
class ObjectTracking(BaseSkill):
    """Adding a a tracker for the specific object in the environment, so that other skill can retrieve the pose information of the object from the observation.
    Especially useful for the skills that need to interact with the object and useful for solving object not found problem.
    Args:
        target_object (str): The target object name.
    """

    def __init__(
        self,
        policy_device: str = "cuda",
        env: Any = None,
        **running_params,
    ) -> None:
        super().__init__()
        print(f"[ObjectTracking] Loaded skill config: {self.cfg}")
        print(f"[SKILL: ObjectTracking: {running_params}")
        self.target_object = running_params.get("target_object", None)
        if self.target_object is None:
            raise ValueError("target_object must be specified.")
        self.pipeline_instance: dict[str, ImagePipeline] = {}
        banned_cameras = self.cfg.get("banned_cameras", [])
        print(f"[ObjectTracking] Banned cameras: {banned_cameras}")
        # handle alice
        print("[ObjectTracking] Initializing AliceControl")
        alice_control = AliceControl(
            alice_right_forearm_rigid_entity=env.scene["alice"],
            policy_device=env.device,
        )
        # TODO zero action 可以用 sim 的直接 update 来代替
        zero_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
            dtype=torch.float32,
        )
        obs_dict = alice_control.initialize(env, zero_action)

        cameras_data = {
            camera_name: data[0].cpu().numpy()
            for camera_name, data in obs_dict["policy"].items()
            if camera_name.startswith("camera_") and camera_name not in banned_cameras
        }
        self.vlm = OpenAIAdapter(
            "qwen2.5-vl-32b-awq",
            api_key="123",
            base_url="http://127.0.0.1:8000/v1",
        )
        sam_checkpoint = "thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"
        sam_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        with hydra_context_base():
            cutie_default_model = get_default_model()
        _init_success = []
        for key in cameras_data:
            self.pipeline_instance[key] = ImagePipeline(
                policy_device,
                sam_checkpoint,
                sam_model_config,
                cutie_default_model,
                self.vlm,
            )
            try:
                frame_image = Image.fromarray(cameras_data[key])
                frame_image.save(f"frame_image_{key}_{int(time.time() * 1000)}.png")
                self.pipeline_instance[key].initialize_with_instruction(
                    cameras_data[key], self.target_object, visualize=True
                )
                _init_success.append(key)
            except ValueError as e:
                # NO OBJECT IS DETECTED
                # TODO 如果所有视角都没看到就要抛出错误进行处理
                print(
                    f"[ObjectTracking] No object detected in {key}, as {e}, please check the camera data."
                )
                pass
        if len(_init_success) == 0:
            raise ValueError(
                f"[ObjectTracking] No camera initialized successfully for {self.target_object}. Please check the camera data."
            )

    def __call__(
        self,
        obs_dict: dict,
    ) -> Any:
        points_list = []
        for instance_key in self.pipeline_instance:
            if not self.pipeline_instance[instance_key].is_initialized:
                print(f"[ObjectTracking] {instance_key} is not initialized, skipping.")
                continue
            # 更新掩码
            all_mask, mask_list = self.pipeline_instance[instance_key].update_masks(
                obs_dict["policy"][instance_key][0],
                visualize=self.cfg.get("visualize", False),
            )
            assert len(mask_list) == 1, "Only one mask is expected."
            if not all_mask.any():
                print(
                    f"[ObjectTracking] No mask found for {self.target_object} in {instance_key}, skipping."
                )
                continue
            pointcloud: torch.tensor = obs_dict["policy"][f"pointcloud_{instance_key}"][
                0
            ]
            mask_flattened = (
                all_mask.transpose().flatten()
            )  # 转置后再展平，匹配列优先顺序
            masked_pointcloud = pointcloud[mask_flattened]
            points_list.append(masked_pointcloud)
            if self.cfg.get("debug_save", False):
                np.savetxt(
                    f"masked_pointcloud_{instance_key}_{int(time.time() * 10)}.txt",
                    masked_pointcloud.cpu().numpy(),
                )
                np.savetxt(
                    f"masked_pointcloud_unmasked_{instance_key}_{int(time.time() * 10)}.txt",
                    pointcloud.cpu().numpy(),
                )
                np.savetxt(
                    f"masked_pointcloud_mask_{instance_key}_{int(time.time() * 10)}.txt",
                    all_mask,
                )
            if self.cfg.get("visualize", False):
                visualize_all(
                    obs_dict["policy"][instance_key][0].cpu().numpy(),
                    None,
                    all_mask,
                    save_path=f"masked_pointcloud_vis_{instance_key}_{int(time.time() * 1000)}.png",
                )

        # 合并点云并转换为 numpy 数组
        if len(points_list) == 0:
            print(
                f"[ObjectTracking] No points found for {self.target_object}, returning empty observation."
            )
            # TODO 删除当前观测技能，并进行反馈！一种可能是随着技能的执行，所有视角中已经都看不到 target object 了导致所有视角的 mask 更新结果都为空
            return obs_dict
        points_np = torch.cat(points_list, dim=0).cpu().numpy()

        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        # 异常值去除和下采样
        pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
        pcd_downsampled = pcd_filtered.voxel_down_sample(voxel_size=0.01)

        # 获取 AABB
        aabb = pcd_downsampled.get_axis_aligned_bounding_box()
        obs_dict["policy"][f"{self.target_object}_aabb"] = aabb

        if self.cfg.get("visualize", False):
            import open3d.visualization as o3d_vis

            print(f"aabb of {self.target_object}: {aabb}")
            o3d_vis.draw_geometries([pcd_downsampled])

        if self.cfg.get("debug_save", False):
            o3d.io.write_point_cloud(
                f"masked_pointcloud_pcd_filtered_{self.target_object}_{int(time.time() * 10)}.ply",
                pcd_filtered,
            )
            o3d.io.write_point_cloud(
                f"masked_pointcloud_pcd_downsampled_{self.target_object}_{int(time.time() * 1000)}.ply",
                pcd_downsampled,
            )
        return obs_dict
