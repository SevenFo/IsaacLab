# from thirdparty.ImagePipeline.imagepipeline import imagepipeline
from typing import Any
import torch
import open3d as o3d

from robot_brain_system.core.skill_manager import skill_register
from robot_brain_system.core.types import SkillType, ExecutionMode
from robot_brain_system.core.model_adapters_v2 import OpenAIAdapter
from robot_brain_system.skills.imagepipleline import ImagePipeline

from cutie.utils.get_default_model import get_default_model


@skill_register(
    name="object_tracking",
    skill_type=SkillType.OBSERVATION,
    execution_mode=ExecutionMode.PREACTION,
    enable_monitoring=False,  # Disable monitoring for this skill
    requires_env=True,
)
class ObjectTracking:
    """Adding a a tracker for the specific object in the environment, so that other skill can retrieve the pose information of the object from the observation.
    Especially useful for the skills that need to interact with the object and useful for solving object not found problem.
    Args:
        target_object (str): The target object name.
    """

    def __init__(
        self,
        policy_device: str = "cuda",
        obs_dict: dict = {},
        **running_params,
    ) -> None:
        print(f"[SKILL: ObjectTracking: {running_params}")
        self.target_object = running_params.get("target_object", None)
        if self.target_object is None:
            raise ValueError("target_object must be specified.")
        self.pipeline_instance: dict[str, ImagePipeline] = {}
        cameras_data = {
            camera_name: data[0].cpu().numpy()
            for camera_name, data in obs_dict["policy"].items()
            if camera_name.startswith("camera_")
        }
        self.vlm = OpenAIAdapter(
            "qwen2.5-vl-32b-awq",
            api_key="123",
            base_url="http://127.0.0.1:8000/v1",
        )
        sam_checkpoint = "thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"
        sam_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
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

    def __call__(self, obs_dict: dict, visualize: bool = False) -> Any:
        points_list = []
        for instance_key in self.pipeline_instance:
            if not self.pipeline_instance[instance_key].is_initialized:
                print(f"[ObjectTracking] {instance_key} is not initialized, skipping.")
                continue
            # 更新掩码
            all_mask, mask_list = self.pipeline_instance[instance_key].update_masks(
                obs_dict["policy"][instance_key][0], visualize=visualize
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
            masked_pointcloud = pointcloud[all_mask.flatten()]
            points_list.append(masked_pointcloud)
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

        if visualize:
            import open3d.visualization as o3d_vis

            print(f"aabb of {self.target_object}: {aabb}")
            o3d_vis.draw_geometries([pcd_downsampled])

        return obs_dict
