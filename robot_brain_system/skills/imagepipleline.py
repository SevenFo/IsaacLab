import torch
import json_repair
import numpy as np
import base64
import time
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from cutie.inference.inference_core import InferenceCore
from torchvision.transforms.functional import to_tensor
from io import BytesIO

from robot_brain_system.utils.visualization_utils import visualize_all
from robot_brain_system.utils.retry_utils import retry
from robot_brain_system.utils.config_utils import hydra_config_context
from robot_brain_system.core.brain import BrainMemory

try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Realsense library not available. Skipping RealsenseCamera functionality.")


class ImagePipeline:
    def __init__(
        self,
        device: str,
        sam_checkpoint: str,
        sam_model_config: str,
        cutie_model,
        vl_adapter,
    ):
        """
        多目标分割与跟踪管道（集成视觉语言模型）

        参数:
        sam_checkpoint: SAM模型权重路径
        sam_model_config: SAM模型类型配置 (i.e. "configs/sam2.1/sam2.1_hiera_l.yaml")
        cutie_model: 预加载的CUTIE模型实例
        vl_adapter: 视觉语言模型适配器实例（QwenVLAdapter）
        """
        self.device = device

        # 使用封装的上下文管理器初始化 SAM2
        with hydra_config_context("pkg://sam2"):
            self.predictor = SAM2ImagePredictor(
                build_sam2(sam_model_config, sam_checkpoint, device=self.device)
            )

        # 初始化CUTIE - 确保模型在正确设备上
        self.cutie = cutie_model.to(self.device)
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = -1

        # 集成视觉语言模型
        self.vl_adapter = vl_adapter

        # 状态跟踪
        self.current_objects = []
        self.is_initialized = False
        self.instruction = ""

    def _image_to_base64(self, image: np.ndarray) -> str:
        """将numpy图像转换为base64编码字符串"""
        pil_img = Image.fromarray(image)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @retry(
        max_attempts=3,
        delay_seconds=1.0,
        backoff_factor=2.0,
        exceptions_to_retry=(ValueError,),
        logger_func=lambda msg: print(f"[ImagePipeline] {msg}"),
    )
    def get_bbox_from_vl(self, frame: np.ndarray, instruction: str) -> list:
        """
        使用视觉语言模型生成目标边界框

        参数:
        frame: RGB格式的输入图像 [H, W, 3]
        instruction: 自然语言指令（如"the red cup on the left"）

        返回:
        bboxes: 边界框列表 [[x1,y1,x2,y2], ...]
        """

        def validate_bbox(bbox, frame_shape):
            """验证边界框是否在图像范围内"""
            if len(bbox) != 4:
                return False
            x1, y1, x2, y2 = bbox
            return (
                0 <= x1 < frame_shape[1]
                and 0 <= y1 < frame_shape[0]
                and 0 < x2 <= frame_shape[1]
                and 0 < y2 <= frame_shape[0]
            )

        # 构建VL模型输入
        prompt = BrainMemory()
        prompt.add_user_input(
            [
                f"Analyze the image and identify ALL objects matching: {instruction}.\n"
                "Return bboxes for ALL matching objects in this format:\n"
                '[{"bbox_2d": [x1,y1,x2,y2], "label": "..."}, ...]',
                Image.fromarray(frame),
            ]
        )

        # 生成响应

        response, _ = self.vl_adapter.generate(prompt.history, max_tokens=512)

        print(f"response from vl: {response}")

        # 提取JSON部分
        json_str = response[response.find("```json") : response.rfind("```") + 3]
        bbox_list = json_repair.loads(json_str)

        valid_bboxes = []

        if isinstance(bbox_list, dict):
            bbox_list = [bbox_list]  # 确保是列表格式

        if not all(isinstance(item, dict) and "bbox_2d" in item for item in bbox_list):
            # treat as one instance if bbox list
            if validate_bbox(bbox_list, frame.shape):
                valid_bboxes.append(bbox_list)
                return valid_bboxes
            else:
                raise ValueError("Invalid bbox format or values in response.")

        # 验证bbox格式
        for item in bbox_list:
            bbox = item.get("bbox_2d", [])
            if len(bbox) == 4 and all(
                0 <= v <= frame.shape[1] if i % 2 == 0 else 0 <= v <= frame.shape[0]
                for i, v in enumerate(bbox)
            ):
                valid_bboxes.append(bbox)

        if not valid_bboxes:
            raise ValueError("No valid bounding boxes found in VL model response.")

        return valid_bboxes

    def initialize_with_instruction(
        self,
        frame: np.ndarray,
        instruction: str,
        return_bbox: bool = False,
        visualize: bool = False,
    ) -> tuple[np.ndarray, list | None]:
        """
        端到端初始化流程：VL生成bbox -> SAM分割 -> CUTIE初始化

        参数:
        frame: RGB格式的输入图像
        instruction: 自然语言指令

        返回:
        combined_mask: 组合后的多目标mask
        """
        # Step 1: 通过VL模型获取bbox
        self.instruction = instruction
        bboxes = self.get_bbox_from_vl(frame, instruction)
        if not bboxes:
            raise ValueError("No valid bounding boxes detected by VL model")
        if visualize:
            visualize_all(
                frame,
                bboxes,
                save_path=f"vis_{instruction}_{int(time.time() * 10)}.png",
            )
        # Step 2: SAM生成mask
        if return_bbox:
            return self.initialize_masks(frame, bboxes), bboxes
        else:
            return self.initialize_masks(frame, bboxes), None

    def initialize_masks(
        self, frame: np.ndarray, bboxes: list, visualize: bool = True
    ) -> np.ndarray:
        """
        初始化多目标分割

        参数:
        frame: RGB格式的输入图像 [H, W, 3]
        bboxes: 多个目标的边界框列表 [[x1, y1, x2, y2], ...]

        返回:
        combined_mask: 组合后的多目标mask，每个目标用不同整数ID表示
        """
        # 转换颜色空间并设置SAM图像
        rgb_frame = frame
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(rgb_frame)

            # 生成并组合多个目标的mask
            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            object_ids = []
            for obj_idx, bbox in enumerate(bboxes):
                # SAM预测最佳mask
                masks, scores, _ = self.predictor.predict(
                    box=np.array(bbox), multimask_output=True
                )
                best_mask = masks[np.argmax(scores)]

                # 确保mask是布尔类型
                best_mask = best_mask.astype(bool)

                # 分配唯一对象ID (从1开始)
                obj_id = obj_idx + 1
                combined_mask[best_mask] = obj_id
                object_ids.append(obj_id)

        # 初始化CUTIE处理器
        mask_tensor = torch.from_numpy(combined_mask).to(self.device)
        self.processor.clear_memory()
        self.processor.step(
            to_tensor(rgb_frame).to(self.device), mask_tensor, object_ids
        )

        if visualize:
            visualize_all(
                rgb_frame,
                bboxes=bboxes,
                mask=combined_mask,
                save_path=f"vis_init_{self.instruction}_{int(time.time() * 10)}.png",
            )

        # 更新状态
        self.current_objects = object_ids
        self.is_initialized = True

        return combined_mask

    def update_masks(
        self, frame: np.ndarray | torch.Tensor | Image.Image, visualize: bool = False
    ) -> tuple[np.ndarray, list]:
        """
        更新多目标跟踪结果

        参数:
        frame: RGB格式的新帧 [H, W, 3]

        返回:
        list: 每个目标的二值mask列表 [mask1, mask2, ...]
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize_masks first.")

        # 准备输入数据
        if not isinstance(frame, torch.Tensor):
            rgb_frame = frame
            image_tensor = to_tensor(rgb_frame).to(self.device)
        else:
            if frame.dim() == 3 and frame.shape[0] != 3:
                # 如果是 [H, W, 3] 格式，转换为 [3, H, W]
                image_tensor = frame.permute(2, 0, 1).to(self.device)
            else:
                image_tensor = frame.to(self.device)
            if image_tensor.dtype == torch.uint8 or torch.max(image_tensor) > 1.0:
                image_tensor = image_tensor.float() / 255.0

        # CUTIE推理
        with torch.no_grad():
            output_prob = self.processor.step(image_tensor)
            current_mask = self.processor.output_prob_to_mask(output_prob)
            current_mask_np = current_mask.cpu().numpy().astype(np.uint8)
        if visualize:
            visualize_all(
                frame.cpu().numpy() if isinstance(frame, torch.Tensor) else frame,
                None,
                current_mask_np,
                save_path=f"vis_mask_{self.instruction}_{int(time.time() * 10)}.png",
            )
        # 分离各个目标的mask
        return current_mask_np, [
            (current_mask_np == obj_id) for obj_id in self.current_objects
        ]

    def reset(self):
        """重置管道状态"""
        self.processor.clear_memory()
        self.current_objects = []
        self.is_initialized = False

    def add_object(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        动态添加新目标到现有跟踪

        参数:
        frame: RGB格式的当前帧
        bbox: 新目标的边界框 [x1, y1, x2, y2]

        返回:
        new_mask: 新目标的单独mask
        """
        # 生成新目标mask
        rgb_frame = frame[..., ::-1]
        self.predictor.set_image(rgb_frame)
        masks, scores, _ = self.predictor.predict(
            box=np.array(bbox), multimask_output=True
        )
        new_mask = masks[np.argmax(scores)]

        # 分配新ID
        new_id = max(self.current_objects) + 1 if self.current_objects else 1
        new_mask_tensor = torch.from_numpy(new_mask.astype(np.uint8) * new_id).to(
            self.device
        )

        # 合并到现有mask
        combined_mask = self.processor.output_prob_to_mask(self.processor.prob)
        combined_mask = torch.where(new_mask_tensor > 0, new_mask_tensor, combined_mask)

        # 更新处理器状态
        self.current_objects.append(new_id)
        self.processor.step(image_tensor, combined_mask, self.current_objects)

        return new_mask


# 使用示例
if __name__ == "__main__":
    # setup_all()

    # for centers in main_rs_iter(
    #     save_video=False,
    #     disable_vlm=False,
    #     video_source="camera",
    #     video_path="/data/shiqi/ImagePipelien/imagepipeline/output.mp4",
    # ):
    #     print(centers)

    import os

    os.environ["DISPLAY"] = ":0"  # 设置显示环境变量
    main_demo()

    # shutdown_all()
