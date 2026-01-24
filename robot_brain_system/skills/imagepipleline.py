import torch
import json_repair
import numpy as np
import base64
import time
import gc
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
from robot_brain_system.ui.console import global_console

try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    global_console.log(
        "info",
        "Realsense library not available. Skipping RealsenseCamera functionality.",
    )


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

        参数：
        sam_checkpoint: SAM 模型权重路径
        sam_model_config: SAM 模型类型配置 (i.e. "configs/sam2.1/sam2.1_hiera_l.yaml")
        cutie_model: 预加载的 CUTIE 模型实例
        vl_adapter: 视觉语言模型适配器实例（QwenVLAdapter）
        """
        self.device = device
        self.device = "cuda:0"

        global_console.log(
            "skill", f"[ImagePipeline] Initializing with device: {self.device}"
        )

        # 使用封装的上下文管理器初始化 SAM2
        with hydra_config_context("pkg://sam2"):
            self.predictor = SAM2ImagePredictor(
                build_sam2(sam_model_config, sam_checkpoint, device=self.device)
            )

        # 初始化 CUTIE - 确保模型在正确设备上
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
        """将 numpy 图像转换为 base64 编码字符串"""
        pil_img = Image.fromarray(image)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @retry(
        max_attempts=3,
        delay_seconds=1.0,
        backoff_factor=2.0,
        exceptions_to_retry=(ValueError,),
        logger_func=lambda msg: global_console.log("skill", f"[ImagePipeline] {msg}"),
    )
    def get_bbox_from_vl(self, frame: np.ndarray, instruction: str) -> list:
        """
        使用视觉语言模型生成目标边界框

        参数：
        frame: RGB 格式的输入图像 [H, W, 3]
        instruction: 自然语言指令（如"the red cup on the left"）

        返回：
        bboxes: 边界框列表 [[x1,y1,x2,y2], ...]
        """

        def validate_bbox(bbox, frame_shape):
            """验证边界框是否在图像范围内（像素坐标）"""
            if len(bbox) != 4:
                return False
            x1, y1, x2, y2 = bbox
            return (
                0 <= x1 < frame_shape[1]
                and 0 <= y1 < frame_shape[0]
                and 0 < x2 <= frame_shape[1]
                and 0 < y2 <= frame_shape[0]
            )

        def convert_relative_to_pixel(bbox, frame_shape):
            """
            支持三种输入格式并转换为像素坐标：
            - 像素坐标 (x1,y1,x2,y2)
            - 归一化坐标 [0,1]
            - 相对坐标 [0,1000]（用户说明的 case）
            返回整数像素 bbox
            """
            x1, y1, x2, y2 = bbox
            h, w = frame_shape[0], frame_shape[1]

            # # 如果已经看起来像像素坐标（任一坐标大于 1 并且接近图像尺寸），直接返回
            # if max(x1, y1, x2, y2) > 1.5 and max(x1, y1, x2, y2) <= max(w, h) * 1.5:
            #     return [
            #         int(max(0, min(w, x1))),
            #         int(max(0, min(h, y1))),
            #         int(max(0, min(w, x2))),
            #         int(max(0, min(h, y2))),
            #     ]

            # 如果在 [0,1] 范围内，视为归一化坐标
            if (
                0.0 <= x1 <= 1.0
                and 0.0 <= y1 <= 1.0
                and 0.0 <= x2 <= 1.0
                and 0.0 <= y2 <= 1.0
            ):
                return [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]

            # 如果在 [0,1000] 范围内，视为相对 coords（按比例映射到图像尺寸）
            if (
                0 <= x1 <= 1000
                and 0 <= y1 <= 1000
                and 0 <= x2 <= 1000
                and 0 <= y2 <= 1000
            ):
                # 将 0..1000 映射到像素范围
                return [
                    int(max(0, min(w, x1 / 1000.0 * w))),
                    int(max(0, min(h, y1 / 1000.0 * h))),
                    int(max(0, min(w, x2 / 1000.0 * w))),
                    int(max(0, min(h, y2 / 1000.0 * h))),
                ]

            # 无法识别的格式，抛出错误以便上层处理
            raise ValueError(f"Unsupported bbox format or out-of-range values: {bbox}")

        # 构建 VL 模型输入
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

        global_console.log("skill", f"response from vl: \n{response}")

        # 提取 JSON 部分
        json_str = response[response.find("```json") : response.rfind("```") + 3]
        global_console.log("skill", f"json_str: \n{json_str}")
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

        # 验证并转换 bbox 格式（支持像素 / 归一化 / 0..1000）
        for item in bbox_list:
            bbox = item.get("bbox_2d", [])
            try:
                pixel_bbox = convert_relative_to_pixel(bbox, frame.shape)
            except Exception as e:
                global_console.log(
                    "skill", f"Invalid bbox: {bbox} in response. Reason: {e}"
                )
                continue
            if validate_bbox(pixel_bbox, frame.shape):
                valid_bboxes.append(pixel_bbox)
            else:
                global_console.log(
                    "skill",
                    f"Converted bbox out of bounds: {pixel_bbox} for frame shape {frame.shape}",
                )

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
        端到端初始化流程：VL 生成 bbox -> SAM 分割 -> CUTIE 初始化

        参数：
        frame: RGB 格式的输入图像
        instruction: 自然语言指令

        返回：
        combined_mask: 组合后的多目标 mask
        """
        # Step 1: 通过 VL 模型获取 bbox
        self.instruction = instruction
        bboxes = self.get_bbox_from_vl(frame, instruction)
        if not bboxes:
            raise ValueError(f"No objects found for instruction: {instruction}")
        if visualize:
            visualize_all(
                frame,
                bboxes,
                save_path=f"vis_{instruction}_{int(time.time() * 10)}.png",
            )
        # Step 2: SAM 生成 mask
        if return_bbox:
            return self.initialize_masks(frame, bboxes), bboxes
        else:
            return self.initialize_masks(frame, bboxes), None

    def initialize_masks(
        self, frame: np.ndarray, bboxes: list, visualize: bool = True
    ) -> np.ndarray:
        """
        初始化多目标分割

        参数：
        frame: RGB 格式的输入图像 [H, W, 3]
        bboxes: 多个目标的边界框列表 [[x1, y1, x2, y2], ...]

        返回：
        combined_mask: 组合后的多目标 mask，每个目标用不同整数 ID 表示
        """
        # 转换颜色空间并设置 SAM 图像
        rgb_frame = frame
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(rgb_frame)

            # 生成并组合多个目标的 mask
            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            object_ids = []
            for obj_idx, bbox in enumerate(bboxes):
                # SAM 预测最佳 mask
                masks, scores, _ = self.predictor.predict(
                    box=np.array(bbox), multimask_output=True
                )
                best_mask = masks[np.argmax(scores)]

                # 确保 mask 是布尔类型
                best_mask = best_mask.astype(bool)

                # 分配唯一对象 ID (从 1 开始)
                obj_id = obj_idx + 1
                combined_mask[best_mask] = obj_id
                object_ids.append(obj_id)

        # 初始化 CUTIE 处理器
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

        参数：
        frame: RGB 格式的新帧 [H, W, 3]

        返回：
        list: 每个目标的二值 mask 列表 [mask1, mask2, ...]
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

        # CUTIE 推理
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
        # 分离各个目标的 mask
        return current_mask_np, [
            (current_mask_np == obj_id) for obj_id in self.current_objects
        ]

    def add_object(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        动态添加新目标到现有跟踪

        参数：
        frame: RGB 格式的当前帧
        bbox: 新目标的边界框 [x1, y1, x2, y2]

        返回：
        new_mask: 新目标的单独 mask
        """
        # 生成新目标 mask
        rgb_frame = frame[..., ::-1]
        self.predictor.set_image(rgb_frame)
        masks, scores, _ = self.predictor.predict(
            box=np.array(bbox), multimask_output=True
        )
        new_mask = masks[np.argmax(scores)]

        # 分配新 ID
        new_id = max(self.current_objects) + 1 if self.current_objects else 1
        new_mask_tensor = torch.from_numpy(new_mask.astype(np.uint8) * new_id).to(
            self.device
        )

        # 合并到现有 mask
        combined_mask = self.processor.output_prob_to_mask(self.processor.prob)
        combined_mask = torch.where(new_mask_tensor > 0, new_mask_tensor, combined_mask)

        # 更新处理器状态
        self.current_objects.append(new_id)
        # 构建 image_tensor（与 update_masks 中的处理一致）
        if not isinstance(frame, torch.Tensor):
            image_tensor = to_tensor(rgb_frame).to(self.device)
        else:
            if frame.dim() == 3 and frame.shape[0] != 3:
                image_tensor = frame.permute(2, 0, 1).to(self.device)
            else:
                image_tensor = frame.to(self.device)
            if image_tensor.dtype == torch.uint8 or torch.max(image_tensor) > 1.0:
                image_tensor = image_tensor.float() / 255.0

        self.processor.step(image_tensor, combined_mask, self.current_objects)

        return new_mask

    def move_to(self, device: str):
        """
        将所有内部模型和处理器移动到指定的设备，但不修改它们的类定义。

        Args:
            device (str): 目标设备，例如 "cuda:1" 或 "cpu"。
        """
        target_device = torch.device(device)
        if (
            hasattr(self.predictor.model, "device")
            and self.predictor.model.device == target_device
        ):
            global_console.log(
                "skill", f"[ImagePipeline] Already on device: {device}. Nothing to do."
            )
            return

        global_console.log(
            "skill", f"[ImagePipeline] Moving all components to {device}..."
        )

        # 1. 移动 SAM2 的内部模型
        # 直接访问并调用 .to() 方法
        self.predictor.model = self.predictor.model.to(target_device)
        # 关键一步：移动后必须重置 predictor，以清除旧设备上的特征缓存
        self.predictor.reset_predictor()

        # 2. 移动 CUTIE 的内部模型
        self.cutie = self.cutie.to(target_device)
        # 更新 processor 对模型实例的引用，因为 .to() 可能返回新对象
        self.processor.network = self.cutie
        # 关键一步：清除 processor 的内存，因为它包含设备相关的张量
        self.processor.clear_memory()

        # 4. 更新管道自身的设备状态
        self.device = device
        self.is_initialized = False  # 移动后需要重新初始化跟踪状态

        global_console.log(
            "skill",
            f"[ImagePipeline] Successfully moved to {device}. Tracking state has been reset.",
        )

    def cleanup(self):
        """
        彻底清理并释放所有占用的资源，特别是 GPU 显存。
        """
        global_console.log("skill", "[ImagePipeline] Starting cleanup process...")

        # 步骤 1: 将所有模型移到 CPU，主动释放 GPU 张量
        if "cuda" in self.device:
            self.move_to("cpu")

        # 步骤 2: 删除对重量级对象的引用
        global_console.log(
            "skill", "[ImagePipeline] Deleting model and processor references..."
        )
        if hasattr(self, "predictor"):
            del self.predictor
        if hasattr(self, "cutie"):
            del self.cutie
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "vl_adapter"):
            del self.vl_adapter

        # 步骤 3: (关键!) 调用垃圾回收并强制 PyTorch 清理其缓存
        gc.collect()
        if torch.cuda.is_available():
            global_console.log(
                "skill", "[ImagePipeline] Clearing PyTorch CUDA cache..."
            )
            torch.cuda.empty_cache()

        global_console.log("skill", "[ImagePipeline] Cleanup complete.")

    def reset(self):
        """重置管道状态"""
        if hasattr(self, "processor"):
            self.processor.clear_memory()
        self.current_objects = []
        self.is_initialized = False
        self.instruction = ""
        global_console.log("skill", "[ImagePipeline] State has been reset.")


# 使用示例
if __name__ == "__main__":
    # setup_all()

    # for centers in main_rs_iter(
    #     save_video=False,
    #     disable_vlm=False,
    #     video_source="camera",
    #     video_path="/data/shiqi/ImagePipelien/imagepipeline/output.mp4",
    # ):
    #     global_console.log("skill",centers)

    import os

    os.environ["DISPLAY"] = ":0"  # 设置显示环境变量
    global_console.log(
        "skill",
        "ImagePipeline: no demo available in module context. Import ImagePipeline and call its methods from your application.",
    )

    # shutdown_all()
