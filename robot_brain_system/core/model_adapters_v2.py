# -*- coding: utf-8 -*-
"""
统一的模型适配器模块（已修正版本），为不同推理后端提供标准化接口。
"""

import base64
from abc import ABC, abstractmethod
import copy
from io import BytesIO
import json
from typing import Any, Dict, List, Optional, Tuple
import argparse

import torch
import os
from PIL import Image

from robot_brain_system.utils.metric_utils import (
    with_metrics,
    get_total_gpu_memory_allocated_mb,
)
from robot_brain_system.utils.retry_utils import retry

# --- 依赖导入，带错误处理 ---
try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoProcessor,
    )

    # 显式导入 Qwen2.5-VL 的模型类，确保逻辑完整
    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers.models.glm4v import Glm4vForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    from openai import APIError, RateLimitError, AuthenticationError

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from lmdeploy import (
        pipeline,
        TurbomindEngineConfig,
        ChatTemplateConfig,
        GenerationConfig,
    )
    from lmdeploy.vl.constants import IMAGE_TOKEN

    LMDEPLOY_AVAILABLE = True
except ImportError:
    LMDEPLOY_AVAILABLE = False
    IMAGE_TOKEN = "<image>"

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# --- 抽象基类 ---
class BaseModelAdapter(ABC):
    @abstractmethod
    def generate(
        self,
        history: List[Dict[str, Any]],
        max_tokens: int = 2048,
        **kwargs,
    ) -> Tuple[str, Any]:
        pass


# --- 四种适配器实现（修正版） ---


class TransformersAdapter(BaseModelAdapter):
    """
    基于 Hugging Face `transformers` 库的适配器。
    """

    def __init__(self, model_path: str, device: str = "auto"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "`transformers` 和 `qwen_vl_utils` 是使用 TransformersAdapter 的必要依赖。"
            )

        self.model_path = model_path
        self.device = device

        # FIX: 修正了模型加载逻辑，使用 config 判断并加载正确的模型类
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "").lower()
        print(model_type)
        _initial_total_memory_mb = get_total_gpu_memory_allocated_mb()
        if "qwen2_5_vl" in model_type:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="float16",
                device_map=device,
                trust_remote_code=True,
            )
            # limit pic token from 256 to 1280
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                min_pixels=64 * 28 * 28,  # 224^2
                max_pixels=1280 * 28 * 28,  # 1000^2
            )
            self._handler = self._generate_qwen
            print("[TransformersAdapter] 已加载 Qwen2.5-VL 模型。")
        elif "glm4v" in model_type:
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device
            )
            self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
            self._handler = self._generate_generic

            print("[TransformersAdapter] 已加载 GLM-4V 模型。")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
            self._handler = self._generate_generic
            print(f"[TransformersAdapter] 已加载通用模型: {model_type}。")
        _loaded_total_memory_mb = get_total_gpu_memory_allocated_mb()
        _cost_mb = _loaded_total_memory_mb - _initial_total_memory_mb
        print(
            f"[TransformersAdapter] 模型已通过 `device_map='{device}'` 分布到可用设备。"
        )
        print(f"[TransformersAdapter] 模型加载占用总显存: {_cost_mb:.2f} MB")

    def _generate_qwen(
        self, history: List[Dict[str, Any]], gen_kwargs: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        text_prompt = self.processor.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        vision_outputs = process_vision_info(history)
        images, videos = vision_outputs[0], vision_outputs[1]

        inputs = self.processor(
            text=[text_prompt],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        return generated_ids, inputs

    def _generate_generic(
        self, history: List[Dict[str, Any]], gen_kwargs: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        inputs = self.processor.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            do_sample_frames=True,
        ).to(self.model.device)
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        return generated_ids, inputs

    @with_metrics(metrics=["time", "gpu_memory"])
    def generate(
        self, history: List[Dict[str, Any]], max_tokens: int = 2048, **kwargs
    ) -> Tuple[str, Any]:
        print(f"\n\n{'=' * 20}input{'=' * 20}")
        print(f"{history}")
        _initial_memory = torch.cuda.memory_allocated()
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            # "top_k": 2,
            # "repetition_penalty": 1.0,
            # "temperature": 0.1,
            # "do_sample": True,
        }
        gen_kwargs.update(kwargs)

        generated_ids, inputs = self._handler(history, gen_kwargs)

        input_token_len = inputs.input_ids.shape[1]
        response_ids = generated_ids[0, input_token_len:]
        response = self.processor.decode(response_ids, skip_special_tokens=True)

        print(f"\n\n{'=' * 20}response{'=' * 20}")
        print(f"{response.strip()}")

        return response.strip(), generated_ids


# --- MODIFIED VLLMAdapter ---
class VLLMAdapter(BaseModelAdapter):
    """
    基于 `vLLM` 的高效推理适配器。
    该版本对 Qwen 系列模型进行了特殊处理，以获得最佳性能和兼容性。
    """

    def __init__(self, model_path: str, **vllm_kwargs):
        if not VLLM_AVAILABLE:
            raise ImportError("`vllm` 是使用 VLLMAdapter 的必要依赖。")
        vllm_kwargs.setdefault("trust_remote_code", True)
        self.llm = LLM(model=model_path, **vllm_kwargs)
        self.model_path = model_path

        # --- NEW: Qwen-specific setup ---
        self.is_qwen_model = "qwen" in model_path.lower()
        if self.is_qwen_model:
            print(
                "[VLLMAdapter] Qwen model detected. Initializing Qwen-specific processor."
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            # For Qwen, the processor's tokenizer is the source of truth
            self.tokenizer = self.processor.tokenizer
        else:
            print("[VLLMAdapter] Initializing with generic vLLM tokenizer.")
            self.processor = None
            self.tokenizer = self.llm.get_tokenizer()

    def _prepare_vllm_input(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将内部历史记录格式转换为 vLLM 所需的输入格式。
        对 Qwen 模型使用专门的处理流程。
        """
        # --- NEW: Qwen-specific input preparation ---
        if self.is_qwen_model:
            # For Qwen, we use its processor to create the prompt from the rich history format.
            # The `process_vision_info` utility then extracts the image/video data.
            # NOTE: Our internal format stores video as a list of frames (PIL.Image).
            # The `qwen_vl_utils.process_vision_info` expects a video path/URL.
            # To bridge this, we treat video frames as a sequence of images, which is a
            # robust and compatible approach for multi-image models like Qwen-VL.

            # Step 1: Transform our 'video' entries into multiple 'image' entries.
            transformed_history = []
            for message in history:
                new_content = []
                for item in message.get("content", []):
                    if item.get("type") == "video" and isinstance(
                        item.get("video"), list
                    ):
                        # Add a text hint that these are video frames
                        new_content.append(
                            {
                                "type": "text",
                                "text": "The following images are sequential frames from a video:",
                            }
                        )
                        for frame in item["video"]:
                            if isinstance(frame, Image.Image):
                                new_content.append({"type": "image", "image": frame})
                    else:
                        new_content.append(item)
                transformed_history.append(
                    {"role": message["role"], "content": new_content}
                )

            # Step 2: Use Qwen's tools to process the transformed history
            prompt = self.processor.apply_chat_template(
                transformed_history,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                transformed_history, return_video_kwargs=True
            )

            mm_data = {}
            if image_inputs:
                mm_data["image"] = image_inputs
            # video_inputs will be None due to our transformation, which is expected.
            if video_inputs:
                mm_data["video"] = video_inputs

            return {
                "prompt": prompt,
                "multi_modal_data": mm_data if mm_data else None,
                "mm_processor_kwargs": video_kwargs,
            }

        # --- Fallback to generic processing for other models ---
        else:
            images, processed_history = [], []
            for message in history:
                role, content = message["role"], message.get("content", [])
                new_content_parts = []
                for item in content:
                    if item["type"] == "text":
                        new_content_parts.append(item["text"])
                    elif item["type"] in ["image", "video"]:
                        # Flatten all frames from images or videos into a single list
                        frames = (
                            item.get("video")
                            if item.get("type") == "video"
                            else [item.get("image")]
                        )
                        if frames:
                            for frame in frames:
                                if frame:
                                    new_content_parts.append("<image>")
                                    images.append(frame)
                processed_history.append(
                    {"role": role, "content": "".join(new_content_parts)}
                )

            prompt = self.tokenizer.apply_chat_template(
                conversation=processed_history,
                tokenize=False,
                add_generation_prompt=True,
            )
            return {
                "prompt": prompt,
                "multi_modal_data": {"image": images} if images else None,
            }

    def generate(
        self, history: List[Dict[str, Any]], max_tokens: int = 2048, **kwargs
    ) -> Tuple[str, Any]:
        vllm_input = self._prepare_vllm_input(history)
        sampling_params = SamplingParams(max_tokens=max_tokens, **kwargs)

        # Unpack the prepared inputs for the generate call
        outputs = self.llm.generate(
            prompts=[vllm_input["prompt"]],
            sampling_params=sampling_params,
            multi_modal_data=vllm_input.get("multi_modal_data"),
            # Pass special kwargs if they exist (for Qwen video)
            mm_processor_kwargs=vllm_input.get("mm_processor_kwargs"),
        )
        response = outputs[0].outputs[0].text
        return response.strip(), outputs


class LMDeployAdapter(BaseModelAdapter):
    """基于 `lmdeploy` pipeline 的推理适配器（已修正数据转换逻辑）。"""

    def __init__(self, model_path: str, **pipeline_kwargs):
        if not LMDEPLOY_AVAILABLE:
            raise ImportError("`lmdeploy` 是使用 LMDeployAdapter 的必要依赖。")
        self.pipe = pipeline(
            model_path,
            backend_config=TurbomindEngineConfig(
                session_len=4096 * 2, device_num=2, dp=1, tp=2, **pipeline_kwargs
            ),
            chat_template_config=ChatTemplateConfig(model_name="internvl2_5"),
        )

    def convert_video_to_images(self, messages):
        """
        将消息中的video类型转换为image类型
        """
        converted_messages = []

        for message in messages:
            converted_message = message.copy()

            if "content" in message:
                converted_content = []

                for content_item in message["content"]:
                    if content_item.get("type") == "video":
                        # 提取视频帧
                        video_data = content_item["video"]
                        frames = (
                            video_data if isinstance(video_data, list) else [video_data]
                        )

                        # 添加文本说明
                        converted_content.append(
                            {
                                "type": "text",
                                "text": "The following is the observation of the video frame sequence of the current scene:",
                            }
                        )

                        # 将每一帧转换为图像
                        for i, frame in enumerate(frames):
                            # 可选：添加帧序号说明
                            converted_content.append(
                                {"type": "text", "text": f"Frame{i + 1}/{len(frames)}"}
                            )
                            converted_content.append(
                                {
                                    "type": "image_data",
                                    "image_data": {
                                        "data": frame,
                                        "max_dynamic_patch": 12,
                                    },
                                }
                            )
                    elif content_item.get("type") == "image":
                        # 保持图像类型但是修改为image_data
                        frame = content_item["image"]
                        converted_content.append(
                            {
                                "type": "image_data",
                                "image_data": {"data": frame, "max_dynamic_patch": 12},
                            }
                        )
                    else:
                        # 保持其他类型不变
                        converted_content.append(content_item)

                converted_message["content"] = converted_content

            converted_messages.append(converted_message)

        return converted_messages

    def generate(
        self, history: List[Dict[str, Any]], max_tokens: int = 2048, **kwargs
    ) -> Tuple[str, Any]:
        # FIX: lmdeploy 的 pipeline 可以直接处理标准 history 格式，无需复杂转换
        # 它会自动处理 content 列表中的文本和图像
        history = self.convert_video_to_images(history)
        print("-------- Input messages --------")
        print(history)
        gen_args = {
            "top_k": 0,
            "top_p": 0.8,
            "temperature": 0.8,
            "max_new_tokens": max_tokens,
            "do_sample": True,
        }
        gen_args.update(kwargs)
        gen_config = GenerationConfig(**gen_args)
        output = self.pipe(history, gen_config=gen_config)
        if hasattr(output, "text"):
            response = output.text  # type: ignore
        elif isinstance(output, list) and len(output) > 0:
            first_item = output[0]
            response = (
                first_item.text if hasattr(first_item, "text") else str(first_item)
            )  # type: ignore
        else:
            response = str(output)
        print(f"-------- Generated response --------\n{response.strip()}\n")
        return response.strip(), output


class OpenAIAdapter(BaseModelAdapter):
    """
    适配器，用于调用符合 OpenAI API 规范的模型服务（如 vLLM server）。
    该版本支持对视频进行两种不同的处理策略，可通过参数切换。
    """

    def __init__(
        self,
        model_name: str,
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
        video_conversion_strategy: str = "as_images",  # ## NEW: 添加策略切换参数
    ):
        """
        初始化 OpenAIAdapter。

        Args:
            model_name (str): 要调用的模型名称。
            api_key (str, optional): API 密钥. 默认为 "EMPTY".
            base_url (Optional[str], optional): API 的基础URL. 默认为 None.
            video_conversion_strategy (str, optional): 视频处理策略。
                - "as_images": 将视频帧作为一系列图像发送 (默认, 兼容性好)。
                - "as_video_url": 将整个视频文件作为 Base64 URL 发送 (需要vLLM等后端支持)。
                默认为 "as_images".
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("`openai` 是使用 OpenAIAdapter 的必要依赖。")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = self.client.models.list().data[0].id
        print(f"[OpenAIAdapter] 已连接到 OpenAI API: {self.model_name}")
        # ## NEW: 验证并设置策略
        if video_conversion_strategy not in ["as_images", "as_video_url"]:
            raise ValueError(
                "video_conversion_strategy 必须是 'as_images' 或 'as_video_url'"
            )
        self.video_conversion_strategy = video_conversion_strategy
        print(f"[OpenAIAdapter] 视频处理策略已设置为: {self.video_conversion_strategy}")

    def _data_to_base64_url(self, data_bytes: bytes, mime_type: str) -> str:
        """通用函数，将原始字节数据编码为 Base64 data URL。"""
        encoded_str = base64.b64encode(data_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_str}"

    def _image_to_base64_url(self, image: Image.Image) -> str:
        """将 PIL Image 对象转换为 JPEG Base64 data URL。"""
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return self._data_to_base64_url(buffered.getvalue(), "image/jpeg")

    def _convert_history_to_openai_input(
        self, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        将内部历史记录格式转换为 OpenAI API 所需的 `messages` 格式。
        ## MODIFIED: 此函数现在会根据 video_conversion_strategy 选择不同的视频处理方式。
        """
        openai_messages = []
        for message in history:
            role = message["role"]
            content = message.get("content", [])

            if role in ["system", "assistant"]:
                text_content = "".join(
                    [item["text"] for item in content if item["type"] == "text"]
                )
                openai_messages.append({"role": role, "content": text_content})
                continue

            if role == "user":
                openai_content_parts = []
                for item in content:
                    item_type = item.get("type")

                    if item_type == "text":
                        openai_content_parts.append(
                            {"type": "text", "text": item["text"]}
                        )

                    elif item_type == "image" and item.get("image"):
                        base64_url = self._image_to_base64_url(item["image"])
                        openai_content_parts.append(
                            {"type": "image_url", "image_url": {"url": base64_url}}
                        )

                    # ## MODIFIED: 根据策略处理视频 ##
                    elif item_type == "video":
                        # --- 策略1: 将视频帧作为图像序列 ---
                        if self.video_conversion_strategy == "as_images":
                            frames = item.get("video")
                            if frames and isinstance(frames, list):
                                openai_content_parts.append(
                                    {
                                        "type": "text",
                                        "text": "The following images are sequential frames from a video.",
                                    }
                                )
                                for frame in frames:
                                    if isinstance(frame, Image.Image):
                                        base64_url = self._image_to_base64_url(frame)
                                        openai_content_parts.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": base64_url},
                                            }
                                        )

                        # --- 策略2: 将整个视频作为 video_url (vLLM 扩展) ---
                        elif self.video_conversion_strategy == "as_video_url":
                            # 这种策略要求输入的是视频路径或URL，而不是帧
                            video_path = item.get("video_path")
                            video_url = item.get("video_url")

                            if video_path and os.path.exists(video_path):
                                with open(video_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                # 推断mime type，这里简化为mp4
                                final_url = self._data_to_base64_url(
                                    video_bytes, "video/mp4"
                                )
                                openai_content_parts.append(
                                    {
                                        "type": "video_url",
                                        "video_url": {"url": final_url},
                                    }
                                )
                            elif video_url:
                                # 如果直接提供了URL，则直接使用
                                openai_content_parts.append(
                                    {
                                        "type": "video_url",
                                        "video_url": {"url": video_url},
                                    }
                                )

                if openai_content_parts:
                    openai_messages.append(
                        {"role": role, "content": openai_content_parts}
                    )

        return openai_messages

    # --- NEW: 日志净化工具方法 ---
    def _sanitize_payload_for_logging(
        self, payload: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        创建一个payload的深拷贝，并用占位符替换其中大的多媒体数据，以便清晰地打印日志。
        """
        # 使用深拷贝以确保不修改原始payload
        sanitized_payload = copy.deepcopy(payload)

        for message in sanitized_payload:
            if isinstance(message.get("content"), list):
                # 统计图像帧数量
                new_content = []
                muilt_frames_begin = False
                nframes = 0
                for part in message["content"]:
                    # 如果是图像URL，替换其内容
                    if part.get("type") == "image_url" and "url" in part.get(
                        "image_url", {}
                    ):
                        muilt_frames_begin = True
                        nframes += 1
                    # 保持文本内容不变，但跳过视频帧的描述文本
                    else:
                        if part.get("type") == "text":
                            new_content.append(part)
                        else:
                            new_content.append(part)  # 保留其他所有部分

                        if muilt_frames_begin:
                            muilt_frames_begin = False
                            new_content.append(
                                {
                                    "type": "text",
                                    "text": f"<image: {nframes} frames>",
                                }
                            )
                            nframes = 0
                if nframes > 0:
                    new_content.append(
                        {
                            "type": "text",
                            "text": f"<image: {nframes} frames>",
                        }
                    )
                message["content"] = new_content
        return sanitized_payload

    @retry(
        max_attempts=3,
        delay_seconds=1.0,
        exceptions_to_retry=(APIError, RateLimitError, AuthenticationError),
    )
    @with_metrics(metrics=["time"])
    def generate(
        self, history: List[Dict[str, Any]], max_tokens: int = 2048, **kwargs
    ) -> Tuple[str, Any]:
        messages = self._convert_history_to_openai_input(history)
        sanitized_messages_for_log = self._sanitize_payload_for_logging(messages)
        print("\n--- [OpenAIAdapter] Sending Payload ---")
        try:
            json_str = json.dumps(
                sanitized_messages_for_log, indent=2, ensure_ascii=False
            )
            print(json_str.replace("\\n", "\n"))
        except TypeError:
            print(sanitized_messages_for_log)
        print("-------------------------------------\n")
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=messages, max_tokens=max_tokens, **kwargs
        )
        response = completion.choices[0].message.content or ""
        print(f"\n--- [OpenAIAdapter] Received Response ---\n{response.strip()}\n")
        return response.strip(), completion


def resize_image_by_short_side(image: Image.Image, target_size: int) -> Image.Image:
    """
    按照短边进行图像缩放

    Args:
        image: PIL图像对象
        target_size: 目标短边尺寸，如果为None则使用实例的target_size

    Returns:
        缩放后的PIL图像对象
    """
    width, height = image.size

    # 计算缩放比例（按短边）
    if width < height:
        # 宽度是短边
        scale = target_size / width
        new_width = target_size
        new_height = int(height * scale)
    else:
        # 高度是短边
        scale = target_size / height
        new_height = target_size
        new_width = int(width * scale)

    # 使用高质量的重采样方法
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image


def extract_frames_with_decord(
    video_path: str, frames_per_second: int = 1
) -> List[Image.Image]:
    import decord

    """Extracts frames using Decord, which is generally faster."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # decord.bridge.set_bridge('torch') # Optional: for PyTorch tensors
    vr = decord.VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    frame_interval = int(video_fps / frames_per_second) if video_fps > 0 else 1
    if frame_interval == 0:
        frame_interval = 1

    # More efficient way to get frames with decord
    frame_indices = list(range(0, len(vr), frame_interval))
    frames_data = vr.get_batch(
        frame_indices
    ).asnumpy()  # Get all frames at once as a NumPy array

    return [
        resize_image_by_short_side(Image.fromarray(frame_np), 256)
        for frame_np in frames_data
    ]


if __name__ == "__main__":
    from robot_brain_system.core.brain import BrainMemory

    """
    独立的测试入口，用于验证各个适配器是否能正常工作。
    """
    parser = argparse.ArgumentParser(description="独立的模型适配器测试脚本。")
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        choices=["transformers", "vllm", "lmdeploy", "openai"],
        help="要测试的适配器类型。",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="本地模型的路径 (transformers, vllm, lmdeploy需要)。",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="模型名称 (OpenAI需要)，例如 'gpt-4-vision-preview'。",
    )
    parser.add_argument(
        "--api_key", type=str, default="123456", help="API密钥 (OpenAI需要)。"
    )
    parser.add_argument(
        "--base_url", type=str, default="", help="API的基础URL (OpenAI可选)。"
    )

    args = parser.parse_args()

    # --- 1. 创建测试数据 ---
    print("正在创建用于测试的虚拟红色图片...")
    dummy_image = Image.open(
        "./logs/20250710_213819/1_0_monitor_press_button_input_5.png"
    )
    print("虚拟图片创建完毕。")

    video_frames = extract_frames_with_decord(
        "./thirdparty/sam2/notebooks/videos/bedroom.mp4", frames_per_second=1
    )
    # 测试用例1: 复杂的多轮对话，包含图片
    multi_turn_image_memory = BrainMemory()
    multi_turn_image_memory.add_system_prompt(
        "You are a helpful assistant for object detection."
    )
    multi_turn_image_memory.add_user_input(
        contents=[
            "Detect output bbox of the red box in the image. The format of output should be like {'bbox_2d': [x1, y1, x2, y2], 'label': 'red box'}.",
            dummy_image,
        ]
    )
    multi_turn_image_memory.add_user_input(
        contents=["that is one the red box? what is silver object?", dummy_image]
    )

    # 测试用例2: 视频作为图像帧序列
    video_as_images_memory = BrainMemory()
    video_as_images_memory.add_user_input(
        contents=[
            "Describe the content of this video based on its frames.",
            video_frames,
        ]
    )

    test_cases = []
    test_cases.append(
        {"name": "Multi-turn Image Test", "memory": multi_turn_image_memory}
    )
    test_cases.append(
        {"name": "Video as Image Frames Test", "memory": video_as_images_memory}
    )

    # --- 3. 初始化选择的适配器 ---
    adapter: Optional[BaseModelAdapter] = None
    print(f"\n正在初始化适配器: {args.adapter}...")
    try:
        if args.adapter == "transformers":
            adapter = TransformersAdapter(model_path=args.model_path)
        elif args.adapter == "vllm":
            adapter = VLLMAdapter(model_path=args.model_path)
        elif args.adapter == "lmdeploy":
            adapter = LMDeployAdapter(model_path=args.model_path)
        elif args.adapter == "openai":
            # ## MODIFIED: 使用新的 openai_video_strategy 参数
            adapter = OpenAIAdapter(
                model_name=args.model_name,
                api_key=args.api_key,
                base_url=args.base_url,
            )
        print("适配器初始化成功！")
    except Exception as e:
        print(f"\n--- ❌ 初始化错误 ---\n初始化适配器时发生错误: {e}")
        exit(1)

    # --- 4. 执行生成并打印结果 ---
    if adapter:
        print("\n--- 🚀 开始生成响应 ---")
        try:
            for test_case in test_cases:
                name = test_case["name"]
                memory = test_case["memory"]
                print(f"\n{'=' * 20} Running Test Case: {name} {'=' * 20}")

                response_text, raw_output = adapter.generate(
                    history=memory.history, max_tokens=256
                )

                print("\n--- ✅ 测试成功 ---")
                print(f"\n🤖 模型对 '{name}' 的响应:")
                print("-" * 20)
                print(response_text)
                print("-" * 20)

        except Exception as e:
            import traceback

            print(f"\n--- ❌ 生成错误 ---\n生成响应时发生错误: {e}")
            traceback.print_exc()
            exit(1)
