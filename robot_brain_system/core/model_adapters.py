"""
Model adapters for different AI models (Qwen VL, OpenAI, etc.)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Any
from PIL import Image
import torch

try:
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.models.auto.modeling_auto import AutoModel
    from transformers.models.auto.processing_auto import AutoProcessor
    from transformers.models.qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available")

try:
    from qwen_vl_utils import process_vision_info

    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    print("Warning: qwen_vl_utils not available")

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not available")

from lmdeploy import (
    pipeline,
    TurbomindEngineConfig,
    ChatTemplateConfig,
    GenerationConfig,
)
from lmdeploy.vl.constants import IMAGE_TOKEN

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vllm library not available")


class BaseModelAdapter(ABC):
    """Base class for model adapters."""

    @abstractmethod
    def prepare_input(
        self,
        text: str,
        audio_path: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Prepare input for the model."""
        pass

    @abstractmethod
    def generate_response(
        self, input_data: List[Dict[str, Any]], max_tokens: int = 512, **kwargs
    ) -> Tuple[str, Optional[str]]:
        """Generate response from the model.

        Args:
            input_data: processed data returned by `self.prepare_input`
            max_tokens: maximum number of tokens to generate

        Returns:
            tuple: [generated text, generated audio path (None means no audio generated)]
        """
        pass


class QwenVLAdapter(BaseModelAdapter):
    """Adapter for Qwen VL model."""

    def __init__(self, model_path: str, device_map: str):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for QwenVLAdapter")
        if not QWEN_VL_UTILS_AVAILABLE:
            raise ImportError("qwen_vl_utils is required for QwenVLAdapter")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            device_map=device_map,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def prepare_input(
        self,
        text: str,
        audio_path: Optional[str] = None,
        image: Optional[List[Image.Image]] = None,
    ) -> List[Dict[str, Any]]:
        """构建Qwen-VL的输入格式"""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]

        # 添加图像输入
        if image:
            if not isinstance(image, list):
                image = [image]
            for img in image:
                messages[1]["content"].append(
                    {
                        "type": "image",
                        "image": img,  # 假设image_url是base64编码或本地路径
                    }
                )

        return messages

    def generate_response(
        self, input_data: List[Dict[str, Any]], max_tokens: int = 512, **kwargs
    ) -> Tuple[str, Optional[str]]:
        """生成文本响应"""
        messages = input_data

        # 处理多模态输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 分离视觉信息 - 注意process_vision_info返回3个值
        vision_outputs = process_vision_info(messages)
        if len(vision_outputs) == 3:
            image_inputs, video_inputs, _ = vision_outputs
        else:
            image_inputs, video_inputs = vision_outputs[:2]
        print(
            f"-------- Processed vision inputs --------\n{image_inputs}\n{video_inputs}\n"
        )
        # 准备模型输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 生成响应
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_tokens, **kwargs
        )

        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print(f"-------- Generated response --------\n{response}\n")
        return response, None  # Qwen-VL不支持音频输出


class InternVLAdapter(BaseModelAdapter):
    """Adapter for InternVL model."""

    def __init__(self, model_path: str):
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()


class LMDAdapter(BaseModelAdapter):
    """Adapter for LMD."""

    def __init__(self, model_path: str, target_size: int = 224):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for QwenVLAdapter_LMD")
        if not QWEN_VL_UTILS_AVAILABLE:
            raise ImportError("qwen_vl_utils is required for QwenVLAdapter_LMD")

        self.target_size = target_size  # 目标短边尺寸
        self.pipe = pipeline(
            model_path,
            backend_config=TurbomindEngineConfig(
                session_len=4096 * 2, device_num=2, dp=1, tp=2
            ),
            chat_template_config=ChatTemplateConfig(model_name="internvl2_5"),
        )

    def resize_image_by_short_side(
        self, image: Image.Image, target_size: Optional[int] = None
    ) -> Image.Image:
        """
        按照短边进行图像缩放

        Args:
            image: PIL图像对象
            target_size: 目标短边尺寸，如果为None则使用实例的target_size

        Returns:
            缩放后的PIL图像对象
        """
        if target_size is None:
            target_size = self.target_size

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

    def prepare_input(
        self,
        text: str,
        audio_path: Optional[str] = None,
        image: Optional[List[Image.Image]] = None,
    ) -> List[Dict[str, Any]]:
        """构建Qwen-VL的输入格式"""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": ""}]},
        ]

        # 添加图像输入
        if image:
            if not isinstance(image, list):
                image = [image]
            for frame_idx, img in enumerate(image):
                # 对图像进行resize处理
                messages[1]["content"][0]["text"] += (
                    f"Frame{frame_idx + 1}: {IMAGE_TOKEN}\n"
                )
                messages[1]["content"].append(
                    {
                        "type": "image_data",
                        "image_data": {"data": img, "max_dynamic_patch": 1},
                    }
                )

        messages[1]["content"][0]["text"] += text

        return messages

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
                        # 保持图像类型不变
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

    def generate_response(
        self, input_data: List[Dict[str, Any]], max_tokens: int = 512, **kwargs
    ) -> Tuple[str, Optional[str]]:
        """生成文本响应"""
        messages = self.convert_video_to_images(input_data)
        print("-------- Input messages --------")
        print(messages)
        out = self.pipe(
            messages,
            gen_config=GenerationConfig(
                top_k=0,
                top_p=0.8,
                temperature=0.8,
                max_new_tokens=max_tokens,
                do_sample=True,
            ),
        )

        # 处理 lmdeploy 的返回结果
        if hasattr(out, "text"):
            response = out.text  # type: ignore
        elif isinstance(out, list) and len(out) > 0:
            first_item = out[0]
            response = (
                first_item.text if hasattr(first_item, "text") else str(first_item)
            )  # type: ignore
        else:
            response = str(out)

        print(f"-------- Generated response --------\n{response}\n")
        return response, None  # Qwen-VL不支持音频输出


class VLLMAdapter(BaseModelAdapter):
    """
    用于 VLLM 多模态推理的适配器，兼容对话历史。
    """

    def __init__(self, model_path: str, **kwargs):
        """
        初始化 VLLM 适配器。

        Args:
            model_path (str): 模型在 Hugging Face Hub 或本地的路径。
            **kwargs: 传递给 vllm.LLM 构造函数的额外参数，
                      例如 tensor_parallel_size=2, trust_remote_code=True。
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vllm library is required for VLLMAdapter")

        # 允许通过 kwargs 传递 trust_remote_code 是一个好习惯
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        self.llm = LLM(model=model_path, trust_remote_code=trust_remote_code, **kwargs)
        # 获取分词器，用于后续的模板应用
        self.tokenizer = self.llm.get_tokenizer()

    def _convert_history_to_vllm_input(
        self, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        将 BrainMemory 中的对话历史转换为 vLLM 的 generate 方法可以理解的格式。
        """
        # 使用 vLLM 的分词器 `apply_chat_template` 是格式化 prompt 最可靠的方法。
        # 但我们需要先将图像数据分离出来。
        images = []
        processed_history = []

        for message in history:
            role = message["role"]
            content = message["content"]

            # 重建 content，用占位符替换图像
            new_content_parts = []
            for item in content:
                if item["type"] == "text":
                    new_content_parts.append(item["text"])
                elif item["type"] == "image":
                    # 为图像创建一个占位符，例如 <image_1>, <image_2>
                    image_placeholder = f"<image_{len(images) + 1}>"
                    new_content_parts.append(image_placeholder)
                    images.append(item["image"])
                elif item["type"] == "video":
                    # 对于视频，将每一帧作为单独的图像处理
                    for frame in item["video"]:
                        image_placeholder = f"<image_{len(images) + 1}>"
                        new_content_parts.append(image_placeholder)
                        images.append(frame)

            # 将该消息的所有文本部分连接起来
            full_text_content = "\n".join(new_content_parts)
            processed_history.append({"role": role, "content": full_text_content})

        # 使用分词器应用聊天模板，生成最终的、包含所有特殊token的prompt
        prompt = self.tokenizer.apply_chat_template(
            conversation=processed_history,
            tokenize=False,
            add_generation_prompt=True,
        )

        multi_modal_data = {"image": images} if images else None

        return {"prompt": prompt, "multi_modal_data": multi_modal_data}

    def prepare_input(
        self,
        text: str,
        audio_path: Optional[str] = None,
        image: Optional[List[Image.Image]] = None,
    ) -> List[Dict[str, Any]]:
        """
        以对话历史的格式准备单轮输入。
        这确保了与 Brain 的内存结构兼容。
        """
        # 为单次查询创建一个简单的一轮对话历史
        history = [{"role": "user", "content": []}]

        # 首先添加图像
        if image:
            if not isinstance(image, list):
                image = [image]
            for img in image:
                history[0]["content"].append({"type": "image", "image": img})

        # 然后添加文本
        history[0]["content"].append({"type": "text", "text": text})

        return history

    def generate_response(
        self, input_data: List[Dict[str, Any]], max_tokens: int = 512, **kwargs
    ) -> Tuple[str, Optional[str]]:
        """
        使用 vLLM 引擎从对话历史中生成响应。

        Args:
            input_data: 从 BrainMemory 获取的对话历史列表。
            max_tokens: 最大生成 token 数。
            **kwargs: 传递给 vllm.SamplingParams 的额外参数。
        """
        # 1. 将对话历史 (input_data) 转换为 vLLM 格式
        vllm_input = self._convert_history_to_vllm_input(input_data)

        # 2. 设置采样参数
        sampling_params = SamplingParams(max_tokens=max_tokens, **kwargs)

        # 3. 生成响应
        # vLLM 的 generate 方法可以批量处理，但我们这里只处理单个 prompt
        outputs = self.llm.generate(
            prompts=vllm_input["prompt"],
            sampling_params=sampling_params,
            multi_modal_data=vllm_input.get("multi_modal_data"),
        )

        # 4. 提取并返回生成的文本
        response = outputs[0].outputs[0].text.strip()
        print(f"-------- Generated response --------\n{response}\n")

        return response, None


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for OpenAI models."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library is required for OpenAIAdapter")

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = (
            self.client.models.list().data[0].id if model_name is None else model_name
        )

    def prepare_input(
        self,
        text: str,
        audio_path: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Prepare input for OpenAI models."""
        # OpenAI 不支持音频输入，暂时忽略
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        if image_url is not None:
            messages[1]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

        return messages

    def generate_response(
        self, input_data: List[Dict[str, Any]], max_tokens: int = 512, **kwargs
    ) -> Tuple[str, Optional[str]]:
        """Generate response using OpenAI API."""
        # Convert to the proper format for OpenAI API
        messages = []
        for msg in input_data:
            if msg["role"] == "system":
                content = ""
                for item in msg["content"]:
                    if item["type"] == "text":
                        content += item["text"]
                messages.append({"role": "system", "content": content})
            elif msg["role"] == "user":
                content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        content.append(item)
                messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content, None
