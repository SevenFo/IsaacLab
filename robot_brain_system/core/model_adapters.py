"""
Model adapters for different AI models (Qwen VL, OpenAI, etc.)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Any

try:
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

    def __init__(self, model_path: str):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for QwenVLAdapter"
            )
        if not QWEN_VL_UTILS_AVAILABLE:
            raise ImportError("qwen_vl_utils is required for QwenVLAdapter")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="bfloat16", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def prepare_input(
        self,
        text: str,
        audio_path: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """构建Qwen-VL的输入格式"""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]

        # 添加图像输入
        if image_url:
            messages[1]["content"].append(
                {
                    "type": "image",
                    "image": image_url,  # 假设image_url是base64编码或本地路径
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
            self.client.models.list().data[0].id
            if model_name is None
            else model_name
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
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ],
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
