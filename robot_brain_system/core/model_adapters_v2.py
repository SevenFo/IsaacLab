# -*- coding: utf-8 -*-
"""
统一的模型适配器模块（已修正版本），为不同推理后端提供标准化接口。
"""

import base64
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import argparse

import torch
from PIL import Image

# --- 依赖导入，带错误处理 ---
try:
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                              Glm4vForConditionalGeneration)
    # 显式导入 Qwen2.5-VL 的模型类，确保逻辑完整
    from transformers.models.qwen2_5_vl import \
        Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    class AutoProcessor: pass
    class Glm4vForConditionalGeneration: pass
    class Qwen2_5_VLForConditionalGeneration: pass
    def process_vision_info(*args, **kwargs): return [], []

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from lmdeploy import GenerationConfig, pipeline
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
    基于 Hugging Face `transformers` 库的适配器（已修正模型加载和解码逻辑）。
    """
    def __init__(self, model_path: str, device: str = "auto"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("`transformers` 和 `qwen_vl_utils` 是使用 TransformersAdapter 的必要依赖。")

        self.model_path = model_path
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        
        # FIX: 修正了模型加载逻辑，使用 config 判断并加载正确的模型类
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "").lower()
        print(model_type)

        if "qwen2_5_vl" in model_type:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
            self._handler = self._generate_qwen
            print("[TransformersAdapter] 已加载 Qwen2.5-VL 模型。")
        elif "glm4v" in model_type:
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device
            )
            self._handler = self._generate_generic
            print("[TransformersAdapter] 已加载 GLM-4V 模型。")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
            self._handler = self._generate_generic
            print(f"[TransformersAdapter] 已加载通用模型: {model_type}。")

    def _generate_qwen(self, history: List[Dict[str, Any]], gen_kwargs: Dict) -> Tuple[torch.Tensor, Dict]:
        text_prompt = self.processor.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        vision_outputs = process_vision_info(history)
        images, videos = (vision_outputs[0], vision_outputs[1]) if len(vision_outputs) >= 2 else ([], [])

        inputs = self.processor(
            text=[text_prompt], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        return generated_ids, inputs

    def _generate_generic(self, history: List[Dict[str, Any]], gen_kwargs: Dict) -> Tuple[torch.Tensor, Dict]:
        inputs = self.processor.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            do_sample_frames=True
        ).to(self.model.device)
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        return generated_ids, inputs

    def generate(
        self, history: List[Dict[str, Any]], max_tokens: int = 2048, **kwargs
    ) -> Tuple[str, Any]:
        
        print(f'\n\n{"="*20}input{"="*20}')
        print(f"{history}")
        
        gen_kwargs = {"max_new_tokens": max_tokens,"top_k":2, "repetition_penalty": 1.0, "temperature": 1.0, "do_sample": True}
        gen_kwargs.update(kwargs)
        
        # FIX: _handler 现在返回 (generated_ids, inputs) 以确保 input_token_len 准确
        generated_ids, inputs = self._handler(history, gen_kwargs)
        
        input_token_len = inputs.input_ids.shape[1]
        response_ids = generated_ids[0, input_token_len:]
        response = self.processor.decode(response_ids, skip_special_tokens=True)
        
        print(f'\n\n{"="*20}response{"="*20}')
        print(f"{response.strip()}")
        
        return response.strip(), generated_ids

class VLLMAdapter(BaseModelAdapter):
    """基于 `vLLM` 的高效推理适配器。"""
    def __init__(self, model_path: str, **vllm_kwargs):
        if not VLLM_AVAILABLE:
            raise ImportError("`vllm` 是使用 VLLMAdapter 的必要依赖。")
        vllm_kwargs.setdefault("trust_remote_code", True)
        self.llm = LLM(model=model_path, **vllm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _convert_history_to_vllm_input(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        images, processed_history = [], []
        for message in history:
            role, content = message["role"], message.get("content", [])
            new_content_parts = []
            for item in content:
                if item["type"] == "text":
                    new_content_parts.append(item["text"])
                elif item["type"] in ["image", "video"]:
                    frames = item.get("video", [item.get("image")])
                    for frame in frames:
                        if frame:
                            new_content_parts.append("<image>")
                            images.append(frame)
            processed_history.append({"role": role, "content": "".join(new_content_parts)})

        prompt = self.tokenizer.apply_chat_template(
            conversation=processed_history, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "multi_modal_data": {"image": images} if images else None}

    def generate(self, history: List[Dict[str, Any]], max_tokens: int = 2048, **kwargs) -> Tuple[str, Any]:
        vllm_input = self._convert_history_to_vllm_input(history)
        sampling_params = SamplingParams(max_tokens=max_tokens, **kwargs)
        outputs = self.llm.generate(
            prompts=[vllm_input["prompt"]],
            sampling_params=sampling_params,
            multi_modal_data=vllm_input.get("multi_modal_data"),
        )
        response = outputs[0].outputs[0].text
        return response.strip(), outputs

class LMDeployAdapter(BaseModelAdapter):
    """基于 `lmdeploy` pipeline 的推理适配器（已修正数据转换逻辑）。"""
    def __init__(self, model_path: str, **pipeline_kwargs):
        if not LMDEPLOY_AVAILABLE:
            raise ImportError("`lmdeploy` 是使用 LMDeployAdapter 的必要依赖。")
        self.pipe = pipeline(model_path, **pipeline_kwargs)

    def generate(self, history: List[Dict[str, Any]], max_tokens: int = 2048, **kwargs) -> Tuple[str, Any]:
        # FIX: lmdeploy 的 pipeline 可以直接处理标准 history 格式，无需复杂转换
        # 它会自动处理 content 列表中的文本和图像
        gen_config = GenerationConfig(max_new_tokens=max_tokens, **kwargs)
        output = self.pipe(history, gen_config=gen_config)
        response = output.text if hasattr(output, 'text') else str(output)
        return response.strip(), output

class OpenAIAdapter(BaseModelAdapter):
    """适配器，用于调用符合 OpenAI API 规范的模型服务（已优化图像处理）。"""
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("`openai` 是使用 OpenAIAdapter 的必要依赖。")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _image_to_base64_url(self, image: Image.Image) -> str:
        buffered = BytesIO()
        # FIX: 转换为RGB以处理透明通道，增加代码健壮性
        image.convert("RGB").save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _convert_history_to_openai_input(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        openai_messages = []
        for message in history:
            role, content = message["role"], message.get("content", [])
            if role in ["system", "assistant"]:
                text_content = "".join([item["text"] for item in content if item["type"] == "text"])
                openai_messages.append({"role": role, "content": text_content})
            elif role == "user":
                openai_content = []
                for item in content:
                    if item["type"] == "text":
                        openai_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] in ["image", "video"]:
                        frames = item.get("video", [item.get("image")])
                        for frame in frames:
                            if frame:
                                base64_url = self._image_to_base64_url(frame)
                                openai_content.append({"type": "image_url", "image_url": {"url": base64_url}})
                openai_messages.append({"role": role, "content": openai_content})
        return openai_messages

    def generate(self, history: List[Dict[str, Any]], max_tokens: int = 2048, **kwargs) -> Tuple[str, Any]:
        messages = self._convert_history_to_openai_input(history)
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=messages, max_tokens=max_tokens, **kwargs
        )
        response = completion.choices[0].message.content
        return response.strip(), completion
    

if __name__ == '__main__':
    """
    独立的测试入口，用于验证各个适配器是否能正常工作。
    """
    parser = argparse.ArgumentParser(description="独立的模型适配器测试脚本。")
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        choices=["transformers", "vllm", "lmdeploy", "openai"],
        help="要测试的适配器类型。"
    )
    parser.add_argument("--model_path", type=str, help="本地模型的路径 (transformers, vllm, lmdeploy需要)。")
    parser.add_argument("--model_name", type=str, default="", help="模型名称 (OpenAI需要)，例如 'gpt-4-vision-preview'。")
    parser.add_argument("--api_key", type=str, default="", help="API密钥 (OpenAI需要)。")
    parser.add_argument("--base_url", type=str,default="", help="API的基础URL (OpenAI可选)。")

    args = parser.parse_args()

    # --- 1. 创建测试数据 ---
    print("正在创建用于测试的虚拟红色图片...")
    dummy_image = Image.open('/data/shiqi/IsaacLab/logs/20250710_110850/1_0_monitor_press_button_input_5.png')
    print("虚拟图片创建完毕。")

    test_history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Detect output bbox of red box and yellow button， The format of output should be like {“bbox_2d”: [x1, y1, x2, y2], “label”: “motorcyclist”, “sub_label”: “wearing helmat” # or “not wearing helmat”}."},
                {"type": "image", "image": dummy_image}
            ]
        }
    ]
    print("测试对话历史已准备好。")

    # --- 2. 初始化选择的适配器 ---
    adapter: Optional[BaseModelAdapter] = None
    print(f"\n正在初始化适配器: {args.adapter}...")
    try:
        if args.adapter == "transformers":
            if not args.model_path: raise ValueError("--model_path 是 transformers 适配器所必需的。")
            adapter = TransformersAdapter(model_path=args.model_path)
        elif args.adapter == "vllm":
            if not args.model_path: raise ValueError("--model_path 是 vllm 适配器所必需的。")
            adapter = VLLMAdapter(model_path=args.model_path)
        elif args.adapter == "lmdeploy":
            if not args.model_path: raise ValueError("--model_path 是 lmdeploy 适配器所必需的。")
            adapter = LMDeployAdapter(model_path=args.model_path)
        elif args.adapter == "openai":
            if not args.model_name or not args.api_key:
                raise ValueError("--model_name 和 --api_key 是 openai 适配器所必需的。")
            adapter = OpenAIAdapter(model_name=args.model_name, api_key=args.api_key, base_url=args.base_url)
        
        print("适配器初始化成功！")

    except Exception as e:
        print(f"\n--- ❌ 初始化错误 ---")
        print(f"初始化适配器时发生错误: {e}")
        exit(1)

    # --- 3. 执行生成并打印结果 ---
    if adapter:
        print("\n--- 🚀 开始生成响应 ---")
        try:
            response_text, raw_output = adapter.generate(history=test_history, max_tokens=2048)
            
            print("\n--- ✅ 测试成功 ---")
            print("\n🤖 模型响应:")
            print("="*20)
            print(response_text)
            print("="*20)

        except Exception as e:
            import traceback
            print(f"\n--- ❌ 生成错误 ---")
            print(f"生成响应时发生错误: {e}")
            traceback.print_exc()
            exit(1)