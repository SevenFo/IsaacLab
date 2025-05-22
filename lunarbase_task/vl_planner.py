import httpx
import json_repair
from openai import OpenAI
import os
import time
import re
import matplotlib.pyplot as plt
from model_adapters import QwenVLAdapter

def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output


def extract_list(text_output: str) -> list:
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, text_output, re.DOTALL)
    if not match:
        raise ValueError(f"No list format content found: {text_output}")

    items_pattern = r'"([^"]*)"'
    items_list = re.findall(items_pattern, match.group(1))

    if not items_list:
        items_pattern = r"'([^']*)'"
        items_list = re.findall(items_pattern, match.group(1))
        
    return items_list

class DexGraspVLAPlanner:
    def __init__(self,
                api_key: str = "EMPTY", 
                base_url: str = "http://localhost:8000/v1",
                model_name: str = None,
                model_type: str = "qwen_omni",  # 可选: "qwen_omni" 或 "openai",
                model_path: str = None,
                enable_stream: bool = False, # 是否开启流式输出
                ):
        self.model_type = model_type
        if model_type == "qwen_omni":
            assert False
            if enable_stream:
                self.model = QwenOmniStreamAdapter(model_path=model_path)
            else:
                self.model = QwenOmniAdapter(model_path=model_path)
        elif model_type == "openai":
            assert False
            self.model = OpenAIAdapter(api_key=api_key, base_url=base_url, model_name=model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.log_file = None
        self.image_dir = None


    def set_logging(self, log_file, image_dir):
        self.log_file = log_file
        self.image_dir = image_dir


    def request_task(self,
            task_name: str,
            image: str = None,
            instruction: str = None,
            instruction_audio_path: str = None,
            max_token: int = 218,
            customize_prompt: str = None,
    ) -> str:
        if task_name == "customize_task":
            assert customize_prompt, "need customize_prompt"
            prompt = customize_prompt 
        elif task_name == "classify_user_prompt_zero":
            prompt = (
                f"Analyze the following user prompt: {instruction}\n\n" if instruction_audio_path is None else "Analyze the USER PROMPT based on following rules:\n"
                ) + (
                f"User prompt types:\n"
                f"- Type I (return True): Action Type, Instruction to do sth.\n"
                f"Examples:\n"
                f"* Color-based: \"green objects\"\n"
                f"* Position-based: \"objects from the right\"\n"
                f"* Property-based: \"all cups\"\n"
                f"* Combination: \"the red cup on the left\"\n\n"
                f"* Abstrction: \"clear the table\", \"clean up\", \"remove everything\"\n\n"
                f"- Type II (return False): QA Type, Rrequest to answer some question\n"
                f"Examples: \"what is on the table\", \"what can you see\", \"please describe what you have seen\"\n\n"
                f"Please determine:\n"
                f"- Is this a Type I prompt? (True/False)\n"
                f"- Provide your reasoning\n\n"
                f"Please provide your response in the following JSON format:\n"
                f'{{\n'
                f'    "is_action": "True/False",\n'
                f'    "reason": "YOUR REASONING"\n'
                f'    "user_prompt": "RETELL THE USE PROMPT"\n'
                f'}}\n\n'
                f"Examples:\n"
                f'- "grab the green cup" -> {{"is_action": True, "reason": "I need to take action to grab the green cup", "user_prompt":"grab the green cup"}}\n'
                f'- "clear the table" -> {{"is_action": True, "reason": "I need to take action to clear the table", "user_prompt":"clear the table"}}\n'
                f'- "what is on the table" -> {{"is_action": False, "reason": "No action need to take, I just need answer the question of user", "user_prompt":"what is on the table"}}\n'
            )
        elif task_name == "response_with_audio":
            prompt = (
                f"Analyze the following user prompt: {instruction}\n\n" if instruction_audio_path is None else "Analyze the USER PROMPT based on following rules:\n"
                ) + (
                # f"Please answer the question in user prompt use SHORT and CLEAR sentences\n"
                f"Please answer the question in user prompt within 15 words\n"
            )
        elif task_name == "classify_user_prompt":
            prompt = (
                f"Analyze the following user prompt: {instruction}\n\n" if instruction_audio_path is None else "Analyze the USER PROMPT based on following rules:\n"
                ) + (
                f"User prompt types:\n"
                f"- Type I (return True): User prompts with any specific descriptions\n"
                f"Examples:\n"
                f"* Color-based: \"green objects\"\n"
                f"* Position-based: \"objects from the right\"\n"
                f"* Property-based: \"all cups\"\n"
                f"* Combination: \"the red cup on the left\"\n\n"
                f"- Type II (return False): Abstract prompts without any object descriptions\n"
                f"Examples: \"clear the table\", \"clean up\", \"remove everything\"\n\n"
                f"Please determine:\n"
                f"- Is this a Type I prompt? (True/False)\n"
                f"- Provide your reasoning\n\n"
                f"Return format:\n"
                f"True/False: your reasoning\n\n"
                f"Examples:\n"
                f"- \"grab the green cup\" -> True: Contains specific object (cup) and property (green)\n"
                f"- \"clear the table\" -> False: No specific object characteristics mentioned"
            )

        elif task_name == "decompose_user_prompt":
            prompt = (
                f"For user prompt: {instruction}\n" if instruction_audio_path is None else "Based on the USER PROMPT do:\n"
            ) + (
                f"Process:\n"
                f"1. Analyze the user prompt and image together:\n"
                f"- Match user prompt descriptions with visible objects in the image\n"
                f"- If a description (e.g., \"green objects\") matches multiple objects, include all matching objects\n"
                f"- Verify each mentioned object actually exists in the image\n\n"
                f"2. Based on the robot arm's position (right edge of the screen) and table layout\n"
                f"3. Determine the most efficient grasping sequence\n"
                f"4. Generate a reordered list of objects to grasp\n\n"
                f"Requirements:\n"
                f"- Only include objects mentioned in the original user prompt\n"
                f"- Keep position information for each object\n"
                f"- Return as a list, ordered by grasping sequence, NO OTHER TEXT, strictly follow the format\n\n"
                f"Expected output format:\n"
                f"[\"object with position 1\", \"object with position 2\", ...]"
            )

        elif task_name == "generate_instruction":
            prompt = (
                f"Analyze the current desktop layout and select the most suitable object to grasp, considering the following factors:\n\n"
            ) + (
                f"Grasping Strategy:\n"
                f"1. The robotic arm is positioned on the far right (outside the frame)\n"
                f"2. Grasping Priority Order:\n"
                f"   - Prioritize objects on the right to avoid knocking over other objects during later operations\n"
                f"   - Then consider objects in the middle\n"
                f"   - Finally, consider objects on the left\n"
                f"3. Accessibility Analysis:\n"
                f"   - Relative positions between objects\n"
                f"   - Potential obstacles\n"
                f"   - Whether the grasping path might interfere with other objects\n\n"
                f"Please provide your response in the following JSON format:\n"
                f"{{\n"
                f"    \"analysis\": {{\n"
                f"        \"priority_consideration\": \"explanation of why this object has priority\",\n"
                f"        \"accessibility\": \"analysis of object's accessibility\",\n"
                f"        \"risk_assessment\": \"potential risks in grasping this object\"\n"
                f"    }},\n"
                f"    \"target\": \"a comprehensive description of the target object (e.g., 'the blue cube on the far right of the desktop, next to the red cylinder')\"\n"
                f"}}\n\n"
                f"Ensure the output is in valid JSON format.\n"
                f"Note: The 'target' field should ONLY contain the object's color, shape, and position in a natural, flowing sentence. Do not include any analysis or reasoning in this field."
            )

        elif task_name == "mark_bounding_box":
            prompt = (
                f"Analyze the image and identify the best matching object with the description: {instruction}.\n" if instruction_audio_path is None else "Analyze the image and identify the best matching object with the the USER PROMPT, based on following rules:\n"
            ) + (
                f"Instructions for object analysis:\n"
                f"1. Select ONE object that best matches the description\n"
                f"2. For the selected object, provide:\n"
                f"- A concise label, object name (3-4 words max)\n"
                f"- A detailed description (position, color, shape, context)\n"
                f"- Accurate bbox coordinates\n\n"
                f"Required JSON format with an example:\n"
                f"```json\n"
                f"{{\n"
                f"    \"bbox_2d\": [x1, y1, x2, y2],\n"
                f"    \"label\": \"green cup\",  # Keep this very brief (3-4 words)\n"
                f"    \"description\": \"a cylindrical green ceramic cup located on the right side of the wooden table, next to the laptop\"  # Detailed description\n"
                f"}}\n"
                f"```\n\n"
                f"Critical requirements:\n"
                f"- Return EXACTLY ONE object\n"
                f"- \"label\": Must be brief (3-4 words)\n"
                f"- \"description\": Must be detailed and include spatial context\n"
                f"- Use single JSON object format, not an array\n"
                f"- Ensure bbox coordinates are within image boundaries"
            )

        elif task_name == "check_grasp_success":
            prompt = (
                f"Briefly analyze the image and determine if the robotic arm has successfully grasped an object:\n"
                f"1. Consider the spatial relationship between the robotic hand and the object\n"
                f"2. Output format: explain your reasoning shortly and precisely, then conclude with a boolean value (True=grasped, False=not grasped)\n"
                f"Keep it short and simple."
            )
        
        elif task_name == "check_instruction_complete":  # TODO: check whether the prompt makes sense.
            prompt = (
                f"Please check whether {instruction} exists on the desktop. If it does not exist, output True; otherwise, output False." if instruction_audio_path is None else f"Please check whether USER PROMPT exists on the desktop. If it does not exist, output True; otherwise, output False." 
            )
        
        elif task_name == "check_user_prompt_complete":
            prompt = (
                f"Please analyze the table in the image:\n\n"
                f"Requirements:\n"
                f"- Only detect physical objects with noticeable height/thickness (3D objects)\n"
                f"- Exclude from consideration:\n"
                f"* Flat items (papers, tablecloths, mats)\n"
                f"* Light projections\n"
                f"* Shadows\n"
                f"* Surface patterns or textures\n\n"
                f"Return format:\n"
                f"- True: if the table is empty of 3D objects\n"
                f"- False: if there are any 3D objects, followed by their names\n\n"
                f"Example responses:\n"
                f"True  (for empty table)\n"
                f"False: cup, bottle, plate  (for table with objects)"
            )

        else:
            raise ValueError(f"The task_name {task_name} is not a valid task name.")

        if instruction_audio_path is not None:
            prompt += "\nReply using the language of User Prompt. For example, if the User Prompt is Chinese, reply in Chinese; if the User Prompt is English, reply in English. If neither Chinese nor English, just use English.\n\nNext, Attention! You'll hear the USER PROMPT."
            
        input_data = self.model.prepare_input(
            text = prompt,
            audio_path=instruction_audio_path,
            image_url=image
        )
        
        if image is not None:
            # self.save_image(frame_path, task_name)
            pass

        self.log(f"Planner requesting task: {task_name}.")
        self.log(f"Planner prompt:\n{prompt}")

        return_audio = True if task_name == 'response_with_audio' else False # 固定只有 `response_with_audio` 任务才返回音频
        
        response, response_audio = self.model.generate_response(input_data, max_token, return_audio=return_audio)

        response_lower = response.lower()

        self.log(f"Planner response:\n{response}")

        if task_name == "classify_user_prompt":
            if 'true' in response_lower:
                return "TypeI"
            elif 'false' in response_lower:
                return "TypeII"
            else:
                raise ValueError(f"The output text {response} is in the wrong format.")
        elif task_name == "decompose_user_prompt":
            list_output = extract_list(response)
            if type(list_output) == list:
                return list_output
            else:
                raise ValueError(f"The output text {list_output} is not a valid list.")
        elif task_name == "generate_instruction":
            generate_task_str = parse_json(response)
            generate_task_json = json_repair.loads(generate_task_str)
            generate_task = generate_task_json['target']
            if type(generate_task) == str:
                return generate_task
            else:
                raise ValueError(f"The output text {generate_task} is not a valid string.")
        elif task_name == "mark_bounding_box":
            bbox_str = parse_json(response)
            bbox_json = json_repair.loads(bbox_str)
            return bbox_json
        elif task_name == 'customize_task':
            return response
        elif task_name == 'classify_user_prompt_zero':
            generate_str = parse_json(response)
            generate_json = json_repair.loads(generate_str)
            is_action = generate_json.get("is_action", False)
            if(type(is_action) is not bool and type(bool) is str):
                is_action = ('true' in is_action.lower())
            reason = generate_json.get("reason", "")
            user_prompt = generate_json.get("user_prompt", "")
            return is_action, user_prompt if len(user_prompt) else reason
        elif task_name == 'response_with_audio':
            assert response_audio is not None, "no audio generated!"
            return response
        else:
            if 'true' in response_lower:
                return True
            elif 'false' in response_lower:
                return False
            else:
                raise ValueError(f"The output text {response} does not contain a valid boolean value.")

    
    def log(self, message):
        log(message, self.log_file)

    
    def save_image(self, image_url, task_name):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image = decode_base64_to_image(image_url)
        image_path = os.path.join(self.image_dir, f"{timestamp}_planner_request_{task_name}.png")
        plt.imsave(image_path, image)
        
        
if __name__ == "__main__":
    import yaml
    from PIL import Image
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    planner = DexGraspVLAPlanner(
        model_path=config["planner"]["model_path"],model_type="qwen_omni",enable_stream=True)
    
    instructions = [
        'clean the table_zh.wav',
        'clean the table.wav',
        'what_is_on_the_disk_zh.wav'
    ]
    
    image = Image.open("croped_head_image.png")
    
    task = 'classify_user_prompt_zero' # response_with_audio
    for ins in instructions:
        isaction, reason = planner.request_task(task,image=image,instruction=None,instruction_audio_path = ins)

        if isaction:
            print(f"action prompt: {reason}")
            object_list = planner.request_task('decompose_user_prompt',image=image,instruction=reason)
            print(f"object_list: {object_list}")
        else:
            print("QA prompt: {reason}")
            response_text = planner.request_task('response_with_audio',image=image,instruction=reason)
            print(f"response: {response_text}")