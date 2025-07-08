from typing import Optional, Union, Dict, List
import json
import re
from json_repair import repair_json


def extract_json_from_text(
    text: str, repair: bool = True
) -> Optional[Union[Dict, List]]:
    """
    Helper to robustly extract and parse JSON from a text string.
    Uses json_repair for better malformed JSON handling.
    """
    if not text or not isinstance(text, str):
        return None

    stripped_text = text.strip()

    # Method 1: ```json ... ``` markdown block
    match = re.search(r"```json\s*([\s\S]*?)\s*```", stripped_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            # Try standard parsing first
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # If standard parsing fails, try repair
                repaired_json = repair_json(json_str)
                return json.loads(repaired_json)
            except Exception as e:
                print(f"[QwenVLBrain] Failed to repair JSON in ```json block: {e}")

    # Method 2: ``` ... ``` generic markdown block
    match = re.search(r"```\s*([\s\S]*?)\s*```", stripped_text, re.DOTALL)
    if match:
        potential_json_str = match.group(1).strip()
        # Check if it plausibly looks like JSON
        if (
            potential_json_str.startswith("{") and potential_json_str.endswith("}")
        ) or (potential_json_str.startswith("[") and potential_json_str.endswith("]")):
            try:
                return json.loads(potential_json_str)
            except json.JSONDecodeError:
                try:
                    repaired_json = repair_json(potential_json_str)
                    return json.loads(repaired_json)
                except Exception as e:
                    print(f"[QwenVLBrain] Failed to repair JSON in ``` block: {e}")

    # Method 3: JSON as the entire content
    if (stripped_text.startswith("{") and stripped_text.endswith("}")) or (
        stripped_text.startswith("[") and stripped_text.endswith("]")
    ):
        try:
            return json.loads(stripped_text)
        except json.JSONDecodeError:
            try:
                repaired_json = repair_json(stripped_text)
                return json.loads(repaired_json)
            except Exception as e:
                print(f"[QwenVLBrain] Failed to repair full JSON: {e}")

    # Method 4: Find and extract JSON using raw_decode first, then repair if needed
    decoder = json.JSONDecoder()
    try:
        first_brace = stripped_text.find("{")
        first_bracket = stripped_text.find("[")

        start_idx = -1
        if first_brace != -1 and first_bracket != -1:
            start_idx = min(first_brace, first_bracket)
        elif first_brace != -1:
            start_idx = first_brace
        elif first_bracket != -1:
            start_idx = first_bracket

        if start_idx != -1:
            try:
                obj, _ = decoder.raw_decode(stripped_text[start_idx:])
                if isinstance(obj, (dict, list)):
                    return obj
            except json.JSONDecodeError:
                # If raw_decode fails, try to repair the extracted part
                try:
                    # Extract the JSON-like part and attempt repair
                    json_part = stripped_text[start_idx:]
                    repaired_json = repair_json(json_part)
                    obj = json.loads(repaired_json)
                    if isinstance(obj, (dict, list)):
                        return obj
                except Exception as e:
                    print(f"[QwenVLBrain] Failed to repair extracted JSON: {e}")

    except Exception:
        pass
    if not repair:
        return None
    # Final attempt: try to repair the entire text
    try:
        repaired_json = repair_json(stripped_text)
        obj = json.loads(repaired_json)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    return None
