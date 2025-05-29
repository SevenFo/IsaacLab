from typing import Optional, Union, Dict, List
import json
import re


def extract_json_from_text(text: str) -> Optional[Union[Dict, List]]:
    """
    Helper to robustly extract and parse JSON from a text string.
    Tries various methods in order of preference/reliability.
    """
    if not text or not isinstance(text, str):
        return None

    stripped_text = text.strip()

    # Method 1: ```json ... ``` markdown block
    match = re.search(r"```json\s*([\s\S]*?)\s*```", stripped_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(
                f"[QwenVLBrain] Malformed JSON in ```json ... ``` block: {e}. Content: {json_str}"
            )
            # Fall through to other methods if this specific extraction fails
    stripped_text = stripped_text.replace(
        '{"{}"}', "{}"
    )  # Fix common formatting issues
    stripped_text = stripped_text.replace('{"{}}', "{}")
    match = re.search(r"```json\s*([\s\S]*?)\s*```", stripped_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(
                f"[QwenVLBrain] Malformed JSON in ```json ... ``` block: {e}. Content: {json_str}"
            )
            # Fall through to other methods if this specific extraction fails
    # Method 2: ``` ... ``` generic markdown block
    match = re.search(r"```\s*([\s\S]*?)\s*```", stripped_text, re.DOTALL)
    if match:
        potential_json_str = match.group(1).strip()
        # Check if it plausibly looks like JSON before attempting to parse
        if (
            potential_json_str.startswith("{")
            and potential_json_str.endswith("}")
        ) or (
            potential_json_str.startswith("[")
            and potential_json_str.endswith("]")
        ):
            try:
                return json.loads(potential_json_str)
            except json.JSONDecodeError as e:
                print(
                    f"[QwenVLBrain] Malformed JSON in ``` ... ``` block: {e}. Content: {potential_json_str[:100]}..."
                )
                # Fall through

    # Method 3: JSON as the entire content (possibly with surrounding whitespace)
    if (stripped_text.startswith("{") and stripped_text.endswith("}")) or (
        stripped_text.startswith("[") and stripped_text.endswith("]")
    ):
        try:
            return json.loads(stripped_text)
        except json.JSONDecodeError:
            # If it looked like full JSON but failed, it's genuinely malformed here.
            # The raw_decode below might still find a smaller valid part if it's prefix.
            pass

    # Method 4: Find the first valid JSON object or array using raw_decode
    # This is good for JSON embedded in text, e.g., "Some text {"key": "value"} more text"
    decoder = json.JSONDecoder()
    try:
        # Find the first occurrence of '{' or '['
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
            # Attempt to decode from this starting point
            obj, _ = decoder.raw_decode(stripped_text[start_idx:])
            # raw_decode can parse simple types like numbers if they are valid JSON.
            # We are typically interested in dicts or lists.
            if isinstance(obj, (dict, list)):
                return obj
    except json.JSONDecodeError:
        # No valid JSON found by raw_decode from the first potential start.
        pass

    return None
