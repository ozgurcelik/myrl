import re
from collections import Counter
from typing import Any, Dict, List, Optional


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    answer_regex = r"<answer>.*?<\/answer>\s*$"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    think_response_order_correct = float(is_thinking_after_response(response))
    think_answer_appears_once = float(is_think_answer_appear_once(response))

    # Check for exactly one <think> and one </think>
    reward += 0.05 * (think_answer_appears_once + think_response_order_correct)

    if answer_match:
        reward += 0.5

    return reward


def answer_reward_function(
    response: str, numbers: List[int] = None, target: int = None
) -> Dict[str, float]:
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target.
    
    Returns:
        Dict with 'reward', 'number_usage_reward', and 'correctness_reward' keys.
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return {"reward": 0.0, "number_usage_reward": 0.0, "correctness_reward": 0.0}

    answer_content = answer_match.group(1)
    if not answer_content:
        return {"reward": 0.0, "number_usage_reward": 0.0, "correctness_reward": 0.0}

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return {"reward": 0.0, "number_usage_reward": 0.0, "correctness_reward": 0.0}

    # if is_answer_correct(answer_content, target):
    #     return {"reward": 1.0, "number_usage_reward": 1.0, "correctness_reward": 1.0}

    reward = 0.0
    number_usage_reward = 0.0
    correctness_reward = 0.0

    if answer_match:
        number_usage_reward = float(number_usage_reward_function(answer_content, numbers, target))
        reward += number_usage_reward * 0.1

        if "=" in answer_content:
            options = answer_content.split("=")
            # Longer one is the math expression
            if len(options[0].strip()) >= len(options[1].strip()):
                correctness_reward = float(is_answer_correct(options[0].strip(), target))
            else:
                correctness_reward = float(is_answer_correct(options[1].strip(), target))
        else:
            correctness_reward = float(is_answer_correct(answer_content, target))

        reward += correctness_reward * 0.8
            
    return {
        "reward": reward,
        "number_usage_reward": number_usage_reward,
        "correctness_reward": correctness_reward,
    }


def is_answer_correct(answer_content: str, target: int) -> bool:
    result = eval(answer_content, {"__builtins__": None}, {})
    if abs(float(result) - float(target)) < 1e-5:
        return True
    return False


def reward_function(
    response: str,
    numbers: List[int] = None,
    target: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Reward function for Countdown Tasks.

    Total reward = 0.1 * format_reward + 0.1 * number_usage_reward + 0.8 * correctness_reward
    
    Returns:
        Dict with:
            - reward: total reward score
            - reward_info: detailed breakdown of all reward components
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_result = answer_reward_function(response, numbers, target)

    return {
        "reward": format_reward * 0.1 + answer_result["reward"],
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_result["reward"],
            "number_usage_reward": answer_result["number_usage_reward"],
            "correctness_reward": answer_result["correctness_reward"],
        },
    }


def is_thinking_after_response(response: str) -> float:
    """Check if the response, <answer>...</answer> appears before <think>...</think>."""
    think_index = response.find("<think>")
    answer_index = response.find("<answer>")
    think_end_index = response.find("</think>")
    answer_end_index = response.find("</answer>")
    if think_index == -1 or answer_index == -1:
        return False
    return (
        (answer_index > think_index) and 
        (answer_end_index > think_end_index) and 
        (answer_index > think_end_index)
    )

def is_think_answer_appear_once(response: str) -> float:
    """Check if <think>...</think> and <answer>...</answer> appear exactly once."""
    think_open_count = response.count("<think>")
    think_close_count = response.count("</think>")
    answer_open_count = response.count("<answer>")
    answer_close_count = response.count("</answer>")
    if think_open_count == 1 and think_close_count == 1 and answer_open_count == 1 and answer_close_count == 1:
        return True
    return False


def number_usage_reward_function(answer_content: str, numbers: list[int], target: int) -> float:
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target
    """
    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if Counter(used_numbers) == Counter(numbers):
        return 1.0
        
    if Counter(numbers + [target]) == Counter(used_numbers):
        return 0.5
    
    return 0.0


if __name__ == "__main__":
    numbers = [11, 6]
    target = 66
    response = " We need to make 66 using the numbers 11 and 6 exactly once. Let's try different combinations of addition, subtraction, multiplication, and division.</think>\n<answer>(11 * 6)</answer>"
    print(reward_function(response, numbers, target))
