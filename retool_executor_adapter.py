"""
Adapter to make your existing PythonExecutor work seamlessly with ReTool training.
This file contains the executor adapter plus all reward functions and helper utilities.
"""

import re
import logging
from typing import Tuple, Any, List

class ReToolExecutorAdapter:
    """
    Adapter to make your existing PythonExecutor compatible with ReTool training.
    Handles the specific formatting and error handling needed for RL training.
    """
    
    def __init__(self, original_executor, max_output_length=1000):
        self.executor = original_executor
        self.max_output_length = max_output_length
        self.logger = logging.getLogger(__name__)
    
    def apply(self, code: str) -> Tuple[str, str]:
        """
        Execute code and return (result, status) in ReTool format.
        
        Args:
            code: Python code string to execute
            
        Returns:
            Tuple of (output_string, status_string)
            - output_string: The actual result or error message
            - status_string: "Done" for success, error description for failure
        """
        try:
            # Clean the code input
            cleaned_code = self._clean_code_input(code)
            
            if not cleaned_code.strip():
                return "", "Error: Empty code block"
            
            # Execute using your existing executor
            result, report = self.executor.apply(cleaned_code)
            
            # Process the result
            if report == "Done":
                # Successful execution
                output = self._format_successful_output(result)
                return output, "Done"
            else:
                # Execution error
                error_msg = self._format_error_output(report)
                return error_msg, report
                
        except Exception as e:
            self.logger.error(f"Unexpected error in code execution: {e}")
            return f"Execution Error: {str(e)}", "Unexpected Error"
    
    def _clean_code_input(self, code: str) -> str:
        """
        Clean and prepare code for execution.
        """
        # Remove markdown code fences if present
        code = re.sub(r'^```python\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        # Basic safety checks (extend as needed)
        dangerous_patterns = [
            r'import\s+os\s*;.*os\.system',
            r'__import__\s*\(\s*[\'"]os[\'"]',
            r'exec\s*\(',
            r'eval\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise ValueError(f"Potentially dangerous code detected: {pattern}")
        
        return code
    
    def _format_successful_output(self, result: Any) -> str:
        """
        Format successful execution output for ReTool.
        """
        if result is None:
            return ""
        
        output_str = str(result)
        
        # Truncate if too long
        if len(output_str) > self.max_output_length:
            output_str = output_str[:self.max_output_length] + "... [truncated]"
        
        return output_str
    
    def _format_error_output(self, error_report: str) -> str:
        """
        Format error messages for ReTool.
        """
        # Clean up common error messages to be more informative
        if "NameError" in error_report:
            return f"NameError: {error_report.split('NameError:')[-1].strip()}"
        elif "SyntaxError" in error_report:
            return f"SyntaxError: {error_report.split('SyntaxError:')[-1].strip()}"
        elif "TypeError" in error_report:
            return f"TypeError: {error_report.split('TypeError:')[-1].strip()}"
        elif "ValueError" in error_report:
            return f"ValueError: {error_report.split('ValueError:')[-1].strip()}"
        else:
            return f"Error: {error_report}"

# Helper functions for answer extraction
def extract_answer_from_retool_response(response: str) -> str:
    """
    Extract the final answer from a ReTool-style response.
    Handles both <answer> tags and \\boxed{} format.
    """
    # First try to find <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        # Look for \\boxed{} within the answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer_content)
        if boxed_match:
            return boxed_match.group(1).strip()
        else:
            return answer_content
    
    # Fallback: look for \\boxed{} anywhere in the response
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    return ""

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML <answer> tags"""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    answer = answer.strip()
    
    # Also handle \\boxed{} format within XML answer tags
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    return answer

def extract_hash_answer(text: str) -> str:
    """Extract answer from GSM8K format (#### answer)"""
    if "####" not in text:
        return ""
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    return str(answer).strip().lower().replace(",", "").replace("$", "").replace(" ", "")

def has_proper_xml_structure(text: str) -> float:
    """Check if text has proper XML structure with reasoning before answer"""
    reward = 0.0
    
    # Check for opening reasoning tag
    if "<reasoning>" in text:
        reward += 0.1
        
    # Check for closing reasoning tag
    if "</reasoning>" in text:
        reward += 0.1
        
    # Check for opening answer tag
    if "<answer>" in text:
        reward += 0.1
        
    # Check for closing answer tag  
    if "</answer>" in text:
        reward += 0.1
        
    # Check proper order: reasoning tags come before answer tags
    reasoning_start = text.find("<reasoning>")
    reasoning_end = text.find("</reasoning>")
    answer_start = text.find("<answer>")
    answer_end = text.find("</answer>")
    
    if (reasoning_start != -1 and reasoning_end != -1 and 
        answer_start != -1 and answer_end != -1):
        
        # Check proper order
        if (reasoning_start < reasoning_end < answer_start < answer_end):
            reward += 0.2
            
            # Check that there's content between tags
            reasoning_content = text[reasoning_start + 11:reasoning_end].strip()
            answer_content = text[answer_start + 8:answer_end].strip()
            
            if len(reasoning_content) > 10:  # Meaningful reasoning content
                reward += 0.15
            if len(answer_content) > 0:  # Has answer content
                reward += 0.15
                
    return reward

# ReTool-compatible reward functions
def accuracy_reward(completions, prompts=None, answer=None, **kwargs) -> List[float]:
    """
    Primary outcome reward: +1 for correct answer, -1 for incorrect.
    This follows the ReTool paper's binary reward scheme.
    """
    if not answer:
        return [0.0] * len(completions)
    
    # Handle different completion formats
    responses = []
    if isinstance(completions[0], list):
        responses = [completion[0]['content'] for completion in completions]
    else:
        responses = [str(completion) for completion in completions]
    
    rewards = []
    for i, response in enumerate(responses):
        predicted_answer = extract_answer_from_retool_response(response)
        ground_truth = answer[i] if i < len(answer) else ""
        
        # Normalize for comparison
        predicted_norm = normalize_answer(predicted_answer)
        truth_norm = normalize_answer(ground_truth)
        
        # ReTool uses binary rewards: +1 correct, -1 incorrect
        if predicted_norm == truth_norm and predicted_norm != "":
            reward = 1.0
        else:
            reward = -1.0
        
        rewards.append(reward)
        
        # Debug logging
        if prompts and i < len(prompts):
            question = prompts[i][-1]['content'] if isinstance(prompts[i], list) else str(prompts[i])
            print(f"\n--- Reward Calculation {i+1} ---")
            print(f"Question: {question[:100]}...")
            print(f"Predicted: '{predicted_answer}' -> '{predicted_norm}'")
            print(f"Ground Truth: '{ground_truth}' -> '{truth_norm}'")
            print(f"Reward: {reward}")
            print(f"Response Sample: {response[:200]}...")
    
    return rewards

# Enhanced reward functions for better incentivization
def mandatory_code_usage_reward(completions, **kwargs) -> List[float]:
    """
    STRONG reward for attempting code usage, PENALTY for not using code.
    This is the primary behavioral incentive.
    """
    # Handle different completion formats
    responses = []
    if isinstance(completions[0], list):
        responses = [completion[0]['content'] for completion in completions]
    else:
        responses = [str(completion) for completion in completions]
    
    rewards = []
    for response in responses:
        reward = 0.0
        
        # Check for code blocks
        code_blocks = re.findall(r'<code>.*?</code>', response, re.DOTALL)
        interpreter_blocks = re.findall(r'<interpreter>(.*?)</interpreter>', response, re.DOTALL)
        
        if code_blocks:
            # STRONG positive reward for attempting code (even if it fails)
            reward += 1.0  # Base reward for code attempt
            
            # Additional reward for multiple code blocks (strategic usage)
            if len(code_blocks) > 1:
                reward += 0.3
            
            # Check for successful executions
            successful_executions = 0
            failed_executions = 0
            
            for block in interpreter_blocks:
                block_content = block.strip()
                if (block_content.startswith('Error:') or 
                    block_content.startswith('Traceback') or
                    'SyntaxError' in block_content or
                    'NameError' in block_content):
                    failed_executions += 1
                else:
                    successful_executions += 1
            
            # Reward successful executions more
            reward += successful_executions * 0.5
            
            # Small penalty for failed executions (but still net positive for trying)
            reward -= failed_executions * 0.1
            
        else:
            # STRONG PENALTY for not attempting code
            reward -= 2.0  # This makes not using code very costly
        
        rewards.append(reward)
    
    return rewards

def mandatory_xml_structure_reward(completions, **kwargs) -> List[float]:
    """
    STRONG reward for proper XML structure, PENALTY for not using it.
    Ensures reasoning comes before answer.
    """
    responses = []
    if isinstance(completions[0], list):
        responses = [completion[0]['content'] for completion in completions]
    else:
        responses = [str(completion) for completion in completions]
    
    rewards = []
    for response in responses:
        reward = 0.0
        
        # Check for basic XML structure
        has_reasoning_start = "<reasoning>" in response
        has_reasoning_end = "</reasoning>" in response
        has_answer_start = "<answer>" in response
        has_answer_end = "</answer>" in response
        
        if has_reasoning_start and has_reasoning_end and has_answer_start and has_answer_end:
            # Base reward for having XML structure
            reward += 1.0
            
            # Check proper order: reasoning before answer
            reasoning_start = response.find("<reasoning>")
            reasoning_end = response.find("</reasoning>")
            answer_start = response.find("<answer>")
            answer_end = response.find("</answer>")
            
            if (reasoning_start < reasoning_end < answer_start < answer_end):
                # Additional reward for proper order
                reward += 0.5
                
                # Check that reasoning has substantial content
                reasoning_content = response[reasoning_start + 11:reasoning_end].strip()
                if len(reasoning_content) > 50:  # Meaningful reasoning
                    reward += 0.3
                
                # Check that answer has content
                answer_content = response[answer_start + 8:answer_end].strip()
                if len(answer_content) > 0:
                    reward += 0.2
                    
        else:
            # STRONG PENALTY for not using XML structure
            reward -= 1.5
        
        rewards.append(reward)
    
    return rewards

def reasoning_before_code_reward(completions, **kwargs) -> List[float]:
    """
    Reward for having reasoning text before code execution.
    This encourages thoughtful code usage.
    """
    responses = []
    if isinstance(completions[0], list):
        responses = [completion[0]['content'] for completion in completions]
    else:
        responses = [str(completion) for completion in completions]
    
    rewards = []
    for response in responses:
        reward = 0.0
        
        # Find reasoning section
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        if reasoning_match:
            reasoning_content = reasoning_match.group(1)
            
            # Find code blocks within reasoning
            code_blocks_in_reasoning = re.findall(r'<code>.*?</code>', reasoning_content, re.DOTALL)
            
            if code_blocks_in_reasoning:
                # Check if there's text before the first code block
                first_code_pos = reasoning_content.find('<code>')
                text_before_code = reasoning_content[:first_code_pos].strip()
                
                if len(text_before_code) > 20:  # Substantial reasoning before code
                    reward += 0.8
                elif len(text_before_code) > 5:  # Some reasoning before code
                    reward += 0.4
                else:
                    # Penalty for jumping straight to code without reasoning
                    reward -= 0.3
                
                # Additional reward for reasoning between code blocks
                code_positions = []
                for match in re.finditer(r'<code>.*?</code>', reasoning_content, re.DOTALL):
                    code_positions.append((match.start(), match.end()))
                
                if len(code_positions) > 1:
                    # Check for reasoning between code blocks
                    reasoning_between_code = 0
                    for i in range(len(code_positions) - 1):
                        end_of_current = code_positions[i][1]
                        start_of_next = code_positions[i + 1][0]
                        between_text = reasoning_content[end_of_current:start_of_next].strip()
                        if len(between_text) > 10:
                            reasoning_between_code += 1
                    
                    reward += reasoning_between_code * 0.2
        
        rewards.append(reward)
    
    return rewards

def comprehensive_behavioral_reward(completions, prompts=None, answer=None, **kwargs) -> List[float]:
    """
    Enhanced comprehensive reward that strongly incentivizes desired behaviors.
    """
    # Get individual reward components
    acc_rewards = accuracy_reward(completions, prompts, answer, **kwargs)
    code_rewards = mandatory_code_usage_reward(completions, **kwargs)
    xml_rewards = mandatory_xml_structure_reward(completions, **kwargs)
    reasoning_rewards = reasoning_before_code_reward(completions, **kwargs)
    
    # Combine with strong weights for behavioral incentives
    combined_rewards = []
    for i in range(len(completions)):
        total_reward = (
            acc_rewards[i] * 2.0 +          # Accuracy is still most important
            code_rewards[i] * 1.5 +         # Strong incentive for code usage
            xml_rewards[i] * 1.2 +          # Strong incentive for XML format
            reasoning_rewards[i] * 0.8      # Incentive for thoughtful reasoning
        )
        combined_rewards.append(total_reward)
    
    return combined_rewards

def strategic_code_placement_reward(completions, **kwargs) -> List[float]:
    """
    Reward for strategic placement of code within reasoning.
    Encourages code to be used for calculation, verification, or exploration.
    """
    responses = []
    if isinstance(completions[0], list):
        responses = [completion[0]['content'] for completion in completions]
    else:
        responses = [str(completion) for completion in completions]
    
    rewards = []
    for response in responses:
        reward = 0.0
        
        # Look for strategic code usage patterns
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        if reasoning_match:
            reasoning_content = reasoning_match.group(1)
            
            # Check for calculation-related keywords before code
            calc_keywords = ['calculate', 'compute', 'find', 'solve', 'determine', 'check', 'verify']
            code_blocks = list(re.finditer(r'<code>.*?</code>', reasoning_content, re.DOTALL))
            
            for code_block in code_blocks:
                # Look at text before this code block
                text_before = reasoning_content[:code_block.start()].lower()
                
                # Reward if calculation keywords appear before code
                keyword_found = any(keyword in text_before[-100:] for keyword in calc_keywords)
                if keyword_found:
                    reward += 0.3
                
                # Look at the actual code content
                code_content = code_block.group(0)
                
                # Reward for mathematical operations
                if any(op in code_content for op in ['+', '-', '*', '/', '**', 'math.', 'sum(', 'len(']):
                    reward += 0.2
                
                # Reward for variable assignments (shows structured thinking)
                if '=' in code_content and not code_content.count('=') == code_content.count('=='):
                    reward += 0.1
                
                # Reward for print statements (shows intent to see results)
                if 'print(' in code_content:
                    reward += 0.1
        
        rewards.append(reward)
    
    return rewards

# Legacy reward functions (keeping for compatibility)
def code_usage_reward(completions, **kwargs) -> List[float]:
    """
    DEPRECATED: Use mandatory_code_usage_reward instead.
    Kept for backward compatibility.
    """
    return mandatory_code_usage_reward(completions, **kwargs)

def format_reward(completions, **kwargs) -> List[float]:
    """
    DEPRECATED: Use mandatory_xml_structure_reward instead.
    Kept for backward compatibility.
    """
    return mandatory_xml_structure_reward(completions, **kwargs)

def comprehensive_reward(completions, prompts=None, answer=None, **kwargs) -> List[float]:
    """
    DEPRECATED: Use comprehensive_behavioral_reward instead.
    Kept for backward compatibility.
    """
    return comprehensive_behavioral_reward(completions, prompts, answer, **kwargs)

def xml_accuracy_reward(completions, prompts=None, answer=None, **kwargs) -> List[float]:
    """
    XML-based accuracy reward that uses extract_xml_answer for extraction.
    Compatible with training_GRPO.py format.
    """
    if not answer:
        return [0.0] * len(completions)
    
    # Handle different completion formats
    responses = []
    if isinstance(completions[0], list):
        responses = [completion[0]['content'] for completion in completions]
    else:
        responses = [str(completion) for completion in completions]
    
    rewards = []
    for i, response in enumerate(responses):
        predicted_answer = extract_xml_answer(response)
        ground_truth = answer[i] if i < len(answer) else ""
        
        # Normalize for comparison
        predicted_norm = normalize_answer(predicted_answer)
        truth_norm = normalize_answer(ground_truth)
        
        # Binary reward: 0.05 for correct (matching training_GRPO.py), 0.0 for incorrect
        if predicted_norm == truth_norm and predicted_norm != "":
            reward = 0.05
        else:
            reward = 0.0
        
        rewards.append(reward)
        
        # Debug logging
        if prompts and i < len(prompts):
            question = prompts[i][-1]['content'] if isinstance(prompts[i], list) else str(prompts[i])
            print(f"\n--- XML Accuracy Calculation {i+1} ---")
            print(f"Question: {question[:100]}...")
            print(f"Predicted: '{predicted_answer}' -> '{predicted_norm}'")
            print(f"Ground Truth: '{ground_truth}' -> '{truth_norm}'")
            print(f"Reward: {reward}")
    
    return rewards

def int_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the extracted answer is a valid number"""
    responses = []
    if isinstance(completions[0], list):
        responses = [completion[0]['content'] for completion in completions]
    else:
        responses = [str(completion) for completion in completions]
    
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks for exact XML format with newlines"""
    responses = []
    if isinstance(completions[0], list):
        responses = [completion[0]['content'] for completion in completions]
    else:
        responses = [str(completion) for completion in completions]
    
    rewards = []
    for response in responses:
        # Check for exact format with newlines
        pattern = r"<reasoning>\s*\n.*?\n\s*</reasoning>\s*\n\s*<answer>\s*\n.*?\n\s*</answer>"
        match = re.search(pattern, response, re.DOTALL)
        rewards.append(0.5 if match else 0.0)
    
    return rewards

def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks for flexible XML format"""
    responses = []
    if isinstance(completions[0], list):
        responses = [completion[0]['content'] for completion in completions]
    else:
        responses = [str(completion) for completion in completions]
    
    rewards = []
    for response in responses:
        # More flexible pattern matching
        pattern = r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>"
        match = re.search(pattern, response, re.DOTALL)
        rewards.append(0.5 if match else 0.0)
    
    return rewards

def xml_reward_func(completions, **kwargs) -> List[float]:
    """Comprehensive XML reward function that incentivizes proper structure and content"""
    responses = []
    if isinstance(completions[0], list):
        responses = [completion[0]['content'] for completion in completions]
    else:
        responses = [str(completion) for completion in completions]
    
    return [has_proper_xml_structure(response) for response in responses]

# Convenience function to create reward function lists
def create_retool_reward_functions(use_separate_rewards=True, include_xml_rewards=True):
    """
    Create list of reward functions for ReTool training.
    
    Args:
        use_separate_rewards: If True, return separate reward functions.
                            If False, return single comprehensive reward.
        include_xml_rewards: If True, include XML format reward functions from training_GRPO.py
    """
    if use_separate_rewards:
        # Enhanced behavioral rewards (primary incentives)
        base_rewards = [
            accuracy_reward,                    # Primary: correctness (+1/-1)
            mandatory_code_usage_reward,        # STRONG: code usage incentive (+1/-2)
            mandatory_xml_structure_reward,     # STRONG: XML format incentive (+1.8/-1.5)
            reasoning_before_code_reward,       # Thoughtful code usage (+0.8/-0.3)
            strategic_code_placement_reward,    # Strategic code usage patterns
        ]
        
        if include_xml_rewards:
            # Additional XML compatibility rewards (smaller weights)
            xml_rewards = [
                xml_accuracy_reward,        # XML-based accuracy (matches training_GRPO.py)
                xml_reward_func,           # Comprehensive XML structure checking
                soft_format_reward_func,   # Flexible XML format checking
                strict_format_reward_func, # Exact XML format checking  
                int_reward_func,           # Numeric answer validation
            ]
            return base_rewards + xml_rewards
        else:
            return base_rewards
    else:
        return [comprehensive_behavioral_reward]

def create_training_grpo_compatible_rewards():
    """
    Create reward functions that exactly match training_GRPO.py setup.
    Use this for direct compatibility with the original training approach.
    """
    return [
        xml_reward_func,           # Comprehensive XML structure
        soft_format_reward_func,   # Flexible XML format  
        strict_format_reward_func, # Exact XML format
        int_reward_func,           # Numeric validation
        xml_accuracy_reward,       # XML-based accuracy
    ]

def create_enhanced_behavioral_rewards():
    """
    Create enhanced reward functions that strongly incentivize:
    1. Always attempting to use code
    2. Using proper XML format with reasoning before answer
    3. Having reasoning before code execution
    4. Strategic code placement
    
    These rewards use penalties for not following desired behaviors.
    """
    return [
        accuracy_reward,                    # Primary: correctness (binary +1/-1)
        mandatory_code_usage_reward,        # STRONG: code usage (+1.3/-2.0)
        mandatory_xml_structure_reward,     # STRONG: XML format (+1.8/-1.5)  
        reasoning_before_code_reward,       # Thoughtful reasoning (+0.8/-0.3)
        strategic_code_placement_reward,    # Strategic patterns (+0.7)
    ]

# Testing function
def test_reward_functions():
    """Test the reward functions with sample data"""
    
    # Sample completions with different formats
    test_completions = [
        # ReTool format with code
        "Let me solve this step by step.\n<code>\n```python\nresult = 2 + 2\nprint(result)\n```\n</code>\n<interpreter>4</interpreter>\nThe answer is 4.\n<answer>\\boxed{4}</answer>",
        
        # XML format (training_GRPO.py style)
        "<reasoning>\nI need to calculate 2 + 2.\n\n<code>\n```python\nresult = 2 + 2\nprint(result)\n```\n</code>\n<interpreter>4</interpreter>\n</reasoning>\n\n<answer>4</answer>",
        
        # Wrong answer, no formatting
        "I think the answer is 5.",
        
        # Correct answer, good formatting, no code
        "<answer>\\boxed{4}</answer>",
    ]
    
    test_prompts = [
        [{'role': 'user', 'content': 'What is 2 + 2?'}]
    ] * 4
    
    test_answers = ["4", "4", "4", "4"]
    
    print("Testing reward functions...")
    print("=" * 50)
    
    # Test enhanced behavioral rewards
    print("\n--- Enhanced Behavioral Rewards ---")
    acc_rewards = accuracy_reward(test_completions, test_prompts, test_answers)
    print(f"Accuracy rewards: {acc_rewards}")
    
    code_rewards = mandatory_code_usage_reward(test_completions)
    print(f"Mandatory code usage rewards: {code_rewards}")
    
    xml_rewards = mandatory_xml_structure_reward(test_completions)
    print(f"Mandatory XML structure rewards: {xml_rewards}")
    
    reasoning_rewards = reasoning_before_code_reward(test_completions)
    print(f"Reasoning before code rewards: {reasoning_rewards}")
    
    strategic_rewards = strategic_code_placement_reward(test_completions)
    print(f"Strategic code placement rewards: {strategic_rewards}")
    
    behavioral_rewards = comprehensive_behavioral_reward(test_completions, test_prompts, test_answers)
    print(f"Comprehensive behavioral rewards: {behavioral_rewards}")
    
    # Test new XML rewards
    print("\n--- New XML Rewards ---")
    xml_acc_rewards = xml_accuracy_reward(test_completions, test_prompts, test_answers)
    print(f"XML accuracy rewards: {xml_acc_rewards}")
    
    xml_struct_rewards = xml_reward_func(test_completions)
    print(f"XML structure rewards: {xml_struct_rewards}")
    
    soft_format_rewards = soft_format_reward_func(test_completions)
    print(f"Soft format rewards: {soft_format_rewards}")
    
    strict_format_rewards = strict_format_reward_func(test_completions)
    print(f"Strict format rewards: {strict_format_rewards}")
    
    int_rewards = int_reward_func(test_completions)
    print(f"Integer rewards: {int_rewards}")
    
    # Test combined reward functions
    print("\n--- Combined Reward Functions ---")
    combined_rewards = create_retool_reward_functions(use_separate_rewards=True, include_xml_rewards=True)
    print(f"Number of reward functions: {len(combined_rewards)}")
    
    grpo_compatible_rewards = create_training_grpo_compatible_rewards()
    print(f"GRPO-compatible reward functions: {len(grpo_compatible_rewards)}")
    
    print("\n--- Testing Answer Extraction ---")
    for i, completion in enumerate(test_completions):
        xml_answer = extract_xml_answer(completion)
        retool_answer = extract_answer_from_retool_response(completion)
        print(f"Completion {i+1}:")
        print(f"  XML extraction: '{xml_answer}'")
        print(f"  ReTool extraction: '{retool_answer}'")
        print(f"  Preview: {completion[:100]}...")
        print()

if __name__ == "__main__":
    test_reward_functions()