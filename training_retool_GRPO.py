# Exact modifications to your training_GRPO.py file for ReTool integration

# 1. IMPORTS - Add these new imports at the top
from retool_grpo_trainer import ReToolGRPOTrainer
from retool_rollout_grpo_trainer import ReToolExecutorAdapter, create_retool_compatible_rewards

# 2. SYSTEM PROMPT - Replace your existing SYSTEM_PROMPT
SYSTEM_PROMPT = """Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in <interpreter>output</interpreter>) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.

Code Format:
Each code snippet is wrapped with
<code>
```python
code snippet
```
</code>

Answer Format: 
The last part of your response should be:
<answer>\\boxed{The final answer goes here.}</answer>
"""

# 3. DATA PREPARATION - Modify your get_gsm8k_questions function
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] 
    data = data.map(lambda x: { 
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            # Remove the example conversation - let the model learn from RL
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) 
    return data

# 4. EXECUTOR SETUP - Modify your executor initialization
def setup_executor():
    """Setup code executor with ReTool adapter"""
    # Your existing executor
    original_executor = PythonExecutor(
        timeout_length=10,  # Increase timeout for complex calculations
        get_answer_from_stdout=True
    )
    
    # Wrap with ReTool adapter
    retool_executor = ReToolExecutorAdapter(original_executor)
    return retool_executor

# 5. REWARD FUNCTIONS - Replace your existing reward functions
def create_reward_functions():
    """Create ReTool-compatible reward functions"""
    
    def accuracy_reward(completions, prompts, answer, **kwargs) -> list[float]:
        """Primary outcome reward: +1 correct, -1 incorrect (ReTool style)"""
        responses = []
        if isinstance(completions[0], list):
            responses = [completion[0]['content'] for completion in completions]
        else:
            responses = [str(comp) for comp in completions]
        
        rewards = []
        for i, response in enumerate(responses):
            # Extract answer from ReTool response format
            predicted = extract_answer_from_retool_response(response)
            ground_truth = answer[i] if i < len(answer) else ""
            
            print('-'*20, f"Question:\n{prompts[i][-1]['content']}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Predicted: {predicted}")
            print(f"Full Response: {response[:300]}...")
            
            # ReTool uses binary reward: +1 or -1
            if predicted.strip().lower() == str(ground_truth).strip().lower():
                reward = 1.0
            else:
                reward = -1.0
            
            rewards.append(reward)
        
        return rewards
    
    def code_usage_reward(completions, **kwargs) -> list[float]:
        """Encourage strategic code usage"""
        responses = []
        if isinstance(completions[0], list):
            responses = [completion[0]['content'] for completion in completions]
        else:
            responses = [str(comp) for comp in completions]
        
        rewards = []
        for response in responses:
            reward = 0.0
            
            # Count successful code executions
            code_blocks = re.findall(r'<code>.*?</code>', response, re.DOTALL)
            interpreter_blocks = re.findall(r'<interpreter>(.*?)</interpreter>', response, re.DOTALL)
            
            # Small reward for using code
            if code_blocks:
                reward += 0.1 * min(len(code_blocks), 2)  # Up to 0.2
            
            # Bonus for successful execution (no errors)
            successful_executions = sum(1 for block in interpreter_blocks 
                                      if not block.strip().startswith('Error:'))
            reward += 0.1 * successful_executions
            
            rewards.append(reward)
        
        return rewards
    
    return [accuracy_reward, code_usage_reward]

# 6. MAIN TRAINING MODIFICATION - Replace your trainer initialization
def main():
    # ... (your existing setup code for model, tokenizer, etc.)
    
    # Setup ReTool executor
    retool_executor = setup_executor()
    
    # Create ReTool reward functions
    reward_functions = create_reward_functions()
    
    # Replace GRPOTrainer with ReToolGRPOTrainer
    trainer = ReToolGRPOTrainer(  # <-- This is the key change
        model=model,
        code_executor=retool_executor,  # <-- Add this parameter
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Training proceeds as normal
    trainer.train()

# 7. HELPER FUNCTION - Add this to extract answers from ReTool responses
def extract_answer_from_retool_response(response: str) -> str:
    """Extract final answer from ReTool-style response"""
    import re
    
    # Try <answer> tags first
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        # Look for \\boxed{} within answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer_content)
        if boxed_match:
            return boxed_match.group(1).strip()
        else:
            return answer_content
    
    # Fallback: look for \\boxed{} anywhere
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    return ""

# 8. TRAINING CONFIGURATION - Modify your training args
training_args = GRPOConfig(
    output_dir="outputs/ReTool-GRPO",
    run_name="ReTool_Qwen-0.5B-GRPO-gsm8k",
    learning_rate=5e-6,  # Slightly lower for stability
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=False,
    per_device_train_batch_size=1,  # Start with 1 for testing
    gradient_accumulation_steps=8,  # Increase to maintain effective batch size
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=1024,  # Increase for code + reasoning
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    cliprange=0.2,  # Important for PPO-style stability
)

# 9. COMPLETE MODIFIED TRAINING SCRIPT
"""
Here's what your modified training_GRPO.py should look like:
"""

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig
from retool_grpo_trainer import ReToolGRPOTrainer
from retool_rollout_grpo_trainer import ReToolExecutorAdapter
from executor import PythonExecutor
import wandb

wandb.login(key="3e508466126c563982f3cf9dedf9e84f13277fc5")
# Note: wandb.init() will be called automatically by TRL GRPOTrainer using run_name from GRPOConfig

# ReTool system prompt
SYSTEM_PROMPT = """Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in <interpreter>output</interpreter>) can be returned to aid your reasoning and help you arrive at the final answer.

Code Format:
Each code snippet is wrapped with
<code>
```python
code snippet
```
</code>

Answer Format: 
<answer>\\boxed{The final answer goes here.}</answer>
"""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def extract_answer_from_retool_response(response: str) -> str:
    # Try <answer> tags first
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer_content)
        if boxed_match:
            return boxed_match.group(1).strip()
        return answer_content
    
    # Fallback to \\boxed anywhere
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    return ""

def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

def accuracy_reward(completions, prompts, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] if isinstance(completion, list) else str(completion) 
                for completion in completions]
    
    rewards = []
    for i, response in enumerate(responses):
        predicted = extract_answer_from_retool_response(response)
        ground_truth = answer[i] if i < len(answer) else ""
        
        print(f"Predicted: '{predicted}' | Ground Truth: '{ground_truth}'")
        
        if predicted.strip().lower() == str(ground_truth).strip().lower():
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    
    return rewards

# Main execution
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=device,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Setup ReTool executor
original_executor = PythonExecutor(timeout_length=10, get_answer_from_stdout=True)
retool_executor = ReToolExecutorAdapter(original_executor)

# Training configuration
training_args = GRPOConfig(
    output_dir="outputs/ReTool-GRPO",
    run_name="ReTool_Qwen-0.5B-GRPO-gsm8k",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=1024,
    num_train_epochs=1,
    save_steps=50,
    cliprange=0.2,
    report_to="wandb",
)

# Load dataset
dataset = get_gsm8k_questions()

# Initialize trainer
trainer = ReToolGRPOTrainer(
    model=model,
    code_executor=retool_executor,
    processing_class=tokenizer,
    reward_funcs=[accuracy_reward],
    args=training_args,
    train_dataset=dataset,
)

# Train
trainer.train()