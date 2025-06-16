#!/usr/bin/env python3
"""
Simplified ReTool GRPO Training Script
Based on training_GRPO.py but with ReTool reward functions for wandb logging.
"""

import os
import re
import torch
import datetime
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force CPU usage - disable MPS completely
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Import ReTool reward functions
import wandb
from retool_executor_adapter import create_enhanced_behavioral_rewards

# SYSTEM PROMPT (same as training_GRPO.py)
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_hash_answer(text: str) -> str:
    """Extract answer from GSM8K format (#### answer)"""
    if "####" not in text:
        return ""
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def get_gsm8k_questions(split="train") -> Dataset:
    """Prepare GSM8K dataset (same as training_GRPO.py)"""
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
                answer="7"
            )},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

def setup_wandb():
    """Setup wandb (same as before)"""
    try:
        os.environ["WANDB_PROJECT"] = "retool-grpo-training"
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_WATCH"] = "false"
        
        wandb_api_key = "3e508466126c563982f3cf9dedf9e84f13277fc5"
        if not wandb.api.api_key:
            wandb.login(key=wandb_api_key)
        
        print("✓ Wandb configured for TRL integration")
        return True
    except Exception as e:
        print(f"⚠ Wandb setup failed: {e}")
        return False

def setup_device():
    """Setup device"""
    device = "cpu"
    print("Using CPU for stability")
    torch.set_default_device("cpu")
    return device

def main():
    """Main training function - closely follows training_GRPO.py structure"""
    
    # Setup wandb
    setup_wandb()
    
    # Configuration (same as training_GRPO.py)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Setup device
    device = setup_device()
    
    # Load dataset
    dataset = get_gsm8k_questions()
    
    # Limit dataset size for testing
    dataset_size = min(len(dataset), 20)
    dataset = dataset.select(range(dataset_size))
    print(f"Using {dataset_size} training examples")
    
    # Model and tokenizer (same as training_GRPO.py)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Training args (same structure as training_GRPO.py)
    training_args = GRPOConfig(
        output_dir="outputs/ReTool-Qwen-0.5B-GRPO",
        run_name="ReTool_Qwen-0.5B-GRPO-gsm8k",
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=1,
        save_steps=50,  # Save more frequently
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
    )
    
    # Use ReTool reward functions instead of training_GRPO.py ones
    print("Setting up ReTool reward functions...")
    reward_functions = create_enhanced_behavioral_rewards()
    
    print("Reward functions that will be logged to wandb:")
    for i, func in enumerate(reward_functions):
        print(f"  {i+1}. {func.__name__}")
    
    # Create trainer (same as training_GRPO.py but with ReTool rewards)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,  # <-- This is the key difference!
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training with ReTool rewards...")
    print("Expected wandb metrics:")
    print("- rewards/accuracy_reward/mean")
    print("- rewards/mandatory_code_usage_reward/mean") 
    print("- rewards/mandatory_xml_structure_reward/mean")
    print("- And individual std metrics for each reward function")
    
    # Train (exactly like training_GRPO.py)
    trainer.train()
    
    print("✓ Training completed!")
    print("✓ All reward metrics should be logged to wandb")
    print("✓ Check wandb dashboard for detailed reward function metrics")

if __name__ == "__main__":
    main()