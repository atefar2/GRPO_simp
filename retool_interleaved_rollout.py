import torch
import re
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

class InterleavedCodeExecutor:
    """Handles interleaved code execution during rollout generation"""
    
    def __init__(self, code_executor, tokenizer, max_code_executions=5):
        self.code_executor = code_executor
        self.tokenizer = tokenizer
        self.max_code_executions = max_code_executions
        
        # Special tokens for parsing
        self.code_start_token = "<code>"
        self.code_end_token = "</code>"
        self.interpreter_start = "<interpreter>"
        self.interpreter_end = "</interpreter>"
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_with_code_execution(
        self, 
        model, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.7,
        do_sample: bool = True,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate response with interleaved code execution.
        Returns dict with full sequence, execution info, and masks.
        """
        # Force CPU device to avoid MPS issues
        device = torch.device("cpu")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model = model.to(device)
        
        batch_size = input_ids.shape[0]
        
        # Track generation state for each batch item
        generation_states = []
        for i in range(batch_size):
            generation_states.append({
                'input_ids': input_ids[i:i+1].clone(),
                'attention_mask': attention_mask[i:i+1].clone(),
                'generated_text': '',
                'code_executions': [],
                'is_complete': False,
                'kv_cache': None,
                'interpreter_masks': []  # Track which tokens to mask from loss
            })
        
        total_tokens_generated = 0
        code_execution_count = 0
        
        while total_tokens_generated < max_new_tokens and code_execution_count < self.max_code_executions:
            active_states = [state for state in generation_states if not state['is_complete']]
            if not active_states:
                break
                
            # Generate until we hit </code> or max tokens
            batch_results = self._generate_until_code_end(
                model, active_states, temperature, top_p, do_sample, **generation_kwargs
            )
            
            # Process each result
            for i, result in enumerate(batch_results):
                state = active_states[i]
                new_text = result['generated_text']
                state['generated_text'] += new_text
                total_tokens_generated += result['tokens_generated']
                
                # Check if we found code to execute
                if self.code_end_token in new_text:
                    code_executed = self._execute_code_and_update_state(state)
                    if code_executed:
                        code_execution_count += 1
                else:
                    # No more code found, mark as complete
                    state['is_complete'] = True
        
        # Convert back to batch format
        return self._states_to_batch_output(generation_states)
    
    def _generate_until_code_end(
        self, 
        model, 
        states: List[Dict], 
        temperature: float,
        top_p: float, 
        do_sample: bool,
        **kwargs
    ) -> List[Dict]:
        """Generate text until we hit </code> or natural stopping point"""
        results = []
        
        for state in states:
            current_input_ids = state['input_ids']
            current_attention_mask = state['attention_mask']
            
            # Use KV cache if available (ReTool optimization)
            past_key_values = state.get('kv_cache', None)
            
            with torch.no_grad():
                # Generate token by token to detect </code>
                max_tokens_this_round = 200  # Prevent infinite loops
                tokens_generated = 0
                generated_text = ""
                
                # Track if this is the first forward pass for this state
                is_first_pass = past_key_values is None
                
                for _ in range(max_tokens_this_round):
                    # Handle KV cache properly
                    if is_first_pass or past_key_values is None:
                        # First pass: use full sequence
                        model_input_ids = current_input_ids
                        model_attention_mask = current_attention_mask
                        is_first_pass = False
                    else:
                        # Subsequent passes: only use the last token
                        model_input_ids = current_input_ids[:, -1:]
                        # Attention mask should still cover the full sequence
                        model_attention_mask = current_attention_mask
                    
                    outputs = model(
                        input_ids=model_input_ids,
                        attention_mask=model_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    # Sample next token
                    logits = outputs.logits[:, -1, :] / temperature
                    if do_sample:
                        probs = torch.softmax(logits, dim=-1)
                        if top_p < 1.0:
                            probs = self._apply_top_p(probs, top_p)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    # Update input_ids and attention_mask
                    current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                    current_attention_mask = torch.cat([
                        current_attention_mask, 
                        torch.ones_like(next_token)
                    ], dim=-1)
                    
                    # Decode new token
                    new_token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                    generated_text += new_token_text
                    tokens_generated += 1
                    
                    # Update KV cache for next iteration
                    past_key_values = outputs.past_key_values
                    
                    # Check if we hit </code> or natural stopping
                    if (self.code_end_token in generated_text or 
                        next_token.item() == self.tokenizer.eos_token_id):
                        break
                
                # Update state
                state['input_ids'] = current_input_ids
                state['attention_mask'] = current_attention_mask
                state['kv_cache'] = past_key_values
                
                results.append({
                    'generated_text': generated_text,
                    'tokens_generated': tokens_generated
                })
        
        return results
    
    def _execute_code_and_update_state(self, state: Dict) -> bool:
        """Extract code, execute it, and update the generation state"""
        generated_text = state['generated_text']
        
        # Extract the most recent code block
        code_pattern = rf'{re.escape(self.code_start_token)}(.*?){re.escape(self.code_end_token)}'
        matches = list(re.finditer(code_pattern, generated_text, re.DOTALL))
        
        if not matches:
            return False
        
        # Get the last code block
        last_match = matches[-1]
        code_content = last_match.group(1).strip()
        
        # Clean up code (remove ```python and ``` markers)
        code_content = re.sub(r'^```python\s*\n?', '', code_content)
        code_content = re.sub(r'\n?```\s*$', '', code_content)
        
        try:
            # Execute code using your existing executor
            execution_result, execution_report = self.code_executor.apply(code_content)
            
            if execution_report == "Done":
                interpreter_output = f"{self.interpreter_start}{execution_result}{self.interpreter_end}"
            else:
                interpreter_output = f"{self.interpreter_start}Error: {execution_report}{self.interpreter_end}"
            
            # Record this execution
            state['code_executions'].append({
                'code': code_content,
                'result': execution_result,
                'report': execution_report
            })
            
            # Add interpreter output to generated text
            state['generated_text'] += interpreter_output
            
            # Tokenize interpreter output and add to input_ids
            interpreter_tokens = self.tokenizer.encode(
                interpreter_output, 
                add_special_tokens=False,
                return_tensors='pt'
            )
            # Ensure tokens are on the same device as existing input_ids
            interpreter_tokens = interpreter_tokens.to(state['input_ids'].device)
            
            # Update input_ids and attention_mask
            state['input_ids'] = torch.cat([state['input_ids'], interpreter_tokens], dim=-1)
            state['attention_mask'] = torch.cat([
                state['attention_mask'],
                torch.ones(interpreter_tokens.shape, device=state['input_ids'].device)
            ], dim=-1)
            
            # Track which tokens should be masked from loss (interpreter output)
            start_mask_pos = state['input_ids'].shape[-1] - interpreter_tokens.shape[-1]
            end_mask_pos = state['input_ids'].shape[-1]
            state['interpreter_masks'].append((start_mask_pos, end_mask_pos))
            
            # Clear KV cache since we're adding external tokens
            state['kv_cache'] = None
            
            self.logger.info(f"Code executed successfully. Result length: {len(execution_result)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            error_output = f"{self.interpreter_start}Execution Error: {str(e)}{self.interpreter_end}"
            state['generated_text'] += error_output
            return False
    
    def _apply_top_p(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus sampling"""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create mask
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0
        return probs / probs.sum(dim=-1, keepdim=True)
    
    def _states_to_batch_output(self, states: List[Dict]) -> Dict[str, Any]:
        """Convert individual states back to batch format"""
        # Pad sequences to same length
        max_length = max(state['input_ids'].shape[-1] for state in states)
        
        batch_input_ids = []
        batch_attention_masks = []
        batch_interpreter_masks = []
        
        for state in states:
            input_ids = state['input_ids'][0]  # Remove batch dimension
            attention_mask = state['attention_mask'][0]
            
            # Pad if necessary
            if len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), self.tokenizer.pad_token_id, device=input_ids.device)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros((pad_length,), device=attention_mask.device)
                ])
            
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            
            # Create interpreter mask for this sequence
            interpreter_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for start_pos, end_pos in state['interpreter_masks']:
                if end_pos <= len(interpreter_mask):
                    interpreter_mask[start_pos:end_pos] = True
            batch_interpreter_masks.append(interpreter_mask)
        
        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_masks),
            'interpreter_masks': torch.stack(batch_interpreter_masks),  # For loss masking
            'code_executions': [state['code_executions'] for state in states],
            'generated_texts': [state['generated_text'] for state in states]
        }