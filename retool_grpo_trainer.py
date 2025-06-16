import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import GRPOTrainer, GRPOConfig
import logging
import datetime
from torch.utils.data import DataLoader, TensorDataset
from retool_interleaved_rollout import InterleavedCodeExecutor
import re

# ANSI color codes for terminal output
class Colors:
    QUESTION = '\033[94m'      # Blue
    RESPONSE = '\033[92m'      # Green
    EXPECTED = '\033[93m'      # Yellow
    ACTUAL = '\033[96m'        # Cyan
    CODE = '\033[95m'          # Magenta
    ERROR = '\033[91m'         # Red
    SUCCESS = '\033[32m'       # Bright Green
    RESET = '\033[0m'          # Reset
    BOLD = '\033[1m'           # Bold
    UNDERLINE = '\033[4m'      # Underline

class ColoredLogger:
    """Logger that saves colorized output to both terminal and HTML file"""
    
    def __init__(self, html_file_path: str = "training_log.html"):
        self.html_file_path = html_file_path
        self.setup_html_file()
        
        # ANSI to HTML color mapping
        self.color_map = {
            '\033[94m': '<span style="color: #5555FF;">',  # Blue
            '\033[92m': '<span style="color: #55FF55;">',  # Green  
            '\033[93m': '<span style="color: #FFFF55;">',  # Yellow
            '\033[96m': '<span style="color: #55FFFF;">',  # Cyan
            '\033[95m': '<span style="color: #FF55FF;">',  # Magenta
            '\033[91m': '<span style="color: #FF5555;">',  # Red
            '\033[32m': '<span style="color: #00FF00;">',  # Bright Green
            '\033[1m': '<strong>',                         # Bold
            '\033[4m': '<u>',                              # Underline
            '\033[0m': '</span></strong></u>',             # Reset (close all tags)
        }
    
    def setup_html_file(self):
        """Initialize the HTML file with CSS and structure"""
        html_header = """<!DOCTYPE html>
<html>
<head>
    <title>ReTool GRPO Training Log</title>
    <meta charset="UTF-8">
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background-color: #1e1e1e; 
            color: #ffffff; 
            margin: 20px;
            line-height: 1.4;
        }
        .log-container { 
            white-space: pre-wrap; 
            padding: 20px;
            border: 1px solid #444;
            border-radius: 5px;
            background-color: #2d2d2d;
        }
        .timestamp {
            color: #888;
            font-size: 0.9em;
        }
        .batch-separator {
            border-top: 2px solid #555;
            margin: 20px 0;
            padding-top: 10px;
        }
        h1 { color: #ffff55; text-align: center; }
        h2 { color: #55ffff; }
    </style>
    <script>
        // Auto-refresh every 5 seconds to see updates
        setInterval(function() {
            location.reload();
        }, 5000);
    </script>
</head>
<body>
    <h1>🚀 ReTool GRPO Training Progress</h1>
    <div class="log-container" id="log-content">
"""
        
        try:
            with open(self.html_file_path, 'w', encoding='utf-8') as f:
                f.write(html_header)
            print(f"📄 Training log will be saved to: {self.html_file_path}")
        except Exception as e:
            print(f"Warning: Could not create HTML log file: {e}")
    
    def log(self, text: str, color: str = "", prefix: str = ""):
        """Log text to both terminal and HTML file"""
        # Print to terminal (existing behavior)
        terminal_text = f"{color}{Colors.BOLD}{prefix}{Colors.RESET}{color}{text}{Colors.RESET}"
        print(terminal_text)
        
        # Convert to HTML and append to file
        self.append_to_html(text, color, prefix)
    
    def append_to_html(self, text: str, color: str = "", prefix: str = ""):
        """Append formatted text to HTML file"""
        try:
            # Add timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Create HTML formatted text
            html_text = ""
            if prefix:
                html_text += f'<strong style="color: {self._ansi_to_css_color(color)};">{prefix}</strong>'
            
            html_text += f'<span style="color: {self._ansi_to_css_color(color)};">{text}</span>'
            
            # Format the line
            html_line = f'<span class="timestamp">[{timestamp}]</span> {html_text}<br>\n'
            
            # Append to file
            with open(self.html_file_path, 'a', encoding='utf-8') as f:
                f.write(html_line)
                
        except Exception as e:
            # Silently continue if HTML logging fails
            pass
    
    def _ansi_to_css_color(self, ansi_code: str) -> str:
        """Convert ANSI color code to CSS color"""
        color_mapping = {
            '\033[94m': '#5555FF',  # Blue
            '\033[92m': '#55FF55',  # Green  
            '\033[93m': '#FFFF55',  # Yellow
            '\033[96m': '#55FFFF',  # Cyan
            '\033[95m': '#FF55FF',  # Magenta
            '\033[91m': '#FF5555',  # Red
            '\033[32m': '#00FF00',  # Bright Green
        }
        return color_mapping.get(ansi_code, '#ffffff')
    
    def log_batch_separator(self, batch_num: int):
        """Add a visual separator between batches"""
        separator_text = f"\n{'='*80}\nBATCH {batch_num} RESULTS\n{'='*80}\n"
        
        # Terminal output
        print(f"{Colors.BOLD}{separator_text}{Colors.RESET}")
        
        # HTML output
        try:
            html_separator = f'<div class="batch-separator"><h2>BATCH {batch_num} RESULTS</h2></div>\n'
            with open(self.html_file_path, 'a', encoding='utf-8') as f:
                f.write(html_separator)
        except:
            pass
    
    def close_html_file(self):
        """Close the HTML file properly"""
        try:
            html_footer = """
    </div>
    <p style="text-align: center; color: #888; margin-top: 20px;">
        Training completed at """ + str(datetime.datetime.now()) + """
    </p>
</body>
</html>"""
            with open(self.html_file_path, 'a', encoding='utf-8') as f:
                f.write(html_footer)
        except:
            pass

# Global logger instance
_colored_logger = None

def get_colored_logger():
    """Get or create the global colored logger"""
    global _colored_logger
    if _colored_logger is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"retool_training_log_{timestamp}.html"
        _colored_logger = ColoredLogger(log_filename)
    return _colored_logger

def print_colored(text: str, color: str, prefix: str = ""):
    """Print colored text with optional prefix to both terminal and HTML log"""
    logger = get_colored_logger()
    logger.log(text, color, prefix)

def extract_answer_from_response(response: str) -> str:
    """Extract the final answer from a response"""
    # First try to find XML <answer> tags (for compatibility with training_GRPO.py format)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        # Look for \\boxed{} within the answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer_content)
        if boxed_match:
            return boxed_match.group(1).strip()
        else:
            # Clean up the answer content
            # Remove extra whitespace and newlines
            clean_answer = ' '.join(answer_content.split())
            return clean_answer
    
    # Fallback: look for \\boxed{} anywhere in the response
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # NEW: Look for common answer patterns in the response
    # Pattern 1: "The answer is X" or "The total is X"
    answer_patterns = [
        r'(?:the answer is|the total (?:is|number of .* is)|the result is|equals?)\s*:?\s*\$?([0-9]+(?:\.[0-9]+)?)',
        r'(?:is|equals?)\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:dollars?|clips?|pages?|gallons?)?\.?\s*$',
        r'needs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:more\s*)?(?:dollars?|clips?|pages?|gallons?)',  # "needs 5 more dollars"
        r'\$([0-9]+(?:\.[0-9]+)?)',  # Currency format
        r'([0-9]+(?:\.[0-9]+)?)\s*(?:dollars?|clips?|pages?|gallons?)',  # Number with units
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            number = match.group(1)
            # Convert to integer if it's a whole number (including .00 format)
            if '.' in number:
                float_num = float(number)
                if float_num == int(float_num):  # Check if it's a whole number
                    return str(int(float_num))
            return number
    
    # Look for any number at the end of the response
    number_match = re.search(r'(\d+(?:\.\d+)?)\s*\.?\s*$', response)
    if number_match:
        number = number_match.group(1)
        # Convert to integer if it's a whole number
        if '.' in number and number.endswith('.0'):
            return str(int(float(number)))
        return number
    
    return ""

class ReToolGRPOTrainer(GRPOTrainer):
    """
    Simplified GRPO Trainer that applies ReTool reward functions.
    Uses TRL's standard generation but applies our custom reward logic.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        code_executor,  # Your existing PythonExecutor
        args: GRPOConfig,
        processing_class: PreTrainedTokenizer,
        train_dataset,
        reward_funcs: List[callable],
        **kwargs
    ):
        # Store code executor for potential use
        self.code_executor = code_executor
        
        # Initialize parent GRPO trainer with our reward functions
        super().__init__(
            model=model,
            args=args,
            processing_class=processing_class,
            train_dataset=train_dataset,
            reward_funcs=reward_funcs,
            **kwargs
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Store reward function names for logging
        self.reward_func_names = [getattr(func, '__name__', f'reward_func_{i}') 
                                  for i, func in enumerate(self.reward_funcs)]
        
        self.logger.info(f"Initialized ReTool GRPO Trainer with {len(reward_funcs)} reward functions")
    
    def on_train_end(self):
        """Called when training ends - close HTML log file"""
        try:
            logger = get_colored_logger()
            logger.close_html_file()
        except:
            pass