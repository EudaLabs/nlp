"""
GPT-2 model wrapper for text generation.
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPT2Generator:
    """
    GPT-2 model wrapper for text generation.
    
    Supports:
    - Creative writing
    - Story completion
    - Dialogue generation
    - Code generation
    - Controlled generation
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None
    ):
        """
        Initialize GPT-2 model.
        
        Args:
            model_name: HuggingFace model name (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
            device: Device to use (cuda/cpu). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading GPT-2 model: {model_name} on {self.device}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad token (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 10,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text using GPT-2 model.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            min_length: Minimum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
            do_sample: Whether to use sampling (vs greedy)
            no_repeat_ngram_size: Size of n-grams to prevent repetition
            early_stopping: Whether to stop at EOS token
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        # Tokenize
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        # Get attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts
    
    def complete_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        **kwargs
    ) -> str:
        """
        Complete text starting with the given prompt.
        
        Args:
            prompt: Starting text
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Completed text (including prompt)
        """
        prompt_length = len(self.tokenizer.encode(prompt))
        max_length = prompt_length + max_new_tokens
        
        outputs = self.generate(
            prompt,
            max_length=max_length,
            min_length=prompt_length + 1,
            num_return_sequences=1,
            temperature=temperature,
            **kwargs
        )
        
        return outputs[0]
    
    def generate_story(
        self,
        prompt: str,
        max_length: int = 300,
        temperature: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate a story from a prompt.
        
        Args:
            prompt: Story beginning or idea
            max_length: Maximum story length
            temperature: Creativity level (higher = more creative)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated story
        """
        outputs = self.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.2,
            **kwargs
        )
        
        return outputs[0]
    
    def generate_dialogue(
        self,
        context: str,
        num_turns: int = 3,
        max_turn_length: int = 50,
        temperature: float = 0.85,
        **kwargs
    ) -> List[str]:
        """
        Generate dialogue turns.
        
        Args:
            context: Dialogue context/prompt
            num_turns: Number of dialogue turns to generate
            max_turn_length: Maximum length per turn
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of dialogue turns
        """
        dialogue = []
        current_context = context
        
        for i in range(num_turns):
            turn = self.complete_text(
                current_context,
                max_new_tokens=max_turn_length,
                temperature=temperature,
                **kwargs
            )
            
            # Extract only the new turn
            new_turn = turn[len(current_context):].strip()
            
            # Split by newline or period to get individual turn
            if '\n' in new_turn:
                new_turn = new_turn.split('\n')[0]
            
            dialogue.append(new_turn)
            current_context = turn + "\n"
        
        return dialogue
    
    def generate_variations(
        self,
        prompt: str,
        num_variations: int = 3,
        max_length: int = 100,
        temperature: float = 1.2,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple variations of text from the same prompt.
        
        Args:
            prompt: Input prompt
            num_variations: Number of variations to generate
            max_length: Maximum length per variation
            temperature: Sampling temperature (higher = more diverse)
            **kwargs: Additional generation parameters
            
        Returns:
            List of text variations
        """
        return self.generate(
            prompt,
            max_length=max_length,
            num_return_sequences=num_variations,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            **kwargs
        )
    
    def generate_with_constraints(
        self,
        prompt: str,
        required_words: Optional[List[str]] = None,
        forbidden_words: Optional[List[str]] = None,
        max_length: int = 100,
        **kwargs
    ) -> str:
        """
        Generate text with word constraints.
        
        Note: This is a simple implementation. For production use,
        consider using constrained beam search or similar advanced techniques.
        
        Args:
            prompt: Input prompt
            required_words: Words that must appear in output
            forbidden_words: Words that must not appear in output
            max_length: Maximum length
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        max_attempts = 10
        
        for _ in range(max_attempts):
            output = self.generate(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                **kwargs
            )[0]
            
            # Check constraints
            output_lower = output.lower()
            
            # Check required words
            if required_words:
                if not all(word.lower() in output_lower for word in required_words):
                    continue
            
            # Check forbidden words
            if forbidden_words:
                if any(word.lower() in output_lower for word in forbidden_words):
                    continue
            
            return output
        
        # If no valid output after max_attempts, return best effort
        logger.warning("Could not satisfy all constraints within max attempts")
        return self.generate(prompt, max_length=max_length, num_return_sequences=1)[0]
    
    def interactive_generation(
        self,
        initial_prompt: str = "",
        max_iterations: int = 10
    ):
        """
        Interactive text generation session.
        
        Args:
            initial_prompt: Starting prompt
            max_iterations: Maximum number of iterations
        """
        print("\n" + "="*80)
        print("GPT-2 INTERACTIVE GENERATION")
        print("="*80)
        print("Type your prompt and press Enter to generate.")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'continue' to continue from last generation.")
        print("="*80 + "\n")
        
        context = initial_prompt
        
        for i in range(max_iterations):
            if not context:
                user_input = input(f"\n[{i+1}] Enter prompt: ").strip()
            else:
                user_input = input(f"\n[{i+1}] Enter prompt (or 'continue'): ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nExiting interactive generation.")
                break
            
            if user_input.lower() == 'continue' and context:
                prompt = context
            elif user_input:
                prompt = user_input
                context = ""
            else:
                continue
            
            print(f"\nGenerating...")
            output = self.complete_text(prompt, max_new_tokens=100)
            print(f"\n{output}")
            
            context = output


def load_model(model_name: str = "gpt2", device: Optional[str] = None) -> GPT2Generator:
    """
    Convenience function to load GPT-2 model.
    
    Args:
        model_name: Model name to load
        device: Device to use
        
    Returns:
        GPT2Generator instance
    """
    return GPT2Generator(model_name=model_name, device=device)
