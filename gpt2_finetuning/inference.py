"""
GPT-2 text generation inference module.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Optional, Union
from .config import GPT2Config, GenerationConfig


class GPT2Generator:
    """GPT-2 text generator."""

    def __init__(
        self,
        model_name: str = "gpt2",
        config: Optional[GPT2Config] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize GPT-2 generator.

        Args:
            model_name: Name of GPT-2 model ('gpt2', 'gpt2-medium', etc.)
            config: GPT2Config object
            device: Device to run model on ('cpu', 'cuda')
        """
        self.config = config or GPT2Config(model_name=model_name)
        if device:
            self.config.device = device

        # Auto-detect device
        if self.config.device == "cpu" and torch.cuda.is_available():
            self.config.device = "cuda"

        print(f"Loading GPT-2 model: {self.config.model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.config.device)
        self.model.eval()
        print(f"Model loaded on {self.config.device}")

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt or list of prompts
            max_length: Maximum length of generated text
            min_length: Minimum length of generated text
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            generation_config: GenerationConfig object
            **kwargs: Additional generation parameters

        Returns:
            Generated text or list of texts
        """
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        # Use generation config if provided
        if generation_config:
            gen_kwargs = {
                "max_length": max_length or generation_config.max_length,
                "min_length": min_length or generation_config.min_length,
                "temperature": temperature or generation_config.temperature,
                "top_k": top_k or generation_config.top_k,
                "top_p": top_p or generation_config.top_p,
                "num_beams": num_beams or generation_config.num_beams,
                "do_sample": do_sample if do_sample is not None else generation_config.do_sample,
                "repetition_penalty": generation_config.repetition_penalty,
                "no_repeat_ngram_size": generation_config.no_repeat_ngram_size,
            }
        else:
            gen_kwargs = {
                "max_length": max_length or self.config.max_length,
                "min_length": min_length or self.config.min_length,
                "temperature": temperature or self.config.temperature,
                "top_k": top_k or self.config.top_k,
                "top_p": top_p or self.config.top_p,
                "num_beams": num_beams or self.config.num_beams,
                "do_sample": do_sample if do_sample is not None else self.config.do_sample,
                "repetition_penalty": self.config.repetition_penalty,
                "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            }

        gen_kwargs.update(kwargs)

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_texts if is_batch else generated_texts[0]

    def complete(
        self, prompt: str, max_length: int = 50, **kwargs
    ) -> str:
        """
        Complete a text prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum length including prompt
            **kwargs: Additional generation parameters

        Returns:
            Completed text
        """
        # Calculate max_new_tokens
        prompt_tokens = len(self.tokenizer.encode(prompt))
        actual_max_length = prompt_tokens + max_length

        return self.generate(
            prompt, max_length=actual_max_length, do_sample=True, **kwargs
        )

    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **kwargs,
    ) -> List[str]:
        """
        Generate text in batches.

        Args:
            prompts: List of prompts
            batch_size: Batch size for processing
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            batch_results = self.generate(batch, **kwargs)
            results.extend(batch_results)
        return results

    def generate_creative(self, prompt: str, **kwargs) -> str:
        """Generate creative text with high temperature."""
        config = GenerationConfig.creative()
        return self.generate(prompt, generation_config=config, **kwargs)

    def generate_focused(self, prompt: str, **kwargs) -> str:
        """Generate focused text with low temperature."""
        config = GenerationConfig.focused()
        return self.generate(prompt, generation_config=config, **kwargs)

    def generate_balanced(self, prompt: str, **kwargs) -> str:
        """Generate balanced text."""
        config = GenerationConfig.balanced()
        return self.generate(prompt, generation_config=config, **kwargs)


def main():
    """Demo of GPT-2 generator."""
    generator = GPT2Generator(model_name="gpt2")

    # Basic generation
    prompt = "Once upon a time"
    text = generator.generate(prompt, max_length=100, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Generated: {text}\n")

    # Creative generation
    creative = generator.generate_creative(prompt, max_length=100)
    print(f"Creative: {creative}\n")

    # Focused generation
    focused = generator.generate_focused(prompt, max_length=100)
    print(f"Focused: {focused}\n")

    # Text completion
    partial = "The future of artificial intelligence is"
    completion = generator.complete(partial, max_length=50)
    print(f"Completion: {completion}")


if __name__ == "__main__":
    main()
