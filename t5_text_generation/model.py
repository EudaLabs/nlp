"""
T5 Model wrapper for various text generation tasks.
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T5TextGenerator:
    """
    T5 model wrapper for multiple text generation tasks.
    
    Supports:
    - Summarization
    - Translation
    - Paraphrasing
    - Question generation
    - Grammar correction
    """
    
    def __init__(
        self,
        model_name: str = "t5-base",
        device: Optional[str] = None
    ):
        """
        Initialize T5 model.
        
        Args:
            model_name: HuggingFace model name (t5-small, t5-base, t5-large)
            device: Device to use (cuda/cpu). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading T5 model: {model_name} on {self.device}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
    
    def generate(
        self,
        text: str,
        task_prefix: str = "",
        max_length: int = 512,
        min_length: int = 10,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text using T5 model.
        
        Args:
            text: Input text
            task_prefix: Task-specific prefix (e.g., "summarize:", "translate English to German:")
            max_length: Maximum length of generated text
            min_length: Minimum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
            length_penalty: Length penalty for beam search
            no_repeat_ngram_size: Size of n-grams to prevent repetition
            early_stopping: Whether to stop when all beams finish
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Prepare input with task prefix
        input_text = f"{task_prefix}{text}" if task_prefix else text
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                **kwargs
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        **kwargs
    ) -> str:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            **kwargs: Additional generation parameters
            
        Returns:
            Summary text
        """
        return self.generate(
            text,
            task_prefix="summarize: ",
            max_length=max_length,
            min_length=min_length,
            **kwargs
        )
    
    def translate(
        self,
        text: str,
        source_lang: str = "English",
        target_lang: str = "German",
        max_length: int = 512,
        **kwargs
    ) -> str:
        """
        Translate text.
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            max_length: Maximum translation length
            **kwargs: Additional generation parameters
            
        Returns:
            Translated text
        """
        task_prefix = f"translate {source_lang} to {target_lang}: "
        return self.generate(
            text,
            task_prefix=task_prefix,
            max_length=max_length,
            **kwargs
        )
    
    def paraphrase(
        self,
        text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Paraphrase text.
        
        Args:
            text: Text to paraphrase
            max_length: Maximum paraphrase length
            num_return_sequences: Number of paraphrases to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of paraphrased texts
        """
        input_text = f"paraphrase: {text}"
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                num_beams=num_return_sequences,
                temperature=1.5,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                **kwargs
            )
        
        paraphrases = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        return paraphrases
    
    def generate_question(
        self,
        context: str,
        answer: Optional[str] = None,
        max_length: int = 64,
        **kwargs
    ) -> str:
        """
        Generate question from context and optional answer.
        
        Args:
            context: Context text
            answer: Answer (if provided, generates specific question)
            max_length: Maximum question length
            **kwargs: Additional generation parameters
            
        Returns:
            Generated question
        """
        if answer:
            input_text = f"generate question: answer: {answer} context: {context}"
        else:
            input_text = f"generate question: {context}"
        
        return self.generate(
            input_text,
            task_prefix="",
            max_length=max_length,
            **kwargs
        )
    
    def correct_grammar(
        self,
        text: str,
        max_length: int = 512,
        **kwargs
    ) -> str:
        """
        Correct grammar in text.
        
        Args:
            text: Text with potential grammar errors
            max_length: Maximum corrected text length
            **kwargs: Additional generation parameters
            
        Returns:
            Grammar-corrected text
        """
        return self.generate(
            text,
            task_prefix="grammar: ",
            max_length=max_length,
            **kwargs
        )
    
    def batch_generate(
        self,
        texts: List[str],
        task_prefix: str = "",
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple inputs in batches.
        
        Args:
            texts: List of input texts
            task_prefix: Task-specific prefix
            batch_size: Batch size for processing
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_inputs = [f"{task_prefix}{text}" if task_prefix else text for text in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **kwargs
                )
            
            # Decode
            batch_results = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            results.extend(batch_results)
        
        return results


def load_model(model_name: str = "t5-base", device: Optional[str] = None) -> T5TextGenerator:
    """
    Convenience function to load T5 model.
    
    Args:
        model_name: Model name to load
        device: Device to use
        
    Returns:
        T5TextGenerator instance
    """
    return T5TextGenerator(model_name=model_name, device=device)
