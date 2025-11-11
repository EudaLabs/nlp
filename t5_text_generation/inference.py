"""
T5 text generation inference module.
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Optional, Union
from .config import T5Config, TaskConfig


class T5Generator:
    """T5-based text generator for various NLP tasks."""

    def __init__(
        self,
        model_name: str = "t5-base",
        config: Optional[T5Config] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize T5 generator.

        Args:
            model_name: Name of the T5 model to use
            config: T5Config object
            device: Device to run model on ('cpu', 'cuda')
        """
        self.config = config or T5Config(model_name=model_name)
        if device:
            self.config.device = device

        # Automatically detect device if not specified
        if self.config.device == "cpu" and torch.cuda.is_available():
            self.config.device = "cuda"

        print(f"Loading T5 model: {self.config.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )
        self.model.to(self.config.device)
        self.model.eval()
        print(f"Model loaded on {self.config.device}")

    def generate(
        self,
        input_text: Union[str, List[str]],
        task_config: Optional[TaskConfig] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text using T5 model.

        Args:
            input_text: Input text or list of texts
            task_config: Task-specific configuration
            max_length: Maximum length of generated text
            min_length: Minimum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text or list of texts
        """
        is_batch = isinstance(input_text, list)
        texts = input_text if is_batch else [input_text]

        # Apply task prefix if provided
        if task_config:
            texts = [task_config.prefix + text for text in texts]

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Prepare generation parameters
        gen_kwargs = {
            "max_length": max_length or task_config.max_length if task_config else self.config.max_length,
            "min_length": min_length or task_config.min_length if task_config else self.config.min_length,
            "num_beams": num_beams or task_config.num_beams if task_config else self.config.num_beams,
            "temperature": temperature or self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "early_stopping": self.config.early_stopping,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "length_penalty": self.config.length_penalty,
        }
        gen_kwargs.update(kwargs)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_texts if is_batch else generated_texts[0]

    def summarize(
        self,
        text: Union[str, List[str]],
        max_length: int = 150,
        min_length: int = 50,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Summarize text.

        Args:
            text: Input text or list of texts
            max_length: Maximum summary length
            min_length: Minimum summary length
            **kwargs: Additional generation parameters

        Returns:
            Summary or list of summaries
        """
        task_config = TaskConfig.summarization(max_length, min_length)
        return self.generate(text, task_config=task_config, **kwargs)

    def paraphrase(
        self, text: Union[str, List[str]], max_length: int = 256, **kwargs
    ) -> Union[str, List[str]]:
        """
        Paraphrase text.

        Args:
            text: Input text or list of texts
            max_length: Maximum paraphrase length
            **kwargs: Additional generation parameters

        Returns:
            Paraphrased text or list of texts
        """
        task_config = TaskConfig.paraphrase(max_length)
        return self.generate(text, task_config=task_config, **kwargs)

    def translate(
        self,
        text: Union[str, List[str]],
        source_lang: str = "English",
        target_lang: str = "German",
        max_length: int = 256,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Translate text.

        Args:
            text: Input text or list of texts
            source_lang: Source language
            target_lang: Target language
            max_length: Maximum translation length
            **kwargs: Additional generation parameters

        Returns:
            Translated text or list of texts
        """
        task_config = TaskConfig.translation(source_lang, target_lang, max_length)
        return self.generate(text, task_config=task_config, **kwargs)

    def answer_question(
        self,
        question: str,
        context: str,
        max_length: int = 128,
        **kwargs,
    ) -> str:
        """
        Answer question based on context.

        Args:
            question: Question to answer
            context: Context containing the answer
            max_length: Maximum answer length
            **kwargs: Additional generation parameters

        Returns:
            Generated answer
        """
        task_config = TaskConfig.question_answering(max_length)
        input_text = f"{question} context: {context}"
        return self.generate(input_text, task_config=task_config, **kwargs)

    def batch_generate(
        self,
        texts: List[str],
        batch_size: int = 8,
        task_config: Optional[TaskConfig] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text in batches for efficiency.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            task_config: Task-specific configuration
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = self.generate(batch, task_config=task_config, **kwargs)
            results.extend(batch_results)
        return results


def main():
    """Demo of T5 generator."""
    generator = T5Generator(model_name="t5-small")

    # Summarization demo
    text = """
    The T5 model was presented in Exploring the Limits of Transfer Learning with a
    Unified Text-to-Text Transformer by Colin Raffel, Noam Shazeer, Adam Roberts,
    Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.
    T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised
    and supervised tasks and for which each task is converted into a text-to-text format.
    """
    summary = generator.summarize(text.strip(), max_length=50)
    print(f"Summary: {summary}\n")

    # Paraphrase demo
    original = "The weather is beautiful today."
    paraphrased = generator.paraphrase(original)
    print(f"Original: {original}")
    print(f"Paraphrased: {paraphrased}\n")

    # Translation demo
    english_text = "Hello, how are you today?"
    german = generator.translate(english_text, "English", "German")
    print(f"English: {english_text}")
    print(f"German: {german}")


if __name__ == "__main__":
    main()
