"""
Question Answering models for extractive and generative QA.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline
)
from typing import List, Dict, Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractiveQA:
    """
    Extractive Question Answering using BERT-based models.
    Finds answer spans within provided context.
    """
    
    def __init__(
        self,
        model_name: str = "deepset/roberta-base-squad2",
        device: Optional[str] = None
    ):
        """
        Initialize extractive QA model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading extractive QA model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Create pipeline for convenience
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        logger.info("Model loaded successfully")
    
    def answer(
        self,
        question: str,
        context: str,
        top_k: int = 1,
        max_answer_length: int = 50,
        handle_impossible_answer: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Answer question based on context.
        
        Args:
            question: Question to answer
            context: Context containing the answer
            top_k: Number of answers to return
            max_answer_length: Maximum answer length
            handle_impossible_answer: Whether to handle impossible answers
            
        Returns:
            List of answer dictionaries with scores
        """
        result = self.qa_pipeline(
            question=question,
            context=context,
            top_k=top_k,
            max_answer_len=max_answer_length,
            handle_impossible_answer=handle_impossible_answer
        )
        
        # Ensure result is a list
        if isinstance(result, dict):
            result = [result]
        
        return result
    
    def batch_answer(
        self,
        questions: List[str],
        contexts: List[str],
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions
            contexts: List of corresponding contexts
            **kwargs: Additional parameters for answer()
            
        Returns:
            List of answer lists
        """
        if len(questions) != len(contexts):
            raise ValueError("Number of questions must match number of contexts")
        
        results = []
        for question, context in zip(questions, contexts):
            answer = self.answer(question, context, **kwargs)
            results.append(answer)
        
        return results
    
    def get_answer_with_confidence(
        self,
        question: str,
        context: str,
        confidence_threshold: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        """
        Get answer only if confidence exceeds threshold.
        
        Args:
            question: Question to answer
            context: Context text
            confidence_threshold: Minimum confidence score
            
        Returns:
            Answer dict or None if confidence too low
        """
        answers = self.answer(question, context, top_k=1)
        
        if answers and answers[0]['score'] >= confidence_threshold:
            return answers[0]
        
        return None


class GenerativeQA:
    """
    Generative Question Answering using T5/BART-based models.
    Generates answer text based on question and context.
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None
    ):
        """
        Initialize generative QA model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading generative QA model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def answer(
        self,
        question: str,
        context: str = "",
        max_length: int = 100,
        min_length: int = 10,
        num_beams: int = 4,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate answer to question.
        
        Args:
            question: Question to answer
            context: Optional context (for closed-book QA, leave empty)
            max_length: Maximum answer length
            min_length: Minimum answer length
            num_beams: Beam search width
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated answer
        """
        # Format input
        if context:
            input_text = f"question: {question} context: {context}"
        else:
            input_text = question
        
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
                early_stopping=True,
                **kwargs
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def batch_answer(
        self,
        questions: List[str],
        contexts: Optional[List[str]] = None,
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions
            contexts: Optional list of contexts
            batch_size: Batch size for processing
            **kwargs: Additional parameters for answer()
            
        Returns:
            List of answers
        """
        if contexts is None:
            contexts = [""] * len(questions)
        
        if len(questions) != len(contexts):
            raise ValueError("Number of questions must match number of contexts")
        
        answers = []
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            # Format inputs
            inputs = []
            for q, c in zip(batch_questions, batch_contexts):
                if c:
                    inputs.append(f"question: {q} context: {c}")
                else:
                    inputs.append(q)
            
            # Tokenize batch
            input_ids = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_ids,
                    **kwargs
                )
            
            # Decode
            batch_answers = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            answers.extend(batch_answers)
        
        return answers


class HybridQA:
    """
    Hybrid QA system combining extractive and generative approaches.
    """
    
    def __init__(
        self,
        extractive_model: str = "deepset/roberta-base-squad2",
        generative_model: str = "google/flan-t5-base",
        device: Optional[str] = None
    ):
        """
        Initialize hybrid QA system.
        
        Args:
            extractive_model: Model for extractive QA
            generative_model: Model for generative QA
            device: Device to use
        """
        self.extractive_qa = ExtractiveQA(extractive_model, device)
        self.generative_qa = GenerativeQA(generative_model, device)
    
    def answer(
        self,
        question: str,
        context: str,
        mode: str = "auto",
        extractive_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Answer question using hybrid approach.
        
        Args:
            question: Question to answer
            context: Context text
            mode: 'extractive', 'generative', or 'auto'
            extractive_threshold: Confidence threshold for extractive QA
            
        Returns:
            Answer dictionary with method used
        """
        if mode == "extractive":
            answer = self.extractive_qa.answer(question, context, top_k=1)[0]
            return {
                "answer": answer['answer'],
                "score": answer['score'],
                "method": "extractive"
            }
        
        elif mode == "generative":
            answer = self.generative_qa.answer(question, context)
            return {
                "answer": answer,
                "method": "generative"
            }
        
        else:  # auto mode
            # Try extractive first
            extractive_answer = self.extractive_qa.get_answer_with_confidence(
                question,
                context,
                extractive_threshold
            )
            
            if extractive_answer:
                return {
                    "answer": extractive_answer['answer'],
                    "score": extractive_answer['score'],
                    "method": "extractive"
                }
            else:
                # Fall back to generative
                generative_answer = self.generative_qa.answer(question, context)
                return {
                    "answer": generative_answer,
                    "method": "generative (fallback)"
                }


def load_extractive_qa(model_name: str = "deepset/roberta-base-squad2") -> ExtractiveQA:
    """Load extractive QA model."""
    return ExtractiveQA(model_name=model_name)


def load_generative_qa(model_name: str = "google/flan-t5-base") -> GenerativeQA:
    """Load generative QA model."""
    return GenerativeQA(model_name=model_name)


def load_hybrid_qa() -> HybridQA:
    """Load hybrid QA system."""
    return HybridQA()
