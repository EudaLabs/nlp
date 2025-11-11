"""Question Answering inference module."""
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from typing import List, Dict, Optional, Union
from .config import QAConfig


class QASystem:
    """Question Answering system using transformers."""

    def __init__(
        self,
        model_name: str = "distilbert-base-cased-distilled-squad",
        config: Optional[QAConfig] = None,
    ):
        """Initialize QA system."""
        self.config = config or QAConfig(model_name=model_name)
        
        if self.config.device == "cpu" and torch.cuda.is_available():
            self.config.device = "cuda"

        print(f"Loading QA model: {self.config.model_name}")
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.config.model_name,
            device=0 if self.config.device == "cuda" else -1,
        )
        print(f"Model loaded on {self.config.device}")

    def answer(
        self,
        question: str,
        context: str,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> Union[Dict, List[Dict]]:
        """Answer a question given context."""
        top_k = top_k or self.config.top_k
        min_conf = min_confidence or self.config.min_confidence

        result = self.qa_pipeline(
            question=question,
            context=context,
            top_k=top_k,
            max_answer_len=self.config.max_answer_length,
        )

        if top_k == 1:
            if result["score"] >= min_conf:
                return result
            return {"answer": "No confident answer found", "score": 0.0}
        
        return [r for r in result if r["score"] >= min_conf]

    def batch_answer(
        self,
        questions: List[str],
        context: str,
        **kwargs,
    ) -> List[Dict]:
        """Answer multiple questions for the same context."""
        return [self.answer(q, context, **kwargs) for q in questions]


def main():
    """Demo QA system."""
    qa = QASystem()
    
    context = """
    Python is a high-level, interpreted programming language. It was created by
    Guido van Rossum and first released in 1991. Python emphasizes code readability
    and allows programmers to express concepts in fewer lines of code.
    """
    
    questions = [
        "Who created Python?",
        "When was Python first released?",
        "What does Python emphasize?"
    ]
    
    for question in questions:
        answer = qa.answer(question, context)
        print(f"Q: {question}")
        print(f"A: {answer['answer']} (score: {answer['score']:.2f})\n")


if __name__ == "__main__":
    main()
