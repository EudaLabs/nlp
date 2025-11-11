"""
Advanced text classification models including zero-shot and multi-label classification.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZeroShotClassifier:
    """
    Zero-shot text classification without training.
    Classify text into arbitrary categories.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: Optional[str] = None
    ):
        """
        Initialize zero-shot classifier.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading zero-shot classifier: {model_name}")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        logger.info("Model loaded successfully")
    
    def classify(
        self,
        text: str,
        candidate_labels: List[str],
        multi_label: bool = False,
        hypothesis_template: str = "This text is about {}."
    ) -> Dict[str, Any]:
        """
        Classify text into candidate labels.
        
        Args:
            text: Text to classify
            candidate_labels: List of possible labels
            multi_label: Whether text can have multiple labels
            hypothesis_template: Template for hypothesis (use {} for label)
            
        Returns:
            Dictionary with labels and scores
        """
        result = self.classifier(
            text,
            candidate_labels,
            multi_label=multi_label,
            hypothesis_template=hypothesis_template
        )
        
        return result
    
    def classify_batch(
        self,
        texts: List[str],
        candidate_labels: List[str],
        multi_label: bool = False,
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple texts.
        
        Args:
            texts: List of texts to classify
            candidate_labels: List of possible labels
            multi_label: Whether texts can have multiple labels
            batch_size: Batch size for processing
            
        Returns:
            List of classification results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [
                self.classify(text, candidate_labels, multi_label)
                for text in batch
            ]
            results.extend(batch_results)
        
        return results
    
    def get_top_label(
        self,
        text: str,
        candidate_labels: List[str],
        **kwargs
    ) -> Tuple[str, float]:
        """
        Get the top predicted label and its score.
        
        Args:
            text: Text to classify
            candidate_labels: List of possible labels
            **kwargs: Additional parameters for classify()
            
        Returns:
            Tuple of (label, score)
        """
        result = self.classify(text, candidate_labels, **kwargs)
        return result['labels'][0], result['scores'][0]


class MultiLabelClassifier:
    """
    Multi-label text classification.
    Assigns multiple labels to a single text.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 10,
        device: Optional[str] = None
    ):
        """
        Initialize multi-label classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of labels
            device: Device to use
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading multi-label classifier: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.label_names = None
        
        logger.info("Model loaded successfully")
    
    def set_label_names(self, label_names: List[str]):
        """Set names for labels."""
        if len(label_names) != self.num_labels:
            raise ValueError(f"Expected {self.num_labels} labels, got {len(label_names)}")
        self.label_names = label_names
    
    def predict(
        self,
        text: str,
        threshold: float = 0.5,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Predict labels for text.
        
        Args:
            text: Input text
            threshold: Probability threshold for positive labels
            top_k: Return only top k labels (None for all above threshold)
            
        Returns:
            Dictionary with predicted labels and scores
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Get labels above threshold
        if top_k:
            top_indices = np.argsort(probs)[-top_k:][::-1]
            labels_scores = [
                (i, float(probs[i]))
                for i in top_indices
            ]
        else:
            labels_scores = [
                (i, float(score))
                for i, score in enumerate(probs)
                if score >= threshold
            ]
        
        # Format result
        if self.label_names:
            result = {
                "labels": [self.label_names[i] for i, _ in labels_scores],
                "scores": [score for _, score in labels_scores]
            }
        else:
            result = {
                "label_ids": [i for i, _ in labels_scores],
                "scores": [score for _, score in labels_scores]
            }
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        threshold: float = 0.5,
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Predict labels for multiple texts.
        
        Args:
            texts: List of texts
            threshold: Probability threshold
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [
                self.predict(text, threshold)
                for text in batch
            ]
            results.extend(batch_results)
        
        return results


class AspectBasedSentiment:
    """
    Aspect-based sentiment analysis.
    Analyzes sentiment towards specific aspects/topics in text.
    """
    
    def __init__(
        self,
        model_name: str = "yangheng/deberta-v3-base-absa-v1.1",
        device: Optional[str] = None
    ):
        """
        Initialize aspect-based sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading aspect-based sentiment model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ABSA model: {e}. Using zero-shot as fallback.")
            self.tokenizer = None
            self.model = None
            self.zero_shot = ZeroShotClassifier()
    
    def analyze(
        self,
        text: str,
        aspect: str
    ) -> Dict[str, Any]:
        """
        Analyze sentiment towards a specific aspect.
        
        Args:
            text: Text to analyze
            aspect: Aspect to analyze sentiment for
            
        Returns:
            Dictionary with sentiment and score
        """
        if self.model is None:
            # Fallback to zero-shot
            result = self.zero_shot.classify(
                f"The aspect '{aspect}' in: {text}",
                ["positive", "negative", "neutral"],
                multi_label=False
            )
            return {
                "aspect": aspect,
                "sentiment": result['labels'][0],
                "score": result['scores'][0]
            }
        
        # Use ABSA model
        input_text = f"{text} [SEP] {aspect}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()
        
        # Map class to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map.get(pred_class, "unknown")
        
        return {
            "aspect": aspect,
            "sentiment": sentiment,
            "score": confidence
        }
    
    def analyze_multiple_aspects(
        self,
        text: str,
        aspects: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple aspects.
        
        Args:
            text: Text to analyze
            aspects: List of aspects
            
        Returns:
            List of sentiment results
        """
        return [self.analyze(text, aspect) for aspect in aspects]


def load_zero_shot_classifier(model_name: str = "facebook/bart-large-mnli") -> ZeroShotClassifier:
    """Load zero-shot classifier."""
    return ZeroShotClassifier(model_name=model_name)


def load_multi_label_classifier(
    model_name: str = "bert-base-uncased",
    num_labels: int = 10
) -> MultiLabelClassifier:
    """Load multi-label classifier."""
    return MultiLabelClassifier(model_name=model_name, num_labels=num_labels)


def load_aspect_sentiment_analyzer(
    model_name: str = "yangheng/deberta-v3-base-absa-v1.1"
) -> AspectBasedSentiment:
    """Load aspect-based sentiment analyzer."""
    return AspectBasedSentiment(model_name=model_name)
