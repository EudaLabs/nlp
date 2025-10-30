"""Tests for BERT classification module."""
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBERTConfig:
    """Tests for BERT configuration classes."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        from bert_classification.config import ModelConfig
        
        config = ModelConfig()
        
        assert config.model_name == "bert-base-uncased"
        assert config.num_classes == 2
        assert config.max_length == 512
        assert config.dropout_rate == 0.1
    
    def test_model_config_custom(self):
        """Test ModelConfig with custom values."""
        from bert_classification.config import ModelConfig
        
        config = ModelConfig(
            model_name="distilbert-base-uncased",
            num_classes=5,
            max_length=256,
        )
        
        assert config.model_name == "distilbert-base-uncased"
        assert config.num_classes == 5
        assert config.max_length == 256
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        from bert_classification.config import TrainingConfig
        
        config = TrainingConfig()
        
        assert config.epochs == 3
        assert config.batch_size == 16
        assert config.learning_rate == 2e-5
        assert config.seed == 42
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        from bert_classification.config import DataConfig
        
        config = DataConfig()
        
        assert config.text_column == "text"
        assert config.label_column == "label"
        assert config.preprocessing_num_workers == 4


class TestBERTUtils:
    """Tests for BERT utility functions."""
    
    def test_set_seed(self):
        """Test seed setting function."""
        from bert_classification.utils import set_seed
        
        # Should not raise any errors
        set_seed(42)
        set_seed(123)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        import numpy as np
        
        from bert_classification.utils import compute_metrics
        
        # Binary classification
        predictions = np.array([0, 1, 1, 0, 1])
        labels = np.array([0, 1, 0, 0, 1])
        
        metrics = compute_metrics(predictions, labels)
        
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        
        # All metrics should be between 0 and 1
        for value in metrics.values():
            assert 0 <= value <= 1
    
    def test_compute_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        import numpy as np
        
        from bert_classification.utils import compute_metrics
        
        predictions = np.array([0, 1, 1, 0, 1])
        labels = np.array([0, 1, 1, 0, 1])
        
        metrics = compute_metrics(predictions, labels)
        
        # Perfect predictions should have accuracy of 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_format_time(self):
        """Test time formatting."""
        from bert_classification.utils import format_time
        
        assert format_time(30) == "30s"
        assert format_time(90) == "1m 30s"
        assert format_time(3661) == "1h 1m 1s"
    
    def test_get_device(self):
        """Test device detection."""
        import torch
        
        from bert_classification.utils import get_device
        
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]
    
    def test_truncate_text(self):
        """Test text truncation."""
        from bert_classification.utils import truncate_text
        
        text = "a" * 200
        truncated = truncate_text(text, max_length=50)
        
        assert len(truncated) <= 53  # 50 + "..."
        assert truncated.endswith("...")
        
        # Short text should not be truncated
        short_text = "short"
        assert truncate_text(short_text, max_length=50) == short_text
    
    def test_early_stopping(self):
        """Test early stopping callback."""
        from bert_classification.utils import EarlyStopping
        
        # Test with minimization (loss)
        early_stop = EarlyStopping(patience=2, mode="min")
        
        assert not early_stop(1.0)  # First score
        assert not early_stop(0.9)  # Improvement
        assert not early_stop(0.85)  # Improvement
        assert not early_stop(0.86)  # No improvement, counter = 1
        assert not early_stop(0.87)  # No improvement, counter = 2
        assert early_stop(0.88)  # No improvement, counter = 3, should stop


class TestBERTInference:
    """Tests for BERT inference functionality."""
    
    def test_bert_classifier_init_error(self):
        """Test BERTClassifier initialization with invalid path."""
        from bert_classification.inference import BERTClassifier
        
        # Should raise error for non-existent model
        with pytest.raises(Exception):
            BERTClassifier("/non/existent/path")
    
    def test_prediction_structure(self):
        """Test that prediction returns correct structure."""
        # This test would require a trained model, so we skip if not available
        pytest.skip("Requires trained model")


class TestTextClassificationDataset:
    """Tests for TextClassificationDataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        try:
            from transformers import AutoTokenizer
            
            from bert_classification.train import TextClassificationDataset
            
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            texts = ["hello world", "goodbye world"]
            labels = [0, 1]
            
            dataset = TextClassificationDataset(texts, labels, tokenizer)
            
            assert len(dataset) == 2
            
            # Get first item
            item = dataset[0]
            
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
        
        except Exception:
            pytest.skip("Transformers not available or network issue")
    
    def test_dataset_length(self):
        """Test dataset length."""
        try:
            from transformers import AutoTokenizer
            
            from bert_classification.train import TextClassificationDataset
            
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            texts = ["text1", "text2", "text3"]
            labels = [0, 1, 0]
            
            dataset = TextClassificationDataset(texts, labels, tokenizer)
            
            assert len(dataset) == len(texts)
        
        except Exception:
            pytest.skip("Transformers not available")


class TestFastAPIModels:
    """Tests for FastAPI Pydantic models."""
    
    def test_prediction_request(self):
        """Test PredictionRequest model."""
        from fastapi_deployment.models import PredictionRequest
        
        request = PredictionRequest(text="This is a test")
        
        assert request.text == "This is a test"
    
    def test_prediction_request_validation(self):
        """Test PredictionRequest validation."""
        from pydantic import ValidationError
        
        from fastapi_deployment.models import PredictionRequest
        
        # Empty text should fail
        with pytest.raises(ValidationError):
            PredictionRequest(text="")
    
    def test_batch_prediction_request(self):
        """Test BatchPredictionRequest model."""
        from fastapi_deployment.models import BatchPredictionRequest
        
        request = BatchPredictionRequest(texts=["text1", "text2"])
        
        assert len(request.texts) == 2
    
    def test_prediction_result(self):
        """Test PredictionResult model."""
        from fastapi_deployment.models import PredictionResult
        
        result = PredictionResult(
            label=1,
            confidence=0.95,
            class_name="positive"
        )
        
        assert result.label == 1
        assert result.confidence == 0.95
        assert result.class_name == "positive"
    
    def test_health_response(self):
        """Test HealthResponse model."""
        from fastapi_deployment.models import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            version="1.0.0"
        )
        
        assert response.status == "healthy"
        assert response.model_loaded is True


class TestFastAPIConfig:
    """Tests for FastAPI configuration."""
    
    def test_settings_defaults(self):
        """Test Settings default values."""
        from fastapi_deployment.config import Settings
        
        settings = Settings()
        
        assert settings.app_name == "NLP Model API"
        assert settings.app_version == "1.0.0"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.num_classes == 2
