"""Pydantic models for request/response validation."""
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    
    text: str = Field(
        ...,
        description="Text to classify",
        min_length=1,
        max_length=5000,
        examples=["This movie is amazing!"],
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    
    texts: List[str] = Field(
        ...,
        description="List of texts to classify",
        min_length=1,
        max_length=100,
    )


class PredictionResult(BaseModel):
    """Result of a single prediction."""
    
    label: int = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    class_name: Optional[str] = Field(None, description="Human-readable class name")
    probabilities: Optional[dict] = Field(None, description="Class probabilities")


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    
    text: str = Field(..., description="Input text")
    prediction: PredictionResult = Field(..., description="Prediction result")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    
    predictions: List[dict] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    num_classes: int = Field(..., description="Number of classes")
    class_names: Optional[List[str]] = Field(None, description="Names of classes")


class MetricsResponse(BaseModel):
    """Metrics response."""
    
    total_requests: int = Field(..., description="Total number of requests")
    total_predictions: int = Field(..., description="Total number of predictions")
    average_latency_ms: float = Field(..., description="Average latency in milliseconds")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
