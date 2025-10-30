"""FastAPI application for NLP model deployment."""
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import settings
from .models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables for model and metrics
model = None
tokenizer = None
device = None
start_time = time.time()
request_count = 0
prediction_count = 0
total_latency = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    global model, tokenizer, device
    
    logger.info("Loading model...")
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from {settings.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="REST API for NLP model inference",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)


def predict_text(text: str) -> dict:
    """
    Perform prediction on a single text.
    
    Args:
        text: Input text
    
    Returns:
        Prediction result dictionary
    """
    global model, tokenizer, device
    
    # Tokenize
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=settings.max_length,
        return_tensors="pt",
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
    
    # Get prediction
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    
    result = {
        "label": predicted_class,
        "confidence": confidence,
    }
    
    # Add class name if available
    if settings.class_names:
        result["class_name"] = settings.class_names[predicted_class]
    
    # Add probabilities if not too many classes
    if settings.num_classes <= 10:
        result["probabilities"] = {
            f"class_{i}": prob.item()
            for i, prob in enumerate(probabilities[0])
        }
    
    return result


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    global model
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version=settings.app_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model information."""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name=settings.model_name,
        model_type=model.__class__.__name__,
        num_classes=settings.num_classes,
        class_names=settings.class_names,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
    tags=["Prediction"],
)
async def predict(request: PredictionRequest):
    """
    Predict class for a single text.
    
    Args:
        request: Prediction request with text
    
    Returns:
        Prediction response with result and processing time
    """
    global model, request_count, prediction_count, total_latency
    
    request_count += 1
    prediction_count += 1
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start = time.time()
        result = predict_text(request.text)
        processing_time = (time.time() - start) * 1000  # Convert to ms
        
        total_latency += processing_time
        
        return PredictionResponse(
            text=request.text,
            prediction=PredictionResult(**result),
            processing_time_ms=round(processing_time, 2),
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
    tags=["Prediction"],
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict classes for multiple texts.
    
    Args:
        request: Batch prediction request with list of texts
    
    Returns:
        Batch prediction response with results
    """
    global model, request_count, prediction_count, total_latency
    
    request_count += 1
    prediction_count += len(request.texts)
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start = time.time()
        predictions = []
        
        for text in request.texts:
            result = predict_text(text)
            predictions.append({
                "text": text,
                **result,
            })
        
        processing_time = (time.time() - start) * 1000  # Convert to ms
        total_latency += processing_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            processing_time_ms=round(processing_time, 2),
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def metrics():
    """Get API metrics."""
    global request_count, prediction_count, total_latency, start_time
    
    uptime = time.time() - start_time
    avg_latency = total_latency / request_count if request_count > 0 else 0
    
    return MetricsResponse(
        total_requests=request_count,
        total_predictions=prediction_count,
        average_latency_ms=round(avg_latency, 2),
        uptime_seconds=round(uptime, 2),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
    )
