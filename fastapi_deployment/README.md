# FastAPI NLP Model Deployment

Production-ready API deployment for NLP models using FastAPI.

## Overview

This project demonstrates how to deploy NLP models as RESTful APIs using FastAPI. It includes:
- Model serving with FastAPI
- Request validation with Pydantic
- Health checks and monitoring
- Swagger/OpenAPI documentation
- Docker containerization
- Load testing examples

## Features

- ✅ RESTful API for model inference
- ✅ Automatic API documentation (Swagger UI)
- ✅ Request/response validation
- ✅ Health check endpoints
- ✅ Error handling and logging
- ✅ CORS support
- ✅ Docker deployment
- ✅ Performance monitoring
- ✅ Batch prediction support

## Quick Start

### Installation

```bash
pip install fastapi uvicorn pydantic python-multipart
```

### Run the API Server

```bash
# Basic usage
python -m fastapi_deployment.app

# With custom settings
uvicorn fastapi_deployment.app:app --host 0.0.0.0 --port 8000 --reload
```

### Access API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 2. Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is amazing!"}'
```

Response:
```json
{
  "text": "This movie is amazing!",
  "prediction": {
    "label": 1,
    "confidence": 0.98,
    "class_name": "positive"
  },
  "processing_time_ms": 45.2
}
```

### 3. Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great product!",
      "Terrible experience",
      "It was okay"
    ]
  }'
```

Response:
```json
{
  "predictions": [
    {"text": "Great product!", "label": 1, "confidence": 0.95},
    {"text": "Terrible experience", "label": 0, "confidence": 0.92},
    {"text": "It was okay", "label": 1, "confidence": 0.65}
  ],
  "total_count": 3,
  "processing_time_ms": 123.4
}
```

### 4. Model Information

```bash
curl http://localhost:8000/model/info
```

Response:
```json
{
  "model_name": "bert-base-uncased",
  "model_type": "SequenceClassification",
  "num_classes": 2,
  "class_names": ["negative", "positive"]
}
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t nlp-api:latest .
```

### Run Container

```bash
docker run -p 8000:8000 nlp-api:latest
```

### Docker Compose

```bash
docker-compose up
```

## Configuration

Create a `.env` file:

```env
MODEL_PATH=./models/bert-imdb
MODEL_NAME=bert-base-uncased
NUM_CLASSES=2
MAX_LENGTH=512
BATCH_SIZE=32
HOST=0.0.0.0
PORT=8000
WORKERS=4
LOG_LEVEL=info
```

## Python Client Example

```python
import requests

# API URL
api_url = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{api_url}/predict",
    json={"text": "This is awesome!"}
)
print(response.json())

# Batch prediction
response = requests.post(
    f"{api_url}/predict/batch",
    json={
        "texts": [
            "Great movie!",
            "Waste of time",
            "Pretty good"
        ]
    }
)
print(response.json())
```

## Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test API performance
ab -n 1000 -c 10 -p request.json -T application/json \
   http://localhost:8000/predict
```

## Performance Optimization

### 1. Model Optimization

```python
# Use smaller models
model_name = "distilbert-base-uncased"

# Quantization
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# ONNX export
torch.onnx.export(model, ...)
```

### 2. Batch Processing

```python
# Process multiple requests in batches
@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    results = model.predict_batch(
        request.texts,
        batch_size=32
    )
    return results
```

### 3. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_cached(text: str):
    return model.predict(text)
```

## Monitoring

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

Response:
```json
{
  "total_requests": 1234,
  "total_predictions": 5678,
  "average_latency_ms": 45.2,
  "uptime_seconds": 3600
}
```

### Logging

```python
import logging

logger = logging.getLogger("nlp-api")
logger.info(f"Prediction made: {result}")
```

## Security Best Practices

1. **API Key Authentication**
```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")
```

2. **Rate Limiting**
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
```

3. **Input Validation**
```python
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
```

## Deployment Options

### 1. Local Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. Docker
```bash
docker run -p 8000:8000 nlp-api
```

### 3. Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
```

### 4. Cloud Platforms
- AWS Lambda + API Gateway
- Google Cloud Run
- Azure Container Instances
- Heroku

## Troubleshooting

**Model Loading Error:**
- Ensure model path is correct
- Check model compatibility
- Verify dependencies are installed

**High Latency:**
- Use batch processing
- Enable GPU if available
- Consider model optimization
- Add caching layer

**Out of Memory:**
- Reduce batch size
- Use smaller model
- Enable gradient checkpointing

## Project Structure

```
fastapi_deployment/
├── __init__.py
├── README.md
├── requirements.txt
├── app.py              # Main FastAPI application
├── models.py           # Pydantic models
├── config.py           # Configuration
├── utils.py            # Utility functions
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose
├── .env.example        # Environment variables template
└── tests/
    └── test_api.py     # API tests
```

## Testing

```bash
# Install test dependencies
pip install pytest httpx

# Run tests
pytest tests/

# With coverage
pytest --cov=fastapi_deployment tests/
```

## Contributing

Contributions are welcome! Please refer to the main repository's contributing guidelines.

## License

This project is part of the EudaLabs NLP repository.

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Docker Documentation](https://docs.docker.com/)
