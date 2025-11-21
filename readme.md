# CommonForms FastAPI - AI-Powered PDF Form Field Detection

> Transform any PDF into a fillable form using AI-powered field detection with GPU acceleration

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready REST API service that automatically detects and adds form fields to PDF documents using the [CommonForms](https://github.com/jbarrow/commonforms) AI model. Features include GPU acceleration (3-5x faster), batch processing, comprehensive performance metrics, and load testing capabilities.

## ✨ Key Features

- 🤖 **AI-Powered Detection** - Automatically detect text boxes, checkboxes, and signature fields
- ⚡ **GPU Acceleration** - CUDA support with 3-5x performance boost
- 📦 **Batch Processing** - Process up to 10 PDFs simultaneously
- 📊 **Performance Metrics** - Track DPI, GPU memory usage, and processing times
- 🔐 **JWT Authentication** - Secure API access with token-based auth
- 🚀 **Load Testing** - Built-in benchmarking and stress testing tools
- 📈 **Persistent Metrics** - JSONL-based metrics storage for analysis
- 🎯 **Performance Testing** - Test different page sizes and concurrent loads

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#%EF%B8%8F-configuration)
- [API Endpoints](#-api-endpoints)
- [Usage Examples](#-usage-examples)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/abdullahdgrey8/commonForms
cd commonForms

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start server (single worker for development)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
# Open http://localhost:8000/docs in your browser
```

## 💻 Installation

### Prerequisites

- Python 3.12+
- NVIDIA GPU (optional, for CUDA acceleration)
- 4GB+ RAM (8GB recommended)
- 50MB free disk space per PDF

### Windows Installation

```powershell
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/FastAPI-CommonForms.git
cd FastAPI-CommonForms

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. For GPU support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Verify GPU (optional)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 6. Start server
uvicorn app.main:app --reload
```

### Linux/Mac Installation

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/FastAPI-CommonForms.git
cd FastAPI-CommonForms

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. For GPU support (NVIDIA GPU required)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Start server
uvicorn app.main:app --reload
```

### Docker Installation (Coming Soon)

```bash
docker pull your-username/commonforms-api
docker run -p 8000:8000 your-username/commonforms-api
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```env
# Application
APP_NAME=CommonForms API
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1  # Use 1 for dev, 2-3 with GPU, 4-8 with CPU

# Security
SECRET_KEY=your-secret-key-change-this-in-production-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# CORS (adjust for your frontend)
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]

# Model Cache
MODEL_CACHE_DIR=./data/models
PRELOAD_MODELS=true

# Processing
DEFAULT_DEVICE=cpu  # or cuda:0 for GPU
DEFAULT_MODEL=FFDNet-L
MAX_UPLOAD_SIZE=52428800  # 50MB

# Test Credentials (CHANGE IN PRODUCTION!)
TEST_USER_USERNAME=admin
TEST_USER_PASSWORD=changeme123
```

### Worker Configuration

**⚠️ Memory Management:**

| Configuration        | Workers | RAM Usage | VRAM Usage | Best For                |
| -------------------- | ------- | --------- | ---------- | ----------------------- |
| **Development**      | 1       | ~700MB    | -          | Testing, debugging      |
| **CPU Production**   | 4-8     | ~2.5GB    | -          | Multi-core CPUs         |
| **GPU Production**   | 2-3     | ~1.5GB    | 5-6GB      | NVIDIA GPUs (8GB+ VRAM) |
| **Colab/Low Memory** | 1       | ~700MB    | 2-3GB      | Free tier environments  |

```bash
# Development (auto-reload)
uvicorn app.main:app --reload

# Production CPU (4 workers)
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000

# Production GPU (2 workers to avoid OOM)
uvicorn app.main:app --workers 2 --host 0.0.0.0 --port 8000
```

## 📚 API Endpoints

### Authentication

#### Login

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "changeme123"
}

Response: {
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### PDF Processing

#### Single File Processing

```http
POST /api/v1/pdf/make-fillable
Authorization: Bearer {token}
Content-Type: multipart/form-data

Parameters:
- pdf: file (required)
- model: FFDNet-L | FFDNet-S (default: FFDNet-L)
- device: cpu | cuda:0 (default: cpu)
- confidence: 0.0-1.0 (default: 0.3)
- track_metrics: boolean (default: false)
- keep_existing: boolean (default: false)
- use_signature_fields: boolean (default: false)
- fast: boolean (default: false)
- multiline: boolean (default: false)

Response: PDF file with form fields
Headers:
- X-Request-ID: Unique request identifier
- X-Processing-Time: Processing time in seconds
- X-Device-Used: Device used (cpu/cuda:0)
- X-Input-DPI: Input PDF DPI (if track_metrics=true)
- X-Output-DPI: Output PDF DPI (if track_metrics=true)
- X-GPU-Peak-Memory-MB: Peak GPU memory (if GPU used)
```

#### Batch Processing

```http
POST /api/v1/pdf/make-fillable-batch
Authorization: Bearer {token}
Content-Type: multipart/form-data

Parameters:
- pdfs: file[] (required, max 10 files)
- model: FFDNet-L | FFDNet-S
- device: cpu | cuda:0
- parallel: boolean (default: true)
- track_metrics: boolean (default: false)
- ... (same parameters as single file)

Response: {
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "total_time": 12.45,
  "batch_id": "20241117_123456_abc123",
  "results": [...]
}
```

#### Check Available Devices

```http
GET /api/v1/pdf/devices
Authorization: Bearer {token}

Response: {
  "cpu_available": true,
  "cuda_available": true,
  "cuda_version": "12.4",
  "gpu_count": 1,
  "gpus": [{
    "id": 0,
    "name": "NVIDIA GeForce RTX 4070",
    "device_string": "cuda:0",
    "total_memory_gb": 8.0,
    "compute_capability": [8, 9]
  }],
  "recommended_device": "cuda:0"
}
```

### Metrics

#### Get Metrics Summary

```http
GET /api/v1/pdf/metrics/summary
Authorization: Bearer {token}

Response: {
  "total_requests": 150,
  "successful_requests": 148,
  "failed_requests": 2,
  "gpu_usage_count": 75,
  "avg_processing_time": 5.67,
  "min_processing_time": 3.21,
  "max_processing_time": 12.45
}
```

#### Get Recent Metrics

```http
GET /api/v1/pdf/metrics/recent?limit=50
Authorization: Bearer {token}

Response: {
  "count": 50,
  "metrics": [...]
}
```

### Performance Testing

#### Test Different Page Sizes

```http
POST /api/v1/performance/test-page-sizes
Authorization: Bearer {token}
Content-Type: application/json

{
  "page_sizes": ["A4", "Letter", "Legal", "A3"],
  "model": "FFDNet-L",
  "device": "cuda:0",
  "confidence": 0.3,
  "track_gpu": true
}

Response: {
  "total_tests": 4,
  "successful_tests": 4,
  "fastest_size": "A5",
  "slowest_size": "A3",
  "results": [...]
}
```

#### Load Testing

```http
POST /api/v1/performance/load-test
Authorization: Bearer {token}
Content-Type: multipart/form-data

Parameters:
- pdf: file (required)
- num_requests: integer (1-100)
- concurrent: boolean (true=parallel, false=sequential)
- model: FFDNet-L | FFDNet-S
- device: cpu | cuda:0

Response: {
  "total_requests": 10,
  "successful_requests": 10,
  "average_processing_time": 5.234,
  "requests_per_second": 1.85,
  "results": [...]
}
```

### Benchmarking

#### Concurrent Benchmark

```http
POST /api/v1/pdf/benchmark
Authorization: Bearer {token}
Content-Type: multipart/form-data

Parameters:
- pdf: file
- num_concurrent: integer (1-50)
- model: FFDNet-L | FFDNet-S
- device: cpu | cuda:0

Response: {
  "total_requests": 10,
  "successful_requests": 10,
  "average_processing_time": 5.123,
  "requests_per_second": 1.95,
  "metrics": [...]
}
```

#### Sequential Benchmark

```http
POST /api/v1/pdf/benchmark/sequential
Authorization: Bearer {token}

Same parameters as concurrent benchmark
```

### Health Check

```http
GET /api/v1/health

Response: {
  "status": "healthy",
  "service": "CommonForms API",
  "version": "1.0.0",
  "timestamp": "2024-11-17T10:30:00.000000",
  "environment": {
    "debug": false,
    "log_level": "INFO"
  }
}
```

## 💡 Usage Examples

### Python Client

```python
import requests
from pathlib import Path

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Login
response = requests.post(
    f"{BASE_URL}/api/v1/auth/login",
    json={"username": "admin", "password": "changeme123"}
)
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Check GPU availability
devices = requests.get(f"{BASE_URL}/api/v1/pdf/devices", headers=headers)
print(f"GPU Available: {devices.json()['cuda_available']}")

# 3. Process single PDF
with open("input.pdf", "rb") as f:
    files = {"pdf": f}
    data = {
        "model": "FFDNet-L",
        "device": "cuda:0",  # or "cpu"
        "track_metrics": "true",
        "confidence": "0.3"
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/pdf/make-fillable",
        files=files,
        data=data,
        headers=headers
    )

    # Save output
    with open("output.pdf", "wb") as out:
        out.write(response.content)

    # Check metrics in headers
    print(f"Processing time: {response.headers.get('X-Processing-Time')}s")
    print(f"GPU memory: {response.headers.get('X-GPU-Peak-Memory-MB')}MB")

# 4. Batch processing
files = [
    ("pdfs", open("file1.pdf", "rb")),
    ("pdfs", open("file2.pdf", "rb")),
    ("pdfs", open("file3.pdf", "rb"))
]
data = {"model": "FFDNet-L", "parallel": "true", "track_metrics": "true"}

response = requests.post(
    f"{BASE_URL}/api/v1/pdf/make-fillable-batch",
    files=files,
    data=data,
    headers=headers
)

result = response.json()
print(f"Processed {result['successful']}/{result['total_files']} files")
print(f"Total time: {result['total_time']:.2f}s")

# 5. Get metrics summary
summary = requests.get(
    f"{BASE_URL}/api/v1/pdf/metrics/summary",
    headers=headers
).json()
print(f"Average processing time: {summary['avg_processing_time']:.2f}s")
```

### cURL Examples

```bash
# Login
TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"changeme123"}' \
  | jq -r '.access_token')

# Process PDF with GPU
curl -X POST "http://localhost:8000/api/v1/pdf/make-fillable" \
  -H "Authorization: Bearer $TOKEN" \
  -F "pdf=@input.pdf" \
  -F "device=cuda:0" \
  -F "model=FFDNet-L" \
  -F "track_metrics=true" \
  -o output.pdf \
  -D headers.txt

# View response headers with metrics
cat headers.txt | grep "X-"

# Batch process
curl -X POST "http://localhost:8000/api/v1/pdf/make-fillable-batch" \
  -H "Authorization: Bearer $TOKEN" \
  -F "pdfs=@file1.pdf" \
  -F "pdfs=@file2.pdf" \
  -F "pdfs=@file3.pdf" \
  -F "parallel=true" \
  -F "device=cuda:0"

# Check GPU status
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/pdf/devices"

# Run load test
curl -X POST "http://localhost:8000/api/v1/performance/load-test" \
  -H "Authorization: Bearer $TOKEN" \
  -F "pdf=@test.pdf" \
  -F "num_requests=10" \
  -F "concurrent=true" \
  -F "device=cuda:0"
```

### Optimization Tips

1. **Use GPU** - 3-5x faster than CPU
2. **Adjust Workers** - More workers = better throughput (watch memory!)
3. **Use FFDNet-S** - 2x faster than FFDNet-L, slightly lower accuracy
4. **Enable Fast Mode** - ONNX acceleration on CPU
5. **Batch Processing** - Process multiple files efficiently
6. **Monitor Metrics** - Track and optimize based on actual usage

## 🐛 Troubleshooting

### API Returns 500 Error

**Problem:** Internal server error when processing PDF

**Check logs:**

```bash
# View logs
tail -f logs/app.log

# Or check console output
```

**Common causes:**

1. Invalid PDF file
2. Insufficient memory
3. Missing model files in `data/models/`
4. GPU/CUDA issues

### Slow Processing

**Problem:** Processing takes too long

**Solutions:**

1. Use GPU: `device=cuda:0`
2. Use faster model: `model=FFDNet-S`
3. Enable fast mode: `fast=true`
4. Increase workers (if CPU): `--workers 4`
5. Check system resources

## 📁 Project Structure

```
FastAPI-CommonForms/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   ├── auth.py          # Authentication
│   │   │   ├── pdf.py           # Single file processing
│   │   │   ├── batch.py         # Batch processing
│   │   │   ├── benchmark.py     # Benchmarking
│   │   │   ├── performance.py   # Performance testing
│   │   │   └── health.py        # Health checks
│   │   └── router.py            # API router
│   ├── core/
│   │   ├── config.py            # Configuration
│   │   ├── logging.py           # Logging setup
│   │   └── security.py          # JWT & auth
│   ├── middleware/
│   │   └── error_handler.py     # Error handling
│   ├── models/
│   │   ├── auth.py              # Auth models
│   │   ├── pdf.py               # PDF models
│   │   ├── batch.py             # Batch models
│   │   └── metrics.py           # Metrics models
│   ├── services/
│   │   ├── pdf_service.py       # PDF processing logic
│   │   ├── batch_services.py    # Batch processing
│   │   ├── metrics_tracker.py   # Metrics storage
│   │   └── model_cache.py       # Model management
│   ├── utils/
│   │   ├── dpi_analyzer.py      # DPI analysis
│   │   ├── gpu_monitor.py       # GPU monitoring
│   │   └── device_validator.py  # Device validation
│   └── main.py                  # Application entry
├── data/
│   ├── models/                  # Cached AI models
│   ├── metrics/                 # Performance metrics
│   └── batch/                   # Batch temp files
├── logs/                        # Application logs
├── .env                         # Environment variables
├── .gitignore                   # Git ignore file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```
