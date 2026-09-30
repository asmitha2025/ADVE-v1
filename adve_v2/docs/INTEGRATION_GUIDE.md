# ADVE Enterprise v3.1 — Client Quick Integration Guide

Welcome to **Anchor-Delta Video Embedding (ADVE) Enterprise v3.1**. ADVE reduces heavy vision encoder compute costs by **68%+** while guaranteeing **>98.8% feature embedding precision** across surveillance, traffic, and indoor video streams.

---

## 1. Quick Start (2 Minutes)

### Option A: Running via Docker (Recommended)

```bash
# 1. Load the ADVE Enterprise Docker image
gunzip -c adve-enterprise-v3.1.tar.gz | docker load

# 2. Run container with GPU acceleration
docker run --gpus all -p 8000:8000 \
    -e ADVE_LICENSE_KEY=EVAL-TATA-2026 \
    -v /path/to/your/videos:/data:ro \
    adve-enterprise:v3.1
```

### Option B: Running via Python

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch FastAPI Server
python -m adve.api.server
```

---

## 2. Health & License Verification (30 Seconds)

### Health Check
```bash
curl http://localhost:8000/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "version": "3.1.0",
  "reconstructor": "reconstructor_v3.pt",
  "cuda_available": true
}
```

### License Validation
```bash
curl -X POST http://localhost:8000/api/v1/license/validate \
    -H "X-License-Key: EVAL-TATA-2026"
```

---

## 3. Process Your First Video (1 Minute)

Send a video embedding request to the REST API:

```bash
curl -X POST http://localhost:8000/api/v1/embed \
    -H "Content-Type: application/json" \
    -H "X-License-Key: EVAL-TATA-2026" \
    -d '{
        "video_path": "/data/your_video.mp4",
        "index_name": "test_index",
        "max_frames": 500
    }'
```

**Response Payload:**
```json
{
  "status": "success",
  "video_path": "/data/your_video.mp4",
  "index_name": "test_index",
  "total_frames": 331,
  "mean_cos_sim": 0.9886,
  "min_cos_sim": 0.8928,
  "encoder_savings_pct": 68.8,
  "effective_fps": 3.0,
  "processing_time_sec": 12.4
}
```

---

## 4. Python SDK Integration

Use the official lightweight Python client (`adve_sdk`):

```python
from sdk.adve_sdk import ADVEClient

# Initialize client
client = ADVEClient(
    base_url="http://localhost:8000",
    license_key="EVAL-TATA-2026"
)

# Process video
result = client.embed_video("/data/your_video.mp4")

# Inspect metrics
print(f"Status             : {result.status}")
print(f"Encoder Savings    : {result.encoder_savings_pct}%")
print(f"Embedding Precision: {result.mean_cos_sim}")
print(f"Minimum Precision  : {result.min_cos_sim}")
```

---

## 5. Configuration Overrides

Customize anchor refresh parameters in `config/client_override.yaml` or `adve/core/config.py`:

```yaml
# config/client_override.yaml
spatial_threshold: 0.30       # Spatial motion trigger threshold
appearance_threshold: 0.10    # Grayscale MSE structure trigger threshold
max_delta_frames: 15          # Max consecutive frames before keyframe anchor refresh
use_ema: true                 # Exponential Moving Average temporal smoothing
use_ego_motion: true          # ORB Homography camera pan/tilt compensation
```

---

## 6. Commercial Support & SLA

For integration assistance or enterprise license renewal:

- **Email Support**: `enterprise-support@adve-ai.com`
- **Support Hours**: IST 9:00 AM – 6:00 PM (Monday – Friday)
- **SLA Response Time**: < 4 hours for Tier-1 evaluation license holders
