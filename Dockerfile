FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/openai/CLIP.git

# Copy codebase
COPY . .

# Download models at build time
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu')"
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create database and upload folders
RUN mkdir -p adve_v2/data/uploads adve_v2/data/main_index

# Environment setup
ENV PYTHONPATH=/app/adve_v2
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda

# Hugging Face Spaces port
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "adve.api.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
