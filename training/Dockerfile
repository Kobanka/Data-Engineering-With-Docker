FROM python:3.12-slim

WORKDIR /workspace

# Install system deps needed for ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (so Docker caches this layer)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

# Copy training script after deps
COPY train.py /workspace/train.py

ENV DATA_X=/workspace/data/engie_X.csv
ENV DATA_Y=/workspace/data/engie_Y.csv
ENV OUTDIR=/models

CMD ["python", "/workspace/train.py"]
