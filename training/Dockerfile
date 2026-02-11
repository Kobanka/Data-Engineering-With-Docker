FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

COPY train.py /workspace/train.py

ENV DATA_X=/workspace/data/engie_X.csv
ENV DATA_Y=/workspace/data/engie_Y.csv
ENV OUTDIR=/models

CMD ["python", "/workspace/train.py"]
