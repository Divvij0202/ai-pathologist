# Builds for whatever platform you build on; on a Graviton instance this is arm64.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pathagent pathagent
COPY service service
COPY scripts scripts
COPY models models
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
