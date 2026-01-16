FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY notebooks/artifacts/resnet50v2_ft.keras notebooks/artifacts/resnet50v2_ft.keras
EXPOSE 9696
CMD ["uvicorn", "service.predict:app", "--host", "0.0.0.0", "--port", "9696"]

## Staged Dockerfile (disk space friendly): 
# Stage 1: Build environment
# FROM python:3.12-slim AS builder
# WORKDIR /app
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
# FROM python:3.12-slim
# WORKDIR /app
# COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
# COPY . .
# COPY notebooks/artifacts/resnet50v2_ft.keras notebooks/artifacts/resnet50v2_ft.keras
# EXPOSE 9696
# CMD ["uvicorn", "service.predict:app", "--host", "0.0.0.0", "--port", "9696"]
