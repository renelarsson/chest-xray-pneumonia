FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY notebooks/artifacts/resnet50v2_ft.keras notebooks/artifacts/resnet50v2_ft.keras
EXPOSE 9696
CMD ["uvicorn", "service.predict:app", "--host", "0.0.0.0", "--port", "9696"]
