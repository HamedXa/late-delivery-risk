# Serves the API and the demo page. Hugging Face Spaces expects port 7860 and user id 1000.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DELIVERY_RISK_HOME=/app

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY artifacts ./artifacts

RUN useradd --create-home --uid 1000 user && chown -R user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "delivery_risk.serve:app", "--host", "0.0.0.0", "--port", "7860"]
