FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt requirements-stage2.txt requirements-stage3.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-stage2.txt -r requirements-stage3.txt \
    && python -m spacy download en_core_web_sm
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
