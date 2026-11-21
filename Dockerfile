FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data/docs ./data/docs
COPY evals ./evals
COPY finetune/data ./finetune/data
COPY scripts ./scripts
RUN useradd -m appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/api/health')"
# Secrets (GROQ_API_KEY, SECRET_KEY, SLACK_WEBHOOK_URL, SMTP_*...) come from environment variables only.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
