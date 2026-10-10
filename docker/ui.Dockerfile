# Streamlit chat UI. It only needs streamlit + requests; all ML runs in the API container.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN pip install "streamlit>=1.38" "requests>=2.32"

COPY app ./app

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
