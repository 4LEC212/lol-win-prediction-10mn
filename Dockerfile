FROM python:3.13.15-slim

LABEL org.opencontainers.image.source=https://github.com/4LEC212/lol-win-prediction-10mn

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# dependencies first so this layer stays cached when only the code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY lol_win/ lol_win/
COPY data/ data/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
