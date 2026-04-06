# Render (and any Docker-based) API runtime: system ffmpeg + Python deps.
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -hide_banner -filters 2>&1 | grep -qw minterpolate \
    || (echo "ERROR: ffmpeg without minterpolate — Pro v3 MCI / true 240 will not run" && exit 1)

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
# Pro v3: prefer motion-compensated 240fps when duration rules allow (override with -e STELLAR_PROV3_USE_FAST_240FPS=1).
ENV STELLAR_PROV3_USE_FAST_240FPS=0
CMD ["sh", "-c", "exec uvicorn main_pro_late_strip_fix:app --host 0.0.0.0 --port ${PORT:-10000}"]
