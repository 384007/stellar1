#!/usr/bin/env bash

# Works from repo root or from backend/
if [ -f "backend/main.py" ]; then
  cd backend
fi

exec uvicorn main_pro_late_strip_fix:app --host 0.0.0.0 --port ${PORT:-10000}
