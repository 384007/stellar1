#!/usr/bin/env bash
set -e

# Works from repo root or from backend/
if [ -f "backend/requirements.txt" ]; then
  pip install -r backend/requirements.txt
elif [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
fi
