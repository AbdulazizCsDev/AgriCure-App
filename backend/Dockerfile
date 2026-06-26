# AgroCure app server (light / proxy mode) for Hugging Face Docker Spaces.
# Serves the website + /api/* and calls MODEL_API_URL for diagnosis.
FROM python:3.11-slim

WORKDIR /app

# Keep the image small: only the light deps (no torch/CLIP).
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# App code + frontend (models/ and other heavy files are excluded via .dockerignore)
COPY . .

# Hugging Face Spaces expose port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
