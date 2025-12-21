FROM python:3.11-slim

# Install system dependencies
# ffmpeg: redundant with static-ffmpeg but good for debugging/fallback
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create temp directory
RUN mkdir -p temp_audio

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
