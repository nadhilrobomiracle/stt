import uuid
import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uvicorn
import static_ffmpeg

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-api")

# Auto-configure FFmpeg (Critical for robust file handling)
static_ffmpeg.add_paths()

app = FastAPI(title="Simple Offline STT")

# CORS (Required for browser access)
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info("Loading Whisper Model (tiny)...")
try:
# int8 quantization for CPU speed
model = WhisperModel("tiny", device="cpu", compute_type="int8")
logger.info("Model loaded.")
except Exception as e:
logger.error(f"Failed to load Whisper: {e}")
raise

@app.get("/")
def home():
return {"status": "online", "mode": "simple-http"}

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
# Save file
file_id = str(uuid.uuid4())
# Try to keep extension or default to .wav
ext = file.filename.split(".")[-1] if "." in file.filename else "wav"
path = os.path.join(TEMP_DIR, f"{file_id}.{ext}")

try:
with open(path, "wb") as f:
shutil.copyfileobj(file.file, f)

# Transcribe (Optimized for speed)
# faster-whisper handles ffmpeg conversion internally if static_ffmpeg is set up
segments, info = model.transcribe(
path,
beam_size=1, # Fastest decoding
language="en", # Skip detection
vad_filter=False # No VAD for short commands
)

text = " ".join(s.text.strip() for s in segments if any(c.isalnum() for c in s.text))

logger.info(f"Transcribed: '{text}'")
return {"text": text}

except Exception as e:
logger.error(f"Error: {e}")
return {"text": "", "error": str(e)}

finally:
if os.path.exists(path):
os.remove(path)

if __name__ == "__main__":
port = int(os.environ.get("PORT", 5000))
uvicorn.run(app, host="0.0.0.0", port=port)
