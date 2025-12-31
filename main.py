import uuid
import os
import shutil
import logging

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uvicorn
import static_ffmpeg

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-api")

# ---------------- FFMPEG ----------------
static_ffmpeg.add_paths()

# ---------------- APP ----------------
app = FastAPI(title="Simple Offline STT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- TEMP DIR ----------------
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
logger.info("Loading Whisper Model (tiny)...")
try:
    model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"  # fastest for CPU
    )
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {
        "status": "online",
        "engine": "faster-whisper",
        "model": "tiny"
    }

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1] if "." in file.filename else "wav"
    path = os.path.join(TEMP_DIR, f"{file_id}.{ext}")

    try:
        # Save uploaded file
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Transcribe
        segments, info = model.transcribe(
            path,
            beam_size=1,      # fastest
            language="en",    # skip detection
            vad_filter=False
        )

        text = " ".join(
            s.text.strip()
            for s in segments
            if any(c.isalnum() for c in s.text)
        )

        logger.info(f"Transcribed: {text}")
        return {"text": text}

    except Exception as e:
        logger.error(f"STT error: {e}")
        return {"text": "", "error": str(e)}

    finally:
        if os.path.exists(path):
            os.remove(path)

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
