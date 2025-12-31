import uuid
import os
import shutil
import logging
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uvicorn
import static_ffmpeg
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-api")

# Auto-configure FFmpeg (Critical for robust file handling)
static_ffmpeg.add_paths()

app = FastAPI(title="Optimized Offline STT")

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

# Create a warmup file
WARMUP_FILE = "warmup.wav"
if not os.path.exists(WARMUP_FILE):
    # Create a 1-second silent WAV file
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=16000:cl=mono",
        "-t", "1",
        "-acodec", "pcm_s16le",
        WARMUP_FILE
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

logger.info("Loading Whisper Model (base.en)...")
try:
    # Use English-only model with int8 quantization for CPU speed
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    logger.info("Model loaded.")
    
    # Warm-up the model (CRITICAL on Render)
    logger.info("Warming up model...")
    model.transcribe(WARMUP_FILE, beam_size=1, language="en")
    logger.info("Warm-up complete.")
except Exception as e:
    logger.error(f"Failed to load Whisper: {e}")
    raise

# Create thread pool for async processing
executor = ThreadPoolExecutor(max_workers=1)

def needs_normalization(path):
    """Check if audio needs normalization to 16kHz mono WAV."""
    return not path.lower().endswith(".wav")

def normalize_audio(input_path, output_path):
    """Normalize audio to 16kHz mono WAV format for better accuracy."""
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_audio(path):
    """Transcribe audio with optimized settings (SYNC function)."""
    # Get file extension
    ext = os.path.splitext(path)[1][1:] if os.path.splitext(path)[1] else "wav"
    
    # Create normalized audio path
    normalized_path = path.replace(f".{ext}", ".wav")
    
    # Only normalize if needed
    if needs_normalization(path):
        normalize_audio(path, normalized_path)
    else:
        normalized_path = path
    
    # Transcribe with optimized settings
    segments, info = model.transcribe(
        normalized_path,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=200, speech_pad_ms=100),
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0
    )
    
    # Use optimized response formatting
    text = "".join(s.text for s in segments).strip()
    
    # Clean up normalized file if it's different from the original
    if normalized_path != path and os.path.exists(normalized_path):
        os.remove(normalized_path)
    
    return text

@app.get("/")
def home():
    return {"status": "online", "mode": "optimized-http"}

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

        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            executor,
            transcribe_audio,
            path
        )
        
        logger.info(f"Transcribed: '{text}'")
        return {"text": text}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"text": "", "error": str(e)}
        
    finally:
        if os.path.exists(path):
            os.remove(path)

@app.on_event("shutdown")
def shutdown_event():
    """Clean shutdown to prevent zombie threads."""
    executor.shutdown(wait=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
