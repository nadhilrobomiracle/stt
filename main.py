import os
import shutil
import uuid
import logging
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uvicorn
import static_ffmpeg

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-api")

# Auto-configure FFmpeg
static_ffmpeg.add_paths()

# Initialize FastAPI App
app = FastAPI(title="Offline Whisper STT API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)
MODEL_SIZE = "tiny" 

logger.info(f"Loading Whisper Model ({MODEL_SIZE})...")
try:
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise RuntimeError("Model loading failed")

@app.get("/")
async def health_check():
    """Health check."""
    return {"status": "online", "engine": f"faster-whisper ({MODEL_SIZE})", "fix": "No pydub/audioop"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file using local Whisper model.
    """
    file_id = str(uuid.uuid4())
    input_ext = file.filename.split(".")[-1] if "." in file.filename else "webm"
    temp_input_path = os.path.join(TEMP_DIR, f"{file_id}.{input_ext}")
    wav_output_path = os.path.join(TEMP_DIR, f"{file_id}.wav")

    try:
        # 1. Save uploaded file
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if os.path.getsize(temp_input_path) < 2048:
            raise HTTPException(status_code=400, detail="Audio file too small")

        # 2. Convert to 16kHz Mono WAV using native FFmpeg (Avoiding pydub crash)
        # Using -y to overwrite, -ar 16000 for rate, -ac 1 for mono
        command = [
            "ffmpeg", "-i", temp_input_path, 
            "-ar", "16000", 
            "-ac", "1", 
            "-y", 
            wav_output_path
        ]
        
        # Run conversion
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg Error: {result.stderr}")
            raise HTTPException(status_code=400, detail="Audio format not supported")

        # 3. Transcribe
        segments, info = model.transcribe(
            wav_output_path, 
            beam_size=3, 
            vad_filter=True
        )
        
        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # 4. Filter and Combine
        text_parts = [
            segment.text.strip()
            for segment in segments
            if any(c.isalnum() for c in segment.text)
        ]
        transcribed_text = " ".join(text_parts)
        
        logger.info(f"Transcription: '{transcribed_text}'")
        
        return {
            "text": transcribed_text,
            "language": info.language
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")
        
    finally:
        # Cleanup
        if os.path.exists(temp_input_path):
            try: os.remove(temp_input_path)
            except: pass
        if os.path.exists(wav_output_path):
            try: os.remove(wav_output_path)
            except: pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
