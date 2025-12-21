import os
import shutil
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
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
# "tiny" is efficient. "base" is better but larger.

logger.info(f"Loading Whisper Model ({MODEL_SIZE})...")
try:
    # int8 quantization for CPU speed
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise RuntimeError("Model loading failed")

@app.get("/")
async def health_check():
    """Health check."""
    return {
        "status": "online",
        "engine": f"faster-whisper ({MODEL_SIZE})",
        "features": ["auto-language", "duration-limit", "size-check"]
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file using local Whisper model.
    Accepts: webm, wav, mp3, m4a
    Returns: { "text": "transcribed text", "language": "detected_code" }
    """
    file_id = str(uuid.uuid4())
    input_ext = file.filename.split(".")[-1] if "." in file.filename else "webm"
    temp_input_path = os.path.join(TEMP_DIR, f"{file_id}.{input_ext}")
    wav_output_path = os.path.join(TEMP_DIR, f"{file_id}.wav")

    try:
        # 1. Save uploaded file
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Prevent empty/tiny audio uploads (< 2KB)
        if os.path.getsize(temp_input_path) < 2048:
            logger.warning("Rejected empty/small audio file")
            raise HTTPException(status_code=400, detail="Audio file too small or empty")

        # 3. Convert to 16kHz Mono WAV & Check Duration
        try:
            audio = AudioSegment.from_file(temp_input_path)
            
            # 4. Long audio safety (Limit to 60 seconds)
            if audio.duration_seconds > 60:
                logger.warning(f"Audio too long: {audio.duration_seconds}s")
                raise HTTPException(status_code=413, detail="Audio too long (max 60s)")
                
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_output_path, format="wav")
            
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Conversion Error: {e}")
            raise HTTPException(status_code=400, detail="Invalid audio format")

        # 5. Transcribe with Conditional VAD & Anti-Hallucination
        # Disable VAD for audio > 15s to prevent cutting off speech in long segments
        use_vad = audio.duration_seconds <= 15
        
        segments, info = model.transcribe(
            wav_output_path, 
            beam_size=3, 
            vad_filter=use_vad
        )
        
        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Collect text and remove hallucinated punctuation (e.g. "...")
        # Whisper Tiny sometimes hallucinates commas/periods in silence.
        text_parts = [
            segment.text.strip()
            for segment in segments
            if any(c.isalnum() for c in segment.text)
        ]
        transcribed_text = " ".join(text_parts)
        
        logger.info(f"Transcription: '{transcribed_text}'")
        
        return {
            "text": transcribed_text,
            "language": info.language,
            "duration": round(audio.duration_seconds, 5)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")
        
    finally:
        # Cleanup
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except: pass
        if os.path.exists(wav_output_path):
            try:
                os.remove(wav_output_path)
            except: pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
