import os
import shutil
import uuid
import logging
import subprocess
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
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
app = FastAPI(title="Streaming Whisper STT API")

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
        "features": ["offline", "conditional-vad", "anti-hallucination"]
    }

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

        # Check for empty file
        if os.path.getsize(temp_input_path) < 2048:
            raise HTTPException(status_code=400, detail="Audio file too small")

        # 2. Convert to 16kHz Mono WAV (using native FFmpeg)
        # -y: overwrite, -ar: rate, -ac: channels
        command_convert = [
            "ffmpeg", "-i", temp_input_path, 
            "-ar", "16000", 
            "-ac", "1", 
            "-y", 
            wav_output_path
        ]
        
        result = subprocess.run(command_convert, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg Error: {result.stderr}")
            raise HTTPException(status_code=400, detail="Audio format not supported")

        # 3. Transcribe (Optimized for Speed)
        # - beam_size=1: Greedy search (Fastest)
        # - language="en": Force English (Skip detection)
        # - vad_filter=False: Skip VAD overhead (safe since we handle silence via text cleanup)
        
        segments, info = model.transcribe(
            wav_output_path, 
            beam_size=1, 
            vad_filter=False,
            language="en"
        )
        
        # 4. Filter and Combine (Anti-Hallucination)
        text_parts = [
            segment.text.strip()
            for segment in segments
            if any(c.isalnum() for c in segment.text)
        ]
        transcribed_text = " ".join(text_parts)
        
        return {
            "text": transcribed_text,
            "language": "en"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")
        
@app.websocket("/ws/transcribe")
async def websocket_stt(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"WS connected: {session_id}")

    pcm_buffer = bytearray()
    last_process = asyncio.get_event_loop().time()
    last_text = ""
    
    # Process Interval: 0.9s for snappier response
    PROCESS_INTERVAL = 0.9 
    
    # Keep last 0.5s of audio (overlap) to maintain context
    KEEP_BYTES = int(16000 * 2 * 0.5)
    
    # Safety: Max buffer to prevent memory spikes (3s)
    MAX_BUFFER_BYTES = int(16000 * 2 * 3)
    
    # Min audio to process (0.6s) to avoid tiny fragment errors
    MIN_BYTES = int(16000 * 2 * 0.6)

    try:
        while True:
            chunk = await websocket.receive_bytes()
            pcm_buffer.extend(chunk)

            # Safety: Cap buffer size
            if len(pcm_buffer) > MAX_BUFFER_BYTES:
                pcm_buffer[:] = pcm_buffer[-MAX_BUFFER_BYTES:]

            now = asyncio.get_event_loop().time()

            # Skip if processing interval not met
            if now - last_process < PROCESS_INTERVAL:
                continue

            # Skip if buffer too small
            if len(pcm_buffer) < MIN_BYTES:
                continue

            last_process = now
            wav_path = os.path.join(TEMP_DIR, f"{session_id}.wav")

            # Write WAV header + PCM
            with open(wav_path, "wb") as f:
                f.write(b"RIFF")
                f.write((36 + len(pcm_buffer)).to_bytes(4, "little"))
                f.write(b"WAVEfmt ")
                f.write((16).to_bytes(4, "little"))
                f.write((1).to_bytes(2, "little"))
                f.write((1).to_bytes(2, "little"))
                f.write((16000).to_bytes(4, "little"))
                f.write((16000 * 2).to_bytes(4, "little"))
                f.write((2).to_bytes(2, "little"))
                f.write((16).to_bytes(2, "little"))
                f.write(b"data")
                f.write(len(pcm_buffer).to_bytes(4, "little"))
                f.write(pcm_buffer)

            # Non-blocking Transcription via ThreadPool
            try:
                segments, _ = await asyncio.to_thread(
                    model.transcribe,
                    wav_path,
                    beam_size=1,
                    vad_filter=False,
                    language="en"
                )

                text = " ".join(
                    s.text.strip()
                    for s in segments
                    if any(c.isalnum() for c in s.text)
                )

                # Deduplicate: Only send if text changed from last valid frame
                if text and text != last_text:
                    await websocket.send_json({"text": text})
                    last_text = text

            except Exception as e:
                logger.error(f"Transcribe error: {e}")

            # Rolling Buffer: Keep Overlap for Context
            pcm_buffer[:] = pcm_buffer[-KEEP_BYTES:]
            
            if os.path.exists(wav_path):
                os.remove(wav_path)

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        try:
            await websocket.close()
        except: pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
