import uuid
import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from pydub import AudioSegment  # Add this for format handling
import uvicorn
import static_ffmpeg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-api")

static_ffmpeg.add_paths()

app = FastAPI(title="Cloud STT API")
recognizer = sr.Recognizer()

# ... (CORS middleware remains the same)

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    original_path = os.path.join(TEMP_DIR, f"{file_id}_{file.filename}")
    wav_path = os.path.join(TEMP_DIR, f"{file_id}.wav")

    try:
        # 1. Save uploaded file (could be mp3, m4a, wav, etc.)
        with open(original_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. Convert to WAV using pydub/ffmpeg (Crucial for Cloud)
        # This bypasses the need for PyAudio
        audio = AudioSegment.from_file(original_path)
        audio.export(wav_path, format="wav")

        # 3. Transcribe
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        return {"text": text}

    except Exception as e:
        logger.error(f"STT error: {e}")
        return {"text": "", "error": str(e)}

    finally:
        # Cleanup both files
        for p in [original_path, wav_path]:
            if os.path.exists(p):
                os.remove(p)
