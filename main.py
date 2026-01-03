import uuid
import os
import shutil
import logging

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from pydub import AudioSegment
import uvicorn
import static_ffmpeg

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-api")

# ---------------- FFMPEG ----------------
static_ffmpeg.add_paths()

# ---------------- APP ----------------
app = FastAPI(title="Simple STT API (SpeechRecognition)")

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

# ---------------- RECOGNIZER ----------------
recognizer = sr.Recognizer()

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {
        "status": "online",
        "engine": "speech_recognition",
        "provider": "google"
    }

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    original_path = os.path.join(TEMP_DIR, f"{file_id}_{file.filename}")
    wav_path = os.path.join(TEMP_DIR, f"{file_id}.wav")

    try:
        # 1️⃣ Save uploaded file
        with open(original_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2️⃣ Convert to WAV (16kHz mono)
        audio = AudioSegment.from_file(original_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav")

        # 3️⃣ Speech Recognition
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language="en-IN")

        logger.info(f"Transcribed: {text}")
        return {"text": text}

    except sr.UnknownValueError:
        return {"text": "", "error": "Speech not understood"}

    except sr.RequestError as e:
        return {"text": "", "error": f"Google API error: {e}"}

    except Exception as e:
        logger.error(f"STT error: {e}")
        return {"text": "", "error": str(e)}

    finally:
        for p in (original_path, wav_path):
            if os.path.exists(p):
                os.remove(p)

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
