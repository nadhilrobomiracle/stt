import uuid
import os
import shutil
import logging

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr  # Changed library
import uvicorn
import static_ffmpeg

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-api")

# ---------------- FFMPEG ----------------
static_ffmpeg.add_paths()

# ---------------- APP ----------------
app = FastAPI(title="Simple SpeechRecognition STT")

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

# ---------------- INITIALIZE RECOGNIZER ----------------
# SpeechRecognition doesn't "load" a heavy model like Whisper into RAM 
# at startup; it initializes a recognizer instance.
recognizer = sr.Recognizer()
logger.info("SpeechRecognition recognizer initialized.")

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {
        "status": "online",
        "engine": "SpeechRecognition",
        "backend": "Google Web Speech API"
    }

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    # Note: SpeechRecognition works best with .wav files
    ext = file.filename.split(".")[-1] if "." in file.filename else "wav"
    path = os.path.join(TEMP_DIR, f"{file_id}.{ext}")

    try:
        # Save uploaded file
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Transcribe using SpeechRecognition
        with sr.AudioFile(path) as source:
            audio_data = recognizer.record(source)
            
            # Using Google Web Speech API (Free tier, requires internet)
            # For offline use, you'd use recognizer.recognize_sphinx(audio_data)
            text = recognizer.recognize_google(audio_data)

        logger.info(f"Transcribed: {text}")
        return {"text": text}

    except sr.UnknownValueError:
        logger.warning("SpeechRecognition could not understand audio")
        return {"text": "", "error": "Unintelligible speech"}
    except sr.RequestError as e:
        logger.error(f"Could not request results from service; {e}")
        return {"text": "", "error": f"Service error: {e}"}
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
