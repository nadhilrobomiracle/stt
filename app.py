import os
import shutil
import uuid
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import uvicorn

# Initialize FastAPI App
app = FastAPI(title="High-Performance Live STT API")

# Configure CORS (Allow all origins for direct access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp directory exists
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "online", "engine": "Google Web Speech API"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text.
    Accepts: webm, wav, mp3, m4a
    Returns: { "text": "transcribed text" }
    """
    
    # Generate unique filenames
    file_id = str(uuid.uuid4())
    # Default to webm if no extension provided (common for Blob uploads)
    input_ext = file.filename.split(".")[-1] if "." in file.filename else "webm"
    temp_input_path = os.path.join(TEMP_DIR, f"{file_id}.{input_ext}")
    wav_output_path = os.path.join(TEMP_DIR, f"{file_id}.wav")

    try:
        # Save uploaded file to disk
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert audio to 16kHz Mono WAV (Standard for Speech Recognition)
        # This normalization greatly improves accuracy and speed
        try:
            audio = AudioSegment.from_file(temp_input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_output_path, format="wav")
        except Exception as e:
            # Likely missing ffmpeg or corrupt audio
            print(f"Audio Conversion Error: {e}")
            raise HTTPException(status_code=400, detail="Invalid audio format or codec missing")

        # Perform Speech Recognition
        recognizer = sr.Recognizer()
        
        # Load the processed WAV file
        with sr.AudioFile(wav_output_path) as source:
            audio_data = recognizer.record(source)

        try:
            # Use Google Web Speech API (Fast, reliable, free tier)
            text = recognizer.recognize_google(audio_data)
            return {"text": text}
            
        except sr.UnknownValueError:
            # Audio was silent or unintelligible
            return {"text": ""}
            
        except sr.RequestError:
            # Connection to Google API failed
            raise HTTPException(status_code=503, detail="Speech service unavailable")

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
        
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(wav_output_path):
            os.remove(wav_output_path)

if __name__ == "__main__":
    # Ready to run using: python main.py
    port = int(os.environ.get("PORT", 5000))
    # Use uvicorn programmatically
    uvicorn.run(app, host="0.0.0.0", port=port)
