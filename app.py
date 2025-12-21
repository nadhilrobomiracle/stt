import io
import os
import uuid
import asyncio
from typing import Union
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import pydub
import numpy as np

app = FastAPI(title="Universal STT Service")

# Enable CORS for Web/Mobile compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper Model (Base is a good balance for Raspberry Pi/CPU)
# Use "cuda" for device if you have a GPU
model = WhisperModel("base", device="cpu", compute_type="int8")

def process_audio(audio_bytes: bytes):
    """Normalizes audio to 16kHz mono wav for the STT engine."""
    audio = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # Convert to numpy array
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, audio.duration_seconds

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        content = await file.read()
        audio_data, duration = process_audio(content)
        
        segments, info = model.transcribe(audio_data, beam_size=5)
        full_text = " ".join([segment.text for segment in segments]).strip()

        return {
            "text": full_text,
            "language": info.language,
            "confidence": round(info.language_probability, 2),
            "duration": round(duration, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive audio chunks (binary) from frontend
            data = await websocket.receive_bytes()
            
            # Note: In a production real-time scenario, you would use a 
            # Voice Activity Detection (VAD) buffer here.
            audio_data, _ = process_audio(data)
            segments, _ = model.transcribe(audio_data, beam_size=1)
            
            for segment in segments:
                await websocket.send_json({
                    "event": "transcript_partial",
                    "text": segment.text
                })
    except WebSocketDisconnect:
        print("Client disconnected")
