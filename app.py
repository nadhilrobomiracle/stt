import os
import logging
import uuid
import speech_recognition as sr
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
# Enable CORS so any frontend can call this API
CORS(app, resources={r"/*": {"origins": "*"}})

# Directory for temporary audio storage
TEMP_DIR = "/tmp" if os.path.exists("/tmp") else "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/', methods=['GET'])
def health_check():
    """Simple health check."""
    return jsonify({
        "status": "online", 
        "service": "Nexus STT (Google Speech API)",
        "info": "No AI/LLM used. Pure Voice-to-Text."
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Endpoint to convert voice to text using SpeechRecognition (Google Web Speech API).
    Input: Multipart form-data with 'file'.
    Output: JSON { "text": "transcribed text" }
    """
    temp_filename = None
    wav_filename = None
    
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Generate unique filenames
        unique_id = str(uuid.uuid4())
        # Try to guess extension or default to .webm (common from browsers)
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'webm'
        
        temp_filename = os.path.join(TEMP_DIR, f"{unique_id}.{ext}")
        wav_filename = os.path.join(TEMP_DIR, f"{unique_id}.wav")
        
        # Save uploaded file
        file.save(temp_filename)
        
        # Convert to WAV (Required for SpeechRecognition)
        try:
            # Pydub handles various formats (webm, mp3, etc) given ffmpeg is installed
            sound = AudioSegment.from_file(temp_filename)
            sound.export(wav_filename, format="wav")
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return jsonify({
                "error": "Failed to process audio format.",
                "details": "Ensure ffmpeg is installed on the server."
            }), 500
            
        # Perform Speech Recognition
        recognizer = sr.Recognizer()
        
        try:
            with sr.AudioFile(wav_filename) as source:
                # Read the audio file
                audio_data = recognizer.record(source)
                
                # Use Google Web Speech API (Free, no key required for low volume)
                text = recognizer.recognize_google(audio_data)
                
                logger.info(f"Transcription successful: '{text}'")
                return jsonify({"text": text})
                
        except sr.UnknownValueError:
            # Audio was not understood
            logger.info("Audio not understood (silence or unclear)")
            return jsonify({"text": ""})
        except sr.RequestError as e:
            # API was unreachable
            logger.error(f"Google Speech API error: {e}")
            return jsonify({"error": "Speech recognition service unavailable"}), 503

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500
        
    finally:
        # Cleanup temporary files
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass
        if wav_filename and os.path.exists(wav_filename):
            try:
                os.remove(wav_filename)
            except:
                pass

if __name__ == '__main__':
    # Run locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
