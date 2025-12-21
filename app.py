import os
import logging
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
# Try to load from local .env context if available (development)
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
# Enable CORS for all domains, allowing the API to be called from any frontend
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set. STT endpoints will fail.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using Gemini 1.5 Flash for speed and multimodal capabilities
        model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini 1.5 Flash model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")

def clean_mime_type(mime_type):
    """
    Cleans the mime type to ensure it's compatible with Gemini.
    Browsers often send 'audio/webm;codecs=opus', but we just want 'audio/webm'.
    """
    if not mime_type:
        return "audio/mp3" # Fallback
    return mime_type.split(';')[0].strip()

@app.route('/', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "online",
        "service": "Nexus STT Backend",
        "model": "gemini-1.5-flash"
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Endpoint to convert voice/audio to text.
    Expects a multipart/form-data POST with a 'file' field containing the audio.
    """
    start_time = time.time()
    
    if not GEMINI_API_KEY:
        return jsonify({"error": "Server configuration error: API Key missing"}), 500

    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({"error": "No file uploaded. Please send a file with key 'file'."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read file content into memory
        audio_data = file.read()
        
        # Determine strict mime type
        user_mime = file.mimetype or "audio/webm"
        clean_mime = clean_mime_type(user_mime)
        
        logger.info(f"Processing audio: {len(audio_data)} bytes, MIME: {clean_mime}")
        
        # Prompt for the model
        prompt = (
            "Transcribe the spoken audio in this file into text. "
            "Output ONLY the transcription. "
            "If the audio is silent or unintelligible, output nothing."
        )
        
        # Call Gemini API with inline data
        response = model.generate_content([
            prompt,
            {
                "mime_type": clean_mime,
                "data": audio_data
            }
        ])
        
        transcribed_text = response.text.strip()
        processing_time = time.time() - start_time
        
        logger.info(f"Transcription complete in {processing_time:.2f}s: '{transcribed_text}'")
        
        return jsonify({
            "text": transcribed_text,
            "processing_time_seconds": round(processing_time, 3)
        })

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        # Improve error messages for common issues
        error_msg = str(e)
        if "400" in error_msg:
             return jsonify({"error": "Invalid audio format or corrupted file."}), 400
        return jsonify({"error": "Failed to process audio", "details": error_msg}), 500

if __name__ == '__main__':
    # Run slightly differently in dev
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
