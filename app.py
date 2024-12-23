from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
from typing import Dict, List, Optional
import whisper
import moviepy.editor as mp
from model import MultilingualTextCleaningModel

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize models
try:
    text_cleaner = MultilingualTextCleaningModel(languages=['en', 'es', 'fr', 'de', 'mr', 'hi', 'ja', 'zh'])
    whisper_model = whisper.load_model("base")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    whisper_model = None
    text_cleaner = None

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def translate_large_text(text, target_language="en", chunk_size=500):
    """
    Splits the text into smaller chunks and translates each chunk.
    """
    translator = Translator()
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translated_text = ""
    
    for chunk in chunks:
        try:
            translated = translator.translate(chunk, dest=target_language)
            translated_text += translated.text + " "
        except Exception as e:
            print(f"Error translating chunk: {chunk}\n{e}")
    
    return translated_text.strip()

def detect_and_transcribe(audio_path):
    """Enhanced transcription using both Whisper and Google Speech Recognition"""
    try:
        # First transcription with Whisper
        print("Transcribing with Whisper...")
        result = whisper_model.transcribe(audio_path)
        initial_text = result["text"].strip()
        detected_language = result.get("language", "en")
        
        print(f"Whisper Transcription: {initial_text}")
        print(f"Detected Language: {detected_language}")

        # Skip Google Speech Recognition for certain languages
        if detected_language in ['en', 'de', 'fr']:
            print("Skipping Google Speech Recognition due to detected language.")
            return initial_text, detected_language

        # Second transcription with Google Speech Recognition
        print("Refining with Google Speech Recognition...")
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
            google_transcription = recognizer.recognize_google(audio_data, language=detected_language)
            print(f"Google Speech Recognition: {google_transcription}")
            
            # Use Google transcription if available, otherwise use Whisper transcription
            final_text = google_transcription if google_transcription else initial_text
            
        except (sr.UnknownValueError, sr.RequestError) as e:
            print(f"Google Speech Recognition failed: {str(e)}")
            final_text = initial_text
            
        return final_text, detected_language
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return "Unable to detect speech in the video.", "en"

def get_language_accent_code(language_code):
    """Returns the language and accent code for gTTS"""
    # Dictionary to map language codes to their respective accent codes
    accents = {
        "en": "en",  # General American English
        "en-uk": "en-uk",  # British English
        "es": "es",  # Spanish
        "fr": "fr",  # French
        "de": "de",  # German
        "hi": "hi",  # Hindi
        "ja": "ja",  # Japanese
        "zh": "zh",  # Chinese
        # Add other languages/accent codes here as needed
    }
    
    # Default to the language code if no specific accent is set
    return accents.get(language_code, language_code)

@app.route('/')
def home():
    """Home route that renders the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    target_language = request.form.get('language', 'en')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed types are: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        # Generate unique ID and save file
        unique_id = str(uuid.uuid4())
        filename = secure_filename(f"{unique_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract audio using moviepy
        print("Extracting audio from video...")
        video = mp.VideoFileClip(filepath)
        audio = video.audio
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_audio.wav")
        audio.write_audiofile(audio_path)

        # Detect language and transcribe using enhanced method
        detected_text, detected_language = detect_and_transcribe(audio_path)
        
        # Save raw transcription to file
        transcription_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_transcription.txt")
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(detected_text)

        # Translate text if needed
        if detected_language != target_language:
            print(f"Translating from {detected_language} to {target_language}")
            final_text = translate_large_text(detected_text, target_language=target_language)
        else:
            final_text = detected_text

        # Clean the final text after translation
        if text_cleaner:
            print(f"Cleaning text in {target_language}")
            final_text = text_cleaner.process_transcription(final_text, target_language)
            print(f"Cleaned text: {final_text}")

        # Get the accent for the target language
        accent_language_code = get_language_accent_code(target_language)
        
        # Convert cleaned text to speech
        print("Converting to speech with accent...")
        speech = gTTS(text=final_text, lang=accent_language_code)
        new_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_new_audio.mp3")
        speech.save(new_audio_path)

        # Add new audio to video
        print("Adding new audio to video...")
        new_audio = mp.AudioFileClip(new_audio_path)
        
        # Adjust audio duration to match video
        video_duration = video.duration
        if new_audio.duration > video_duration:
            new_audio = new_audio.subclip(0, video_duration)
        elif new_audio.duration < video_duration:
            new_audio = mp.CompositeAudioClip([new_audio] * int(video_duration/new_audio.duration + 1)).subclip(0, video_duration)

        # Create final video
        video_with_audio = video.set_audio(new_audio)
        final_video_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{unique_id}_output.mp4")
        video_with_audio.write_videofile(final_video_path, codec="libx264")

        # Cleanup
        video.close()
        audio.close()
        new_audio.close()
        video_with_audio.close()

        # Remove temporary files
        for temp_file in [audio_path, new_audio_path, filepath, transcription_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return jsonify({
            'success': True,
            'video_id': unique_id,
            'detected_language': detected_language,
            'message': 'Video processed successfully'
        })

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<video_id>')
def download_video(video_id):
    try:
        for filename in os.listdir(app.config['PROCESSED_FOLDER']):
            if filename.startswith(video_id) and filename.endswith('_output.mp4'):
                return send_file(
                    os.path.join(app.config['PROCESSED_FOLDER'], filename),
                    as_attachment=True,
                    download_name='processed_video.mp4'
                )
        return jsonify({'error': 'Video not found'}), 404
    except Exception as e:
        print(f"Error during download: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
