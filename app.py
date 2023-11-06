from flask import Flask, render_template_string, request, jsonify
import openai
import base64
import os
from tempfile import NamedTemporaryFile

app = Flask(__name__)
# Ensure the OPENAI_API_KEY is set in your environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
def index():
    # Serve the main HTML page
    return render_template_string(open("index.html").read())


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    # Entry point for audio transcription and response synthesis
    audio_file = request.files["audio_data"]
    transcript = transcribe_audio_file(audio_file)
    audio_base64 = synthesize_speech(transcript)
    return jsonify({"audio_base64": audio_base64})


def transcribe_audio_file(audio_file):
    """
    This function handles the transcription of the audio file.
    It saves the file temporarily, sends it to OpenAI for transcription,
    and then cleans up the temporary file.
    """
    with NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio_file:
        audio_file.save(temp_audio_file)
        temp_audio_file_path = temp_audio_file.name

    try:
        with open(temp_audio_file_path, "rb") as f:
            transcript_response = openai.audio.transcriptions.create(
                model="whisper-1", file=f
            )
        # Extract the text from the transcription response
        transcript = transcript_response.text
    finally:
        # Ensure the temporary file is deleted
        os.unlink(temp_audio_file_path)

    return transcript


def synthesize_speech(text):
    """
    This function takes the transcript text and synthesizes speech
    using OpenAI's text-to-speech API. The audio content is then
    encoded in base64 and returned.
    """
    speech_response = openai.audio.speech.create(
        model="tts-1", voice="nova", input=text
    )

    # Encode the binary audio content in base64 to send to the client
    audio_content = speech_response.content
    audio_base64 = base64.b64encode(audio_content).decode("utf-8")

    return audio_base64


if __name__ == "__main__":
    app.run(debug=True)
