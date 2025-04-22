import whisper
import tempfile
import os
from pydub import AudioSegment


def extract_audio_text(uploaded_audio, chunk_duration=30):
    model = whisper.load_model("large")  # You can use "tiny", "small", "medium", or "large"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(uploaded_audio.read())
        temp_audio_path = temp_audio.name

    audio = AudioSegment.from_file(temp_audio_path)
    duration_seconds = len(audio) // 1000

    chunks = [audio[start * 1000:(start + chunk_duration) * 1000]
              for start in range(0, duration_seconds, chunk_duration)]

    full_text = []
    for chunk in chunks:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
            chunk.export(chunk_file.name, format="wav")

            # Use Whisper for transcription
            result = model.transcribe(chunk_file.name,task='translate')
            full_text.append(result["text"])

            os.remove(chunk_file.name)

    os.remove(temp_audio_path)
    return " ".join(full_text)
