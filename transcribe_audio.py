# transcribe_audio.py

import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os
import assemblyai as aai
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# --- Configuration ---
SAMPLE_RATE = 44100  # Standard sample rate for audio

recorded_chunks = [] # List to store audio chunks

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, flush=True)
    recorded_chunks.append(indata.copy())

def record_audio_manual(sample_rate: int) -> np.ndarray:
    """Records audio manually until Enter is pressed."""
    global recorded_chunks
    recorded_chunks = [] # Reset chunks list

    print("Starting recording... Press Enter to stop.")
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=audio_callback):
            input() # Wait for user to press Enter
        print("Recording finished.")

        if not recorded_chunks:
            print("Warning: No audio data recorded.")
            return np.array([], dtype='int16')

        # Concatenate all recorded chunks
        recording = np.concatenate(recorded_chunks, axis=0)
        return recording

    except Exception as e:
        print(f"Error during recording: {e}")
        return np.array([], dtype='int16') # Return empty array on error

def save_temp_wav(recording: np.ndarray, sample_rate: int) -> str:
    """Saves the recorded audio to a temporary WAV file."""
    # Check if recording is empty
    if recording.size == 0:
        print("Error: Cannot save empty recording.")
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav.write(temp_file.name, sample_rate, recording)
            # print(f"Audio saved temporarily to: {temp_file.name}")
            return temp_file.name
    except Exception as e:
        print(f"Error saving temporary WAV file: {e}")
        return None

def transcribe_with_assemblyai(file_path: str, api_key: str) -> str:
    """Transcribes the audio file using AssemblyAI."""
    if not api_key or api_key == "YOUR_ASSEMBLYAI_API_KEY":
        print("Error: AssemblyAI API key not provided or is a placeholder.")
        print("Please provide your key via the --api-key argument or set the ASSEMBLYAI_API_KEY environment variable.")
        return None

    # print("Configuring AssemblyAI...")
    aai.settings.api_key = api_key

    try:
        # print(f"Uploading {file_path} to AssemblyAI for transcription...")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"Transcription failed: {transcript.error}")
            return None
        elif transcript.status == aai.TranscriptStatus.completed:
            # print("Transcription successful.")
            return transcript.text
        else:
            # Handle other statuses like queued, processing if needed
            print(f"Transcription status: {transcript.status}")
            # Basic polling loop (consider websockets for production)
            while transcript.status not in {aai.TranscriptStatus.completed, aai.TranscriptStatus.error}:
                time.sleep(5)
                transcript = transcript.get()
                print(f"Polling... Current status: {transcript.status}")
                if transcript.status == aai.TranscriptStatus.completed:
                    print("Transcription successful.")
                    return transcript.text
                elif transcript.status == aai.TranscriptStatus.error:
                    print(f"Transcription failed during polling: {transcript.error}")
                    return None
            return None # Should not reach here ideally

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None

def save_transcript_to_file(text: str, filename: str):
    """Saves the provided text to a file."""
    try:
        with open(filename, "w") as f:
            f.write(text)
        print(f"Transcript saved to {filename}")
    except Exception as e:
        print(f"Error saving transcript to file: {e}")

def run_stt():
    temp_file_path = None
    transcript_text = None
    try:
        # 1. Record Audio Manually
        audio_data = record_audio_manual(SAMPLE_RATE)

        # 2. Save Temporarily
        temp_file_path = save_temp_wav(audio_data, SAMPLE_RATE)
        if not temp_file_path:
            raise Exception("Failed to save temporary audio file.")

        # 3. Transcribe
        transcript_text = transcribe_with_assemblyai(temp_file_path, ASSEMBLYAI_API_KEY)

        # 4. Print Result
        if transcript_text:
            print("\n--- Transcription ---")
            print(transcript_text)
        else:
            print("\nNo transcription result received.")

    except Exception as e:
        print(f"An overall error occurred: {e}")
    finally:
        # 5. Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            # print(f"Cleaning up temporary file: {temp_file_path}")
            os.remove(temp_file_path)
        if transcript_text:
            return transcript_text