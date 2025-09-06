import os
import sys
import subprocess
import platform
import sounddevice as sd
import soundfile as sf
import numpy as np
from transformers import pipeline
import torch

# Check if FFmpeg is installed
def is_ffmpeg_installed():
    try:
        # Run 'ffmpeg -version' and capture output
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if result.returncode == 0:
            return True
        else:
            print("FFmpeg is not installed.")
            return False
    except FileNotFoundError:
        print("FFmpeg is not installed (command not found).")
        return False
    except subprocess.CalledProcessError:
        print("FFmpeg check failed (non-zero exit code).")
        return False

# Get OS-specific installation instructions
def get_ffmpeg_install_instructions():
    os_name = platform.system()
    if os_name == 'Darwin':  # macOS
        return "On macOS, run: brew install ffmpeg"
    elif os_name == 'Linux':
        return "On Linux: For Ubuntu/Debian, run: sudo apt update && sudo apt install ffmpeg\nFor other distributions, use your package manager (e.g., sudo yum install ffmpeg on CentOS/RHEL, sudo dnf install ffmpeg on Fedora, sudo pacman -S ffmpeg on Arch)."
    elif os_name == 'Windows':
        return "On Windows: Download the latest build from https://ffmpeg.org/download.html (select Windows builds from gyan.dev or BtbN), extract it, and add the bin folder to your system's PATH environment variable.\nAlternatively, if you have Chocolatey installed[](https://chocolatey.org/install), run: choco install ffmpeg"
    else:
        return f"On your OS ({os_name}): Please visit https://ffmpeg.org/download.html for installation instructions."

# Exit if FFmpeg is not installed, with OS-specific instructions
if not is_ffmpeg_installed():
    instructions = get_ffmpeg_install_instructions()
    print(f"Please install FFmpeg before running this script:\n{instructions}")
    sys.exit(1)

# Enable MPS fallback to CPU for unsupported ops (useful on macOS Apple Silicon)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Configuration
MODEL_PATH = "hindi_models/whisper-medium-hi_alldata_multigpu"  # download from https://indicwhisper.objectstore.e2enetworks.net/hindi_models.zip
RECORD_DURATION = 10  # seconds
SAMPLE_RATE = 16000  # Whisper expects 16kHz

def record_audio(duration=RECORD_DURATION, fs=SAMPLE_RATE):
    print(f"### Recording Started. Duration: {duration} seconds")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished")
    return np.squeeze(audio)  # Convert to 1D array

# Determine device: MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS (Apple Silicon GPU) for acceleration.")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA (NVIDIA GPU).")
else:
    device = "cpu"
    print("Using CPU (this may be slow on large models).")

# Load the IndicWhisper model pipeline
print("Loading model, this may take some time...")
whisper_asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH,
    device=device,
    return_timestamps=True,  # Enable timestamps for debugging
    chunk_length_s=30,      # Process in chunks
)

print(f"Device set to use {device}")

# Set for translation to English
whisper_asr.model.config.forced_decoder_ids = (
    whisper_asr.tokenizer.get_decoder_prompt_ids(task="translate")
)

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("transformers")
logger.setLevel(logging.DEBUG)

# Record audio
audio_data = record_audio()
print("Audio recorded")

# Save audio to file
audio_file = "recorded_audio.wav"
sf.write(audio_file, audio_data, SAMPLE_RATE)
print(f"Audio saved to {audio_file}")

# Perform translation with debug info
print("Transcribing and translating...")
print(f"Audio shape: {audio_data.shape}")
print(f"Audio duration: {len(audio_data) / SAMPLE_RATE:.2f} seconds")
print(f"Audio min/max values: {audio_data.min():.4f} / {audio_data.max():.4f}")

# Check if audio is too quiet
audio_rms = np.sqrt(np.mean(audio_data**2))
print(f"Audio RMS level: {audio_rms:.6f}")
if audio_rms < 0.01:
    print("WARNING: Audio seems very quiet. Try speaking louder.")

# Process with additional parameters for debugging
result = whisper_asr(
    audio_data,
    generate_kwargs={
        "task": "translate",
        "language": "hi",  # Explicitly set Hindi
        "num_beams": 5,
        "do_sample": False,
    }
)

print("Full result:", result)
print("Translated text (to English):", result["text"])

# Show timestamps if available
if "chunks" in result:
    print("Detected chunks:")
    for i, chunk in enumerate(result["chunks"]):
        print(f"  Chunk {i+1}: {chunk['timestamp']} - '{chunk['text']}'")
elif "timestamp" in result:
    print(f"Timestamp: {result['timestamp']}")

# Save processed results for analysis
import json
with open("debug_result.json", "w") as f:
    json.dump(result, f, indent=2)
print("Debug info saved to debug_result.json")