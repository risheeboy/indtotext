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
RECORD_DURATION = 30  # seconds
SAMPLE_RATE = 16000  # Whisper expects 16kHz
TRANSCRIPTION_FILE = "transcription.txt"  # Fixed filename for transcription output

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

# Load the IndicWhisper model and processor directly
print("Loading model, this may take some time...")
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

print(f"Device set to use {device}")
print(f"Model config: {model.config}")

# Enable debug logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

# Record audio
audio_data = record_audio()
print("Audio recorded")

# Save audio to file
audio_file = "recorded_audio.wav"
sf.write(audio_file, audio_data, SAMPLE_RATE)
print(f"Audio saved to {audio_file}")

# Perform transcription
print("Transcribing...")
print(f"Audio shape: {audio_data.shape}")
print(f"Audio duration: {len(audio_data) / SAMPLE_RATE:.2f} seconds")
print(f"Audio min/max values: {audio_data.min():.4f} / {audio_data.max():.4f}")

# Check if audio is too quiet
audio_rms = np.sqrt(np.mean(audio_data**2))
print(f"Audio RMS level: {audio_rms:.6f}")
if audio_rms < 0.01:
    print("WARNING: Audio seems very quiet. Amplifying...")
    # Amplify quiet audio
    amplification_factor = 0.02 / audio_rms if audio_rms > 0 else 1.0
    audio_data = audio_data * min(amplification_factor, 5.0)  # Cap at 5x amplification
    print(f"Amplified by factor: {min(amplification_factor, 5.0):.2f}")
    
    # Check new RMS
    new_rms = np.sqrt(np.mean(audio_data**2))
    print(f"New audio RMS level: {new_rms:.6f}")
    
    # Ensure no clipping
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
        print("Audio normalized to prevent clipping")

# Process the audio using the processor
print("Processing audio features...")
input_features = processor(audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
print(f"Input features shape: {input_features.shape}")

# Generate transcription in Hindi
print("Generating transcription...")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
print(f"Forced decoder IDs: {forced_decoder_ids}")

predicted_ids = model.generate(
    input_features,
    forced_decoder_ids=forced_decoder_ids,
    max_new_tokens=400,  # Reduced to stay within limits
    num_beams=5,
    do_sample=False,
    temperature=0.0,
    use_cache=True,
)

print(f"Predicted IDs shape: {predicted_ids.shape}")
print(f"Predicted IDs: {predicted_ids}")

# Decode the results
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(f"Raw transcription: {transcription}")
hindi_text = transcription[0] if transcription else ""

print("Transcribed text (in Hindi):", hindi_text)

# Save transcription to file
with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
    f.write(hindi_text)
print(f"Transcription saved to {TRANSCRIPTION_FILE}")

# Also try without forced decoder IDs to see what language it detects
print("\n--- Testing without language forcing ---")
predicted_ids_auto = model.generate(
    input_features,
    max_new_tokens=400,  # Reduced to stay within limits
    num_beams=5,
    do_sample=False,
    temperature=0.0,
)

transcription_auto = processor.batch_decode(predicted_ids_auto, skip_special_tokens=True)
print("Auto-detected result:", transcription_auto[0] if transcription_auto else "")

# Save debug info
debug_info = {
    "audio_shape": audio_data.shape,
    "audio_rms": float(audio_rms),
    "input_features_shape": list(input_features.shape),
    "forced_decoder_ids": forced_decoder_ids,
    "predicted_ids": predicted_ids.tolist(),
    "hindi_transcription": hindi_text,
    "auto_transcription": transcription_auto[0] if transcription_auto else ""
}

import json
with open("debug_result.json", "w") as f:
    json.dump(debug_info, f, indent=2)
print("Debug info saved to debug_result.json")