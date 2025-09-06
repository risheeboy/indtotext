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

# Perform translation using direct model approach
print("Transcribing and translating...")
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

# Test multiple language approaches
print("Testing different language detection approaches...")

# Get available language tokens
print("Checking supported languages...")
tokenizer = processor.tokenizer
print("Available language codes in tokenizer:", [f"<|{lang}|>" for lang in ["hi", "en", "te", "ta", "ml", "kn", "gu", "pa", "bn", "or", "as", "mr"]])

results = {}

# 1. Auto-detection (no language forcing)
print("\n--- 1. Auto-detection (let model decide language) ---")
predicted_ids_auto = model.generate(
    input_features,
    max_new_tokens=400,
    num_beams=5,
    do_sample=False,
    temperature=0.0,
)
transcription_auto = processor.batch_decode(predicted_ids_auto, skip_special_tokens=True)[0]
results["auto_detected"] = transcription_auto
print("Auto-detected result:", transcription_auto)

# 2. Test common Indian languages + English
languages_to_test = ["hi", "en", "te", "ta", "ml", "kn", "gu", "pa", "bn", "mr"]
print(f"\n--- 2. Testing specific languages: {languages_to_test} ---")

for lang in languages_to_test:
    try:
        print(f"\nTesting {lang}:")
        
        # Transcribe in original language
        forced_decoder_ids_transcribe = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
        predicted_ids_transcribe = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids_transcribe,
            max_new_tokens=400,
            num_beams=5,
            do_sample=False,
            temperature=0.0,
        )
        transcription_original = processor.batch_decode(predicted_ids_transcribe, skip_special_tokens=True)[0]
        
        # Translate to English
        forced_decoder_ids_translate = processor.get_decoder_prompt_ids(language=lang, task="translate")
        predicted_ids_translate = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids_translate,
            max_new_tokens=400,
            num_beams=5,
            do_sample=False,
            temperature=0.0,
        )
        transcription_english = processor.batch_decode(predicted_ids_translate, skip_special_tokens=True)[0]
        
        results[f"{lang}_original"] = transcription_original
        results[f"{lang}_english"] = transcription_english
        
        print(f"  Original ({lang}): {transcription_original}")
        
    except Exception as e:
        print(f"  Error with {lang}: {e}")
        results[f"{lang}_error"] = str(e)

# 3. Mixed language approach (try without language specification)
print("\n--- 3. Mixed language approach ---")
try:
    # Generate without language specification but with transcription task
    predicted_ids_mixed = model.generate(
        input_features,
        max_new_tokens=400,
        num_beams=5,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
    )
    transcription_mixed = processor.batch_decode(predicted_ids_mixed, skip_special_tokens=True)[0]
    results["mixed_language"] = transcription_mixed
    print("Mixed language result:", transcription_mixed)
except Exception as e:
    print(f"Mixed language error: {e}")
    results["mixed_language_error"] = str(e)

# Find the best result (longest non-empty transcription)
best_result = ""
best_lang = "unknown"
best_english = ""

for key, value in results.items():
    if "_english" not in key and "_error" not in key and len(value.strip()) > len(best_result.strip()):
        best_result = value
        best_lang = key
        # Try to get corresponding English translation
        english_key = f"{key.replace('_original', '')}_english"
        if english_key in results:
            best_english = results[english_key]

print(f"\n--- BEST RESULT ---")
print(f"Language: {best_lang}")
print(f"Original: {best_result}")

# Store final results
original_text = best_result
translated_text = best_english if best_english else best_result

# Save transcriptions to file
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Save to text file
with open(f"transcription_{timestamp}.txt", "w", encoding="utf-8") as f:
    f.write(f"Multi-Language Transcription Results - {datetime.datetime.now()}\n")
    f.write("="*60 + "\n\n")
    f.write(f"BEST RESULT:\n")
    f.write(f"Detected Language: {best_lang}\n")
    f.write(f"Original Text: {original_text}\n")
    f.write("="*60 + "\n")
    f.write("ALL RESULTS:\n")
    f.write("="*60 + "\n")
    
    for key, value in results.items():
        if "_error" not in key:
            f.write(f"{key}: {value}\n")
    
    f.write("\nERRORS:\n")
    for key, value in results.items():
        if "_error" in key:
            f.write(f"{key}: {value}\n")

print(f"\n--- FINAL RESULTS ---")
print(f"DETECTED LANGUAGE: {best_lang}")
print(f"ORIGINAL TEXT: {original_text}")
print(f"Results saved to transcription_{timestamp}.txt")

# Save debug info
debug_info = {
    "timestamp": timestamp,
    "audio_shape": audio_data.shape,
    "audio_rms": float(audio_rms),
    "input_features_shape": list(input_features.shape),
    "best_language": best_lang,
    "original_text": original_text,
    "english_translation": translated_text,
    "all_results": results
}

import json
with open(f"debug_result_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump(debug_info, f, indent=2, ensure_ascii=False)
print(f"Debug info saved to debug_result_{timestamp}.json")