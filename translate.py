import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Configuration
TRANSCRIPTION_FILE = "transcription.txt"  # Fixed filename to read transcription from
TRANSLATION_FILE = "translation.txt"     # Fixed filename to save translation to

# Enable MPS fallback to CPU for unsupported ops (useful on macOS Apple Silicon)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def main():
    # Check if transcription file exists
    if not os.path.exists(TRANSCRIPTION_FILE):
        print(f"Error: {TRANSCRIPTION_FILE} not found. Please run transcribe.py first.")
        return
    
    # Read Hindi transcription from file
    with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
        hindi_text = f.read().strip()
    
    if not hindi_text:
        print("Error: No transcription text found in the file.")
        return
    
    print(f"Read Hindi text: {hindi_text}")
    
    # Try using pipeline instead for better compatibility
    print("Loading translation pipeline...")
    
    try:
        # Use pipeline for translation - this is more robust
        translator = pipeline(
            "translation", 
            model="Helsinki-NLP/opus-mt-hi-en",
            device=-1  # Force CPU
        )
        
        print("Translating Hindi to English...")
        result = translator(hindi_text, max_length=256)
        translated_text = result[0]['translation_text']
        
        print("Translated text (to English):", translated_text)
        
        # Save translation to file
        with open(TRANSLATION_FILE, "w", encoding="utf-8") as f:
            f.write(translated_text)
        print(f"Translation saved to {TRANSLATION_FILE}")
        
    except Exception as e:
        print(f"Translation with pipeline failed: {e}")
        print("Trying alternative approach...")
        
        # Fallback to the original approach with better error handling
        try:
            from IndicTransToolkit.processor import IndicProcessor
            
            ip = IndicProcessor(inference=True)
            tokenizer_trans = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True)
            
            translation_device = "cpu"
            model_trans = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/indictrans2-indic-en-dist-200M", 
                trust_remote_code=True
            ).to(translation_device)
            
            model_trans.eval()
            
            src_lang = "hin_Deva"
            tgt_lang = "eng_Latn"
            
            batch = ip.preprocess_batch([hindi_text], src_lang=src_lang, tgt_lang=tgt_lang)
            inputs = tokenizer_trans(batch, padding="longest", truncation=True, return_tensors="pt").to(translation_device)
            
            with torch.no_grad():
                outputs = model_trans.generate(**inputs, max_length=256)
            
            translated = tokenizer_trans.batch_decode(outputs, skip_special_tokens=True)
            translated_text = ip.postprocess_batch(translated, lang=tgt_lang)[0]
            
            print("Translated text (to English):", translated_text)
            
            with open(TRANSLATION_FILE, "w", encoding="utf-8") as f:
                f.write(translated_text)
            print(f"Translation saved to {TRANSLATION_FILE}")
            
        except Exception as e2:
            print(f"Fallback translation also failed: {e2}")
            print("Consider installing missing dependencies or using a different model.")

if __name__ == "__main__":
    main()