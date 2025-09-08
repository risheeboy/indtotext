# indi1

Python script to capture audio from the device microphone.
Use the IndicWhisper model (Hindi example) to transcribe.

## Prerequisites:

* Download https://indicwhisper.objectstore.e2enetworks.net/hindi_models.zip and unzip it in project directory
   The model path will be 'hindi_models/whisper-medium-hi_alldata_multigpu'
* Install ffmpeg
* Install requirements:

```bash
# Transcription Setup
git clone https://github.com/risheeboy/indtotext.git
cd indtotext
pip -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Translation Setup
cd ..
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit
pip install -e .
cd ../indtotext
```

Recommended for better performance on GPUs (requires nvcc):
```bash
pip install flash-attn --no-build-isolation
```

## Usage:

Run the script: 

```bash
source venv/bin/activate
python indi1.py
```

It will record audio for 10 seconds (adjust duration as needed) and print the English translation.
