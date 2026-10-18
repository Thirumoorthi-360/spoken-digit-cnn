
from pathlib import Path
import time

import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

# Record at 16 kHz, then convert to the dataset's 8 kHz.
RECORD_RATE = 16000
MODEL_RATE = 8000
DURATION = 3

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

print("Spoken Digit Voice Test")
print("-----------------------")
print("You will have 3 seconds to say ONE English digit.")
print("For example: zero, one, two, three...")
print("Speak clearly and keep a little distance from the mic.")

input("\nPress ENTER when you are ready...")

print("Get ready...")
time.sleep(1)
print("RECORDING NOW! Speak your digit.")

try:
    recording = sd.rec(
        int(DURATION * RECORD_RATE),
        samplerate=RECORD_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
except Exception as exc:
    raise SystemExit(
        f"Microphone recording failed: {exc}\n"
        "Check Windows microphone permissions and input device."
    )

audio = recording[:, 0]

# Convert 16 kHz to 8 kHz using proper resampling.
audio_8k = resample_poly(audio, up=1, down=2)

# Convert to PCM 16-bit WAV format.
audio_8k = np.clip(audio_8k, -1.0, 1.0)
audio_int16 = (audio_8k * 32767).astype(np.int16)

output_path = OUTPUT / "my_voice.wav"
wavfile.write(output_path, MODEL_RATE, audio_int16)

print("\nRecording saved successfully!")
print("File:", output_path)
print("Sample rate:", MODEL_RATE, "Hz")
print("Duration:", len(audio_8k) / MODEL_RATE, "seconds")
print("\nNext, run:")
print('python src/predict.py "outputs/my_voice.wav"')
