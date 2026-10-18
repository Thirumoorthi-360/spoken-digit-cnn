
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import (
    SAMPLE_RATE,
    SpokenDigitDataset,
    load_audio,
)

ROOT = Path("data-source/recordings")
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

dataset = SpokenDigitDataset(ROOT, "train")

print("Training samples:", len(dataset))
print("Waveform shape:", dataset[0][0].shape)
print("Label:", dataset[0][1].item())

# Display one real waveform.
path, label = dataset.samples[0]
audio = load_audio(path)
time = np.arange(len(audio)) / SAMPLE_RATE

plt.figure(figsize=(12, 4))
plt.plot(time, audio)
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized amplitude")
plt.title(f"Digit {label}: {path.name}")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT / "waveform.png", dpi=150)
plt.show()

# Spectrogram: an important DSP visualization.
plt.figure(figsize=(12, 4))
plt.specgram(
    audio,
    NFFT=256,
    Fs=SAMPLE_RATE,
    noverlap=192,
    cmap="viridis",
)
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.title(f"Spectrogram: digit {label}")
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.savefig(OUTPUT / "spectrogram.png", dpi=150)
plt.show()
