
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset

SAMPLE_RATE = 8000
TARGET_LENGTH = 16000
NUM_CLASSES = 10


def load_audio(path: Path) -> np.ndarray:
    """Load, center and normalize one FSDD waveform."""
    sample_rate, audio = wavfile.read(path)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{path.name}: expected {SAMPLE_RATE} Hz, "
            f"got {sample_rate} Hz"
        )

    # Convert integer PCM or floating-point audio to float32.
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), abs(info.max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)

    # FSDD recordings should be mono.
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio: {path.name}")

    # Remove DC offset.
    audio = audio - np.mean(audio)

    # Peak normalization.
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio = audio / peak

    # Fixed length: center-crop long signals, zero-pad short ones.
    if len(audio) > TARGET_LENGTH:
        start = (len(audio) - TARGET_LENGTH) // 2
        audio = audio[start:start + TARGET_LENGTH]
    elif len(audio) < TARGET_LENGTH:
        missing = TARGET_LENGTH - len(audio)
        left = missing // 2
        right = missing - left
        audio = np.pad(audio, (left, right))

    return audio.astype(np.float32)


def parse_filename(path: Path):
    # Expected format: digit_speaker_index.wav
    parts = path.stem.split("_")

    if len(parts) != 3:
        raise ValueError(f"Unexpected filename: {path.name}")

    digit = int(parts[0])
    speaker = parts[1]
    index = int(parts[2])

    return digit, speaker, index


def get_split(path: Path) -> str:
    _, _, index = parse_filename(path)

    if 0 <= index <= 4:
        return "test"
    elif 5 <= index <= 9:
        return "validation"
    elif 10 <= index <= 49:
        return "train"

    raise ValueError(f"Unexpected recording index: {path.name}")


class SpokenDigitDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")

        files = sorted(self.root.glob("*.wav"))
        if not files:
            raise FileNotFoundError(
                f"No WAV files found in {self.root}"
            )

        self.samples = [
            (path, parse_filename(path)[0])
            for path in files
            if get_split(path) == split
        ]

        if not self.samples:
            raise RuntimeError(f"No samples found for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        audio = load_audio(path)

        # Conv1d expects (channels, time) per sample.
        audio = torch.from_numpy(audio).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return audio, label
