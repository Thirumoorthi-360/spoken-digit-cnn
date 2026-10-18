
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from data import load_audio
from model import SpokenDigitCNN

CHECKPOINT = Path("outputs/best_model.pt")

if len(sys.argv) != 2:
    raise SystemExit(
        'Usage: python src/predict.py "path/to/audio.wav"'
    )

audio_path = Path(sys.argv[1])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SpokenDigitCNN().to(device)
model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
)
model.eval()

audio = load_audio(audio_path)
audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
audio = audio.to(device)

with torch.no_grad():
    logits = model(audio)
    probabilities = F.softmax(logits, dim=1)[0]

predicted_digit = int(probabilities.argmax().item())
confidence = float(probabilities[predicted_digit].item())

print(f"Audio file: {audio_path.name}")
print(f"Predicted digit: {predicted_digit}")
print(f"Model confidence: {confidence:.2%}")

print("\nClass probabilities:")
for digit, probability in enumerate(probabilities.cpu().tolist()):
    print(f"{digit}: {probability:.2%}")
