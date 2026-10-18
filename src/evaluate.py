
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

from data import SpokenDigitDataset
from model import SpokenDigitCNN

ROOT = Path("data-source/recordings")
OUTPUT = Path("outputs")
CHECKPOINT = OUTPUT / "best_model.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

test_ds = SpokenDigitDataset(ROOT, "test")
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
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

all_labels = []
all_predictions = []

with torch.no_grad():
    for audio, labels in test_loader:
        audio = audio.to(device)

        logits = model(audio)
        predictions = logits.argmax(dim=1).cpu()

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

accuracy = accuracy_score(
    all_labels, all_predictions
)

print(f"Test accuracy: {accuracy:.4f}")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        target_names=[str(i) for i in range(10)],
        digits=3,
        zero_division=0,
    )
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(10)),
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(range(10)),
)

display.plot(cmap="Blues", values_format="d")
plt.title("FSDD Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    OUTPUT / "confusion_matrix.png",
    dpi=150,
)
plt.show()

# Save the exact predictions for later error analysis.
np.savez(
    OUTPUT / "test_predictions.npz",
    labels=np.asarray(all_labels),
    predictions=np.asarray(all_predictions),
)

print("Saved confusion matrix and test predictions.")

