
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import SpokenDigitDataset
from model import SpokenDigitCNN

SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001

ROOT = Path("data-source/recordings")
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Training device:", device)

train_ds = SpokenDigitDataset(ROOT, "train")
val_ds = SpokenDigitDataset(ROOT, "validation")

generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
    num_workers=0,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Training examples:", len(train_ds))
print("Validation examples:", len(val_ds))

model = SpokenDigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

best_val_accuracy = 0.0
best_val_loss = float("inf")
patience = 7
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
}

for epoch in range(EPOCHS):
    # ---------------- Training ----------------
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for audio, labels in train_loader:
        audio = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(audio)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        train_loss_sum += loss.item() * batch_size
        train_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        train_total += batch_size

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    # ---------------- Validation ----------------
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for audio, labels in val_loader:
            audio = audio.to(device)
            labels = labels.to(device)

            logits = model(audio)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            val_loss_sum += loss.item() * batch_size
            val_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            val_total += batch_size

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_accuracy:.3f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_accuracy:.3f}"
    )

    # Save the best checkpoint using validation performance.
    improved = (
        val_accuracy > best_val_accuracy
        or (
            val_accuracy == best_val_accuracy
            and val_loss < best_val_loss
        )
    )

    if improved:
        best_val_accuracy = val_accuracy
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            OUTPUT / "best_model.pt",
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping.")
        break

# Save learning curves.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history["train_accuracy"], label="Train")
axes[1].plot(history["val_accuracy"], label="Validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT / "training_curves.png", dpi=150)
plt.show()

print(f"Best validation accuracy: {best_val_accuracy:.2%}")
print("Saved model:", OUTPUT / "best_model.pt")
