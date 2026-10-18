
import torch
from torch import nn


class SpokenDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = SpokenDigitCNN()
    dummy = torch.randn(8, 1, 16000)
    output = model(dummy)

    print("Input shape:", dummy.shape)
    print("Output shape:", output.shape)
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    ))
