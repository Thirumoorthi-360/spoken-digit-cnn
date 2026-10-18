
# Spoken Digit Recognition using DSP and 1D CNN

## Overview
A spoken English digit classifier built with a real audio
dataset, digital signal processing and PyTorch.

## Dataset
Free Spoken Digit Dataset (FSDD):
https://github.com/Jakobovski/free-spoken-digit-dataset

The recordings are mono speech audio sampled at 8 kHz.
Digit labels are extracted from the WAV filenames.

## Pipeline
1. Load real WAV recordings
2. Remove DC offset and normalize amplitude
3. Pad or crop signals to a fixed length
4. Visualize waveforms and spectrograms
5. Train a 1D convolutional neural network
6. Evaluate using accuracy, precision, recall and F1-score
7. Predict digits from individual WAV files

## Run
Install dependencies:

    pip install -r requirements.txt

Inspect audio:

    python src/inspect_data.py

Train:

    python src/train.py

Evaluate:

    python src/evaluate.py

Predict:

    python src/predict.py "path/to/audio.wav"

## Dataset split
Recording indices 0-4: test
Recording indices 5-9: validation
Recording indices 10-49: training

## Results
Add measured test accuracy, per-class F1-score,
confusion matrix and training curves after running the project.

## Limitations
The dataset is small and contains recordings from a limited
number of speakers. Good test performance does not guarantee
robust recognition of new speakers or noisy real-world audio.

## License
Check the FSDD license and preserve its attribution
and license requirements when redistributing dataset material.
