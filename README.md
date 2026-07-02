# SeizureDetection — CNN + LSTM Hybrid Seizure Classifier

Deep learning pipeline for automated EEG seizure detection. Uses a **CNN + LSTM hybrid architecture** trained on the CHB-MIT Scalp EEG benchmark dataset — CNN layers extract spatial features across electrode channels, LSTM layers model temporal dynamics of seizure evolution.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow) ![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras) ![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy)

---

## Model Architecture

```
Input: (channels, time_steps, 1)
        |
Conv1D + BatchNorm + MaxPool   — spatial feature extraction per channel
        |
Conv1D + BatchNorm + MaxPool   — higher-order spatial patterns
        |
TimeDistributed(Flatten)       — reshape for temporal modelling
        |
LSTM(128) + Dropout(0.3)       — temporal sequence modelling
        |
LSTM(64)  + Dropout(0.3)
        |
Dense(64, relu) + Dense(1, sigmoid)
        |
Binary output: seizure / non-seizure
```

---

## Dataset

**CHB-MIT Scalp EEG Database** — a widely used benchmark for seizure detection research.

- 23 subjects, 664 hours of continuous EEG recordings
- 198 annotated seizure events
- 23-channel 10-20 system electrode placement
- 256 Hz sampling rate

The dataset is available from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/).

---

## Getting Started

### Installation

```bash
git clone https://github.com/SkylordAK/SeizureDetectionProject.git
cd SeizureDetectionProject
pip install -r requirements.txt
```

### Download the CHB-MIT Dataset

```bash
# Using wget (Linux/macOS)
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# Or download via the PhysioNet web interface
```

### Train the Model

```bash
python train.py --data-dir ./chb-mit --epochs 50 --batch-size 32
```

### Evaluate

```bash
python evaluate.py --model checkpoints/best_model.h5 --data-dir ./chb-mit
```

---

## Pipeline

1. **Data loading** — reads `.edf` EEG files and seizure annotation `.seizures` files
2. **Preprocessing** — bandpass filter (0.5–40 Hz), z-score normalisation per channel
3. **Epoch extraction** — fixed-length sliding windows (default: 2 seconds, 50% overlap)
4. **Class balancing** — seizure epochs are heavily outnumbered; addressed via class weights
5. **Training** — CNN+LSTM model with early stopping and model checkpointing
6. **Evaluation** — sensitivity, specificity, AUC-ROC reported per subject and overall

---

## Requirements

```
tensorflow>=2.13
keras>=3.0
numpy>=1.24
pandas>=2.0
scipy>=1.12
pyedflib>=0.1
scikit-learn>=1.4
matplotlib>=3.8
```

---

## Related Projects

- [EpilepsyDetection](https://github.com/SkylordAK/EpilepsyDetection) — Signal-processing-only real-time detector (no deep learning)

---

## Disclaimer

Research and educational project. Not a certified medical device and must not be used for clinical diagnosis or patient monitoring.

## License

MIT
