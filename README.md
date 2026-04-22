# Seizure Detection Study — Data Acquisition & Modeling

This project is dedicated to the scientific study of seizure activity using wearable EEG technology. It includes tools for recording high-fidelity neural data and training machine learning models for seizure classification.

## Project Structure
- `eeg_recorder.py`: A robust tool for recording labelled EEG segments (Normal, Epilepsy, Seizure, Rest).
- `data_prep.py`: Processes raw CSV recordings into windowed features (Band Powers, Ratios) suitable for ML.
- `train_seizure_model.py`: Trains a Random Forest classifier to distinguish between resting and ictal (seizure) states.

## Data Recording Protocol
The recorder follows a structured schedule to ensure balanced datasets:
1. **Break (10s)**: Warm-up.
2. **Normal (60s)**: Baseline resting EEG.
3. **Epilepsy (60s)**: Pre-ictal or inter-ictal activity.
4. **Seizure (60s)**: Active ictal activity.

## Feature Engineering
The system extracts:
- Absolute and Relative band powers.
- Theta/Beta and Alpha/Beta ratios.
- Total spectral power (Root Mean Square).

## Requirements
- Python 3.8+
- `scikit-learn`, `pandas`, `numpy`, `python-osc`

## Usage
1. **Record Data**: Run `eeg_recorder.py` and follow the timed prompts.
2. **Prepare Features**:
   ```bash
   python data_prep.py
   ```
3. **Train Model**:
   ```bash
   python train_seizure_model.py
   ```
   This will output a model file and a classification report with accuracy, precision, and recall metrics.
