# SeizureDetectionProject — Codebase Audit Report

**Audited by:** Claude Sonnet 4.6 (Senior Software Architect mode)
**Date:** 2026-06-28

---

## 1. Project Overview

**Purpose:** Offline seizure detection ML pipeline. Preprocesses EEG data, trains a 1D CNN + LSTM hybrid model, and evaluates it on a 3-class classification task (Normal / Seizure / Epilepsy).

**Tech Stack:** Python / TensorFlow 2.x / Keras / NumPy / Pandas / scikit-learn / matplotlib

**Files:** `data_prep.py`, `train_seizure_model.py`, `eeg_recorder.py`, `README.md`

---

## 2. Issues Found

### HIGH

#### H-1: Model saved as `.h5` — deprecated in TensorFlow 2.12+
**File:** `train_seizure_model.py` (line 78)

```python
checkpoint_filepath = 'Optimized_Seizure_Model.h5'
```

The Keras `.h5` format is deprecated in TensorFlow 2.12+ in favor of the SavedModel format (`.keras`). `ModelCheckpoint` with `.h5` will produce deprecation warnings and may behave incorrectly in newer TF versions.

**Fix:** Change to `'Optimized_Seizure_Model.keras'` and ensure `save_format='keras'`.

---

#### H-2: `EarlyStopping` and `ModelCheckpoint` monitor different metrics — potential mismatch
**File:** `train_seizure_model.py` (lines 79–82)

```python
EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
ModelCheckpoint(..., monitor='val_accuracy', save_best_only=True, ...)
```

`EarlyStopping` monitors `val_loss` (restores best weights for loss), while `ModelCheckpoint` saves based on `val_accuracy`. These can diverge: the "best" loss epoch may not be the "best" accuracy epoch. Combined with `restore_best_weights=True`, the final model weights come from the best `val_loss` epoch, but the saved `.h5` file comes from the best `val_accuracy` epoch. When `model.load_weights(checkpoint_filepath)` is called on line 95, it overrides `EarlyStopping`'s restored weights with the checkpoint — making `restore_best_weights=True` effectively useless.

**Fix:** Make both callbacks monitor the same metric (recommend `val_loss`), or remove the `load_weights` call since `restore_best_weights=True` already handles it.

---

#### H-3: Class "Epilepsy" vs "Seizure" distinction is clinically ambiguous
**File:** `train_seizure_model.py` (line 62)

```python
classes = ['Normal', 'Seizure', 'Epilepsy']
```

The distinction between class 1 ("Seizure") and class 2 ("Epilepsy") is not documented and is clinically unusual — epilepsy is a chronic condition characterized by seizures, not a distinct EEG state. This likely reflects the source dataset's labeling schema (e.g., the Bonn or CHB-MIT dataset), but it should be explicitly documented to avoid misinterpretation.

---

### MEDIUM

#### M-1: `data_prep.py` not audited — data pipeline is a black box
**File:** `data_prep.py`

The training script assumes `ProcessedData/X.npy` and `ProcessedData/y.npy` exist. If `data_prep.py` has bugs (wrong window size, incorrect normalization, label leakage), the trained model will be invalid. The data prep script should be audited separately.

---

#### M-2: No class imbalance handling
**File:** `train_seizure_model.py` (throughout)

EEG seizure datasets are typically highly imbalanced (seizures are rare events). The training script uses no `class_weight` parameter in `model.fit()`. If the dataset is imbalanced, the model will learn to predict the majority class and achieve misleading accuracy scores.

**Fix:** Add `class_weight` parameter using `sklearn.utils.class_weight.compute_class_weight`.

---

#### M-3: No reproducibility seed for TensorFlow operations
**File:** `train_seizure_model.py`

`train_test_split` uses `random_state=42`, but TensorFlow GPU operations are non-deterministic without `tf.random.set_seed()`. Results will differ between runs.

**Fix:** Add `tf.random.set_seed(42)` at the top of `main()`.

---

#### M-4: Confusion matrix printed but not saved
**File:** `train_seizure_model.py` (lines 100–106)

The confusion matrix is printed to stdout and not saved. For a medical application, audit trails and reproducible evaluation artifacts (plots, CSV) are important.

---

### LOW

#### L-1: No `requirements.txt`

TensorFlow, scikit-learn, pandas, matplotlib dependencies are not listed.

#### L-2: Model hyperparameters are hardcoded
`BATCH_SIZE`, `EPOCHS`, `kernel_size`, `filters` are all magic numbers with no configuration mechanism. Consider a config dict or argparse.

---

## 3. Fixes Applied

None — ML pipeline changes require domain expertise validation before applying.

---

## 4. Recommendations

1. Unify the `EarlyStopping` and `ModelCheckpoint` monitoring metric (H-2).
2. Switch checkpoint format to `.keras` (H-1).
3. Add `class_weight` for imbalanced dataset handling (M-2).
4. Add `tf.random.set_seed(42)` for reproducibility (M-3).
5. Save confusion matrix as a CSV/PNG for reproducibility (M-4).
6. Add `requirements.txt`.
7. Document the data sources and labeling schema for the 3 classes (H-3).
