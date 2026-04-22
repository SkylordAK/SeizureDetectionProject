import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Configurations
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = r"ProcessedData"

def build_hybrid_model(input_shape, num_classes):
    """
    Builds an optimized 1D CNN + LSTM hybrid model.
    """
    model = Sequential([
        # 1D Convolutional block for local temporal-spatial feature extraction per channel
        Conv1D(filters=32, kernel_size=10, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=64, kernel_size=10, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # LSTM block for learning long-term dependencies over the window
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        
        # Fully connected layer
        Dense(32, activation='relu'),
        Dropout(0.4),
        
        # Output layer for 3 classes
        Dense(num_classes, activation='softmax')
    ])
    
    # We use sparse categorical crossentropy because labels are integers (0, 1, 2)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Step 1: Loading Preprocessed Data...")
    x_path = os.path.join(DATA_DIR, 'X.npy')
    y_path = os.path.join(DATA_DIR, 'y.npy')
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print("Data files not found in ProcessedData directory. Please run data_prep.py first.")
        return
        
    X = np.load(x_path)
    y = np.load(y_path)
    
    # 0 = Normal, 1 = Seizure, 2 = Epilepsy
    classes = ['Normal', 'Seizure', 'Epilepsy']
    
    print("Step 2: Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    
    print(f"Training set:   {X_train.shape[0]} epochs")
    print(f"Validation set: {X_val.shape[0]} epochs")
    print(f"Test set:       {X_test.shape[0]} epochs")

    print("\nStep 3: Building Active Model...")
    input_shape = (X_train.shape[1], X_train.shape[2]) # (Samples, Channels)
    model = build_hybrid_model(input_shape, num_classes=3)
    model.summary()
    
    # Callbacks
    checkpoint_filepath = 'Optimized_Seizure_Model.h5'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    print("\nStep 4: Training Model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    print("\nStep 5: Evaluating Final Model on Test Data...")
    # Load best weights
    model.load_weights(checkpoint_filepath)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"--- Test Accuracy: {test_acc*100:.2f}% ---")
    
    # Predictions and Confusion Matrix
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df)

if __name__ == "__main__":
    main()
