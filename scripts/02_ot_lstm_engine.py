# 02_ot_lstm_engine.py - Optimized Transformer-LSTM Engine for Time-Series Prediction

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, TimeDistributed, Input
from tensorflow.keras.models import Model
import tensorflow as tf

# --- 1. CONFIGURATION ---
PROCESSED_DATA_FILE = 'data/processed/data_processed.csv'
MODEL_SAVE_PATH = 'models/ot_lstm_model.h5'
SEQUENCE_LENGTH = 30 # Number of time steps (rows) to look back
BATCH_SIZE = 32
EPOCHS = 50

# --- 2. DATA UTILITIES ---

def create_sequences(data, seq_length):
    """Creates time-series sequences and corresponding labels."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        # X: sequences of 'seq_length' rows (e.g., last 30 days)
        X.append(data[i-seq_length:i, :])
        # y: The prediction target (e.g., the first feature of the current row)
        y.append(data[i, 0]) 
    return np.array(X), np.array(y)

# --- 3. OT-LSTM MODEL ARCHITECTURE ---

def transformer_block(inputs, d_k, d_v, d_model, n_heads):
    """Generalized Transformer (Multi-Head Attention) Block."""
    attention_output = MultiHeadAttention(
        key_dim=d_k, num_heads=n_heads, output_shape=d_model
    )(inputs, inputs)
    
    # Add & Norm
    norm_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    return norm_output

def build_ot_lstm_model(input_shape, n_features):
    """Builds the Optimised Transformer-LSTM Model."""
    
    # Inputs: (SEQUENCE_LENGTH, N_FEATURES)
    inputs = Input(shape=input_shape)

    # 1. OPTIMIZED TRANSFORMER (Attention) LAYER
    # This block processes the input sequence features using attention
    transformer = transformer_block(inputs, 
                                    d_k=32, d_v=32, d_model=input_shape[1], n_heads=4)

    # 2. LSTM SEQUENCE LAYER
    # Pass the attention-weighted features into an LSTM layer
    lstm_output = LSTM(units=100, return_sequences=False)(transformer)
    lstm_output = Dropout(0.2)(lstm_output)
    
    # 3. DENSE OUTPUT LAYER
    outputs = Dense(1, activation='linear')(lstm_output)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    print("--- Starting Phase 2: OT-LSTM Engine ---")
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    try:
        # Load processed data
        df = pd.read_csv(PROCESSED_DATA_FILE)
        
        # Prepare data for model
        data = df.values.astype(np.float32)
        n_features = data.shape[1]
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Build Model
        model = build_ot_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), 
                                    n_features=n_features)
        
        print("\nModel Summary:")
        model.summary()
        
        # Train Model
        print("\nStarting Model Training...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save Model
        model.save(MODEL_SAVE_PATH)
        print(f"\n--- SUCCESS: Model saved to {MODEL_SAVE_PATH} ---")
        
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Required data file not found at '{PROCESSED_DATA_FILE}'. Please ensure Phase 1 was completed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")