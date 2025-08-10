import tensorflow as tf
import numpy as np

def predict_seizure(features):
    """Mock prediction (replace with your trained LSTM later)"""
    # For now: 20% chance of fake "seizure"
    return 1 if np.random.rand() > 0.8 else 0  

# Later: Load your real LSTM model
# model = tf.keras.models.load_model('eeg_model.h5')