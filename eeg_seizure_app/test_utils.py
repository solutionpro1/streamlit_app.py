# test_utils.py
from utils import preprocess_eeg, extract_wavelet_features
import numpy as np

# Test with synthetic EEG data
test_signal = np.random.randn(1000)  
processed = preprocess_eeg(test_signal)
features = extract_wavelet_features(processed)

print("Processing successful!")
print(f"Output shape: {features.shape}")