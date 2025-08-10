import numpy as np
import pywt
from scipy.signal import butter, filtfilt

def preprocess_eeg(signal, fs=256):
    """Bandpass filter (0.5-40Hz)"""
    nyq = 0.5 * fs
    low, high = 0.5 / nyq, 40 / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_wavelet_features(signal, wavelet='db4', levels=5):
    """Extract wavelet coefficients' stats"""
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    return np.concatenate([np.mean(c) for c in coeffs])