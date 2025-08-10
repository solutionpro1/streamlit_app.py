import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import pywt

# --- LSTM Model Setup ---
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize or load your trained model
def get_model():
    if 'model' not in st.session_state:
        # Create a dummy model (replace with your trained model)
        model = create_lstm_model((None, 1))  # Adjust input shape as needed
        # Train with dummy data or load weights here
        st.session_state.model = model
    return st.session_state.model

# --- EEG Processing ---
def preprocess_eeg(data):
    """Simple preprocessing - normalize and reshape for LSTM"""
    data = np.array(data)
    # Normalize
    data = (data - np.mean(data)) / np.std(data)
    # Reshape for LSTM (samples, timesteps, features)
    return data.reshape(1, -1, 1)

def predict_seizure(data):
    """Make prediction using LSTM model"""
    model = get_model()
    processed_data = preprocess_eeg(data)
    return model.predict(processed_data)[0][0]

# --- Streamlit App ---
st.set_page_config(layout="centered")

# Custom CSS
st.markdown("""
<style>
    .status-box {
        background: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"# {datetime.now().strftime('%I:%M %p')}")

# Two-column layout
col1, col2 = st.columns([1, 2])

# Left Column - Status Panel
with col1:
    st.markdown("""
    <div class='status-box'>
        <h3>1.0</h3>
        <ul style='margin-top:-15px; list-style-type:none; padding-left:5px'>
            <li>• Non-Seizure</li>
            <li>• Seizure</li>
        </ul>
        <hr style='margin:10px 0; border-top:1px dashed #e0e0e0'>
        <h4>Fork</h4>
        <ul style='margin-top:-15px; list-style-type:none; padding-left:5px'>
            <li>• Tue Predicted</li>
            <li>• Total</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Right Column - EEG Input
with col2:
    st.markdown("### Product from Raw EEG Signal")
    
    # EEG Data Input
    sample_data = "-1.4408, 1.2876, -1.0992, -0.4306, 1.4696, 0.1682, 1.228, -0.1394"
    eeg_data = st.text_area(
        "Enter EEG values (comma separated)",
        value=sample_data,
        height=120
    )
    
    if st.button("📌 Predict from Raw EEG"):
        try:
            data = [float(x.strip()) for x in eeg_data.split(",") if x.strip()]
            
            if len(data) < 10:  # Minimum reasonable length
                st.warning("Warning: Very short EEG segment - accuracy may be affected")
            
            # Make prediction
            with st.spinner("Analyzing..."):
                seizure_prob = predict_seizure(data)
                seizure_detected = seizure_prob > 0.5
            
            # Display results
            if seizure_detected:
                st.error(f"🚨 Seizure Detected (confidence: {seizure_prob:.0%})")
            else:
                st.success(f"✅ Normal EEG (confidence: {1-seizure_prob:.0%})")
            
            # Confidence meter
            st.markdown("""
            <div style='display:flex; margin-top:20px'>
                <div style='display:flex; flex-direction:column; justify-content:space-between; height:220px'>
                    <span>1.0</span><span>0.9</span><span>0.8</span><span>0.7</span><span>0.6</span>
                    <span>0.5</span><span>0.4</span><span>0.3</span><span>0.2</span><span>0.1</span><span>0.0</span>
                </div>
                <div style='margin-left:10px; width:25px; height:220px; background:linear-gradient(to top, red, green); position:relative'>
                    <div style='position:absolute; width:100%; height:{}%; bottom:0; background:rgba(255,255,255,0.5)'></div>
                </div>
            </div>
            """.format(100 - (seizure_prob * 100)), unsafe_allow_html=True)
            
            # Plot EEG
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(data)
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
        except ValueError:
            st.error("Invalid input. Use comma-separated numbers only.")

# Footer
st.markdown("---")
st.markdown("""
<h4>Nitrogen</h4>
<ul style='margin-top:-15px; list-style-type:none; padding-left:5px'>
    <li>• Strain rate</li>
</ul>
""", unsafe_allow_html=True)