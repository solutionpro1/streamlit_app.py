import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import pywt
import re  # Regular expressions for flexible input parsing

# --- LSTM Model Setup ---
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_model():
    if 'model' not in st.session_state:
        model = create_lstm_model((None, 1))
        st.session_state.model = model
    return st.session_state.model

# --- EEG Processing ---
def preprocess_eeg(data):
    data = np.array(data)
    data = (data - np.mean(data)) / np.std(data)
    return data.reshape(1, -1, 1)

def parse_eeg_input(raw_input):
    """Convert various input formats to numeric list"""
    # Remove brackets, commas, and extra spaces
    cleaned = re.sub(r'[\[\],]', ' ', raw_input)
    # Split on any whitespace
    return [float(x) for x in cleaned.split() if x]

def predict_seizure(data):
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
    .eeg-input {
        font-family: monospace;
        white-space: pre;
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
    
    st.markdown("""
    <div style='color: gray; font-size: 0.9em; margin-bottom: 10px;'>
        Enter EEG values separated by spaces, commas, or newlines:
    </div>
    """, unsafe_allow_html=True)
    
    # EEG Data Input - More flexible format
    sample_data = """-1.4408 1.2876 -1.0992 -0.4306 
    1.4696 0.1682 1.228 -0.1394"""
    
    eeg_data = st.text_area(
        "Paste EEG values here:",
        value=sample_data,
        height=120,
        key="eeg-input"
    )
    
    if st.button("📌 Predict from Raw EEG"):
        try:
            data = parse_eeg_input(eeg_data)
            
            if len(data) < 10:
                st.warning(f"Warning: Only {len(data)} points detected. For best results, provide at least 384 samples (1.5s at 256Hz).")
            
            with st.spinner("Analyzing..."):
                seizure_prob = predict_seizure(data)
                seizure_detected = seizure_prob > 0.5
            
            # Display results
            if seizure_detected:
                st.error(f"🚨 Seizure Detected (confidence: {seizure_prob:.0%})")
            else:
                st.success(f"✅ Normal EEG (confidence: {1-seizure_prob:.0%})")
            
            # Confidence meter
            st.markdown(f"""
            <div style='display:flex; margin-top:20px'>
                <div style='display:flex; flex-direction:column; justify-content:space-between; height:220px; font-size:0.8em;'>
                    {"".join(f"<span>{1-i*0.1:.1f}</span>" for i in range(11))}
                </div>
                <div style='margin-left:10px; width:25px; height:220px; background:linear-gradient(to top, #ff4b4b, #4CAF50); position:relative'>
                    <div style='position:absolute; width:100%; height:{100 - (seizure_prob * 100):.1f}%; bottom:0; background:rgba(255,255,255,0.5)'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Plot EEG
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(data)
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude (μV)")
            st.pyplot(fig)
            
            # Show stats
            with st.expander("📊 Signal Statistics"):
                st.write(f"Mean: {np.mean(data):.2f} μV | Std: {np.std(data):.2f} μV")
                st.write(f"Min: {np.min(data):.2f} μV | Max: {np.max(data):.2f} μV")
                
        except ValueError as e:
            st.error(f"Invalid input: {str(e)}. Please enter numeric values only.")

# Footer
st.markdown("---")
st.markdown("""
<h4>Nitrogen</h4>
<ul style='margin-top:-15px; list-style-type:none; padding-left:5px'>
    <li>• Strain rate</li>
</ul>
""", unsafe_allow_html=True)
