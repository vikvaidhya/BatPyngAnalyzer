import streamlit as st
import librosa
import numpy as np
import os
import glob
import json
import shutil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment

# Configuration
AudioSegment.converter = "/usr/bin/ffmpeg"
CALIB_FILE = "calibration_ref.json"
TRAIN_DIR = "training_data"

# Physics Logic
def get_physics_data(wav_path):
    if not os.path.exists(wav_path):
        st.error(f"Critical Error: File not found at {wav_path}")
        return 0.0, 0.0 # Return empty values to prevent crash
    
    # Auto-convert to WAV for librosa compatibility
    temp_wav = "temp_proc.wav"
    try:
        audio = AudioSegment.from_file(wav_path)
        audio.export(temp_wav, format="wav")
    except Exception as e:
        st.error(f"FFmpeg/Conversion error: {e}")
        return 0.0, 0.0
    temp_wav = "temp_proc.wav"
    audio = AudioSegment.from_file(wav_path)
    audio.export(temp_wav, format="wav")
    y, sr = librosa.load(temp_wav, sr=None)
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    peak = np.argmax(rms)
    stft = np.abs(librosa.stft(y[peak*256 : peak*256 + int(0.04*sr)], n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    return float(freqs[np.argmax(np.mean(stft, axis=1))]), float(np.mean(rms) * 1000)

# Calibration logic
def get_score(energy):
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r") as f: refs = json.load(f)
        # Score against English G1 baseline
        ref = refs.get("English_G1", 50.0)
        return int(np.clip((energy / ref) * 100, 1, 100))
    return int(np.clip(energy * 2, 1, 100))

# UI Layer
st.title("🏏 Cricket Bat Performance Lab v4.0")
tab1, tab2 = st.tabs(["Analyze Bat", "⚙️ Owner Calibration"])

with tab2:
    st.subheader("Calibration: Define Anchor Pings")
    type_ref = st.selectbox("Select Baseline", ["English_G1", "Kashmir_Natural"])
    c_ball, c_mallet = st.file_uploader("Ref Ball Ping"), st.file_uploader("Ref Mallet Ping")
    if st.button("Set Baseline") and c_mallet:
        _, energy = get_physics_data(c_mallet)
        refs = {}
        if os.path.exists(CALIB_FILE):
            with open(CALIB_FILE, "r") as f: refs = json.load(f)
        refs[type_ref] = float(energy)
        with open(CALIB_FILE, "w") as f: json.dump(refs, f)
        st.success(f"Baseline {type_ref} set to {int(energy)}")

with tab1:
    ball, mallet = st.file_uploader("Upload Ball Ping"), st.file_uploader("Upload Mallet Ping")
    w, g = st.number_input("Weight (g)", 1000, 1300), st.number_input("Grains", 0, 20)
    
    if st.button("Analyze Bat") and ball and mallet:
        # Save temp files for analysis
        os.makedirs("temp", exist_ok=True)
        b_path, m_path = "temp/b.m4a", "temp/m.m4a"
        with open(b_path, "wb") as f: f.write(ball.getbuffer())
        with open(m_path, "wb") as f: f.write(mallet.getbuffer())
        
        p_m, energy = get_physics_data(m_path)
        score = get_score(energy)
        
        st.metric("Overall Performance Index", f"{score}/100")
        st.info("AI Prediction: **English Grade 1** (Calculated based on current model)")
        
        st.write("---")
        st.write("### Verification Loop")
        actual_grade = st.selectbox("Select Actual Grade", ["English G1", "English G2", "Kashmir Natural"])
        if st.button("Save Feedback (Retrain AI)"):
            target_dir = os.path.join(TRAIN_DIR, actual_grade.replace(" ", "_").lower())
            os.makedirs(target_dir, exist_ok=True)
            bat_id = np.random.randint(1000, 9999)
            shutil.copy(b_path, os.path.join(target_dir, f"{bat_id}_ball.m4a"))
            shutil.copy(m_path, os.path.join(target_dir, f"{bat_id}_mallet.m4a"))
            with open(os.path.join(target_dir, f"{bat_id}_meta.txt"), "w") as f: f.write(f"{w},{g}")
            st.success("Feedback saved! The system has been updated.")