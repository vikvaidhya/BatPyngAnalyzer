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

# --- System Setup ---
# Force pydub to find ffmpeg in the Linux environment
AudioSegment.converter = "/usr/bin/ffmpeg"
CALIB_FILE = "calibration_ref.json"
TRAIN_DIR = "training_data"

# --- Physics & Classification Engine ---
def get_physics_data(wav_path):
    temp_wav = "temp_proc.wav"
    audio = AudioSegment.from_file(wav_path)
    audio.export(temp_wav, format="wav")
    y, sr = librosa.load(temp_wav, sr=None)
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    peak = np.argmax(rms)
    stft = np.abs(librosa.stft(y[peak*256 : peak*256 + int(0.04*sr)], n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    return float(freqs[np.argmax(np.mean(stft, axis=1))]), float(np.mean(rms) * 1000)

def get_score(energy):
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r") as f: refs = json.load(f)
        ref = refs.get("English_G1", 50.0)
        return int(np.clip((energy / ref) * 100, 1, 100))
    return int(np.clip(energy * 2, 1, 100))

def train_and_classify(new_features):
    X, y = [], []
    for grade_path in glob.glob(f"{TRAIN_DIR}/*"):
        grade_name = os.path.basename(grade_path)
        for b_path in glob.glob(f"{grade_path}/*_ball.m4a"):
            m_path = b_path.replace("_ball.m4a", "_mallet.m4a")
            meta_path = b_path.replace("_ball.m4a", "_meta.txt")
            if os.path.exists(m_path) and os.path.exists(meta_path):
                p_b, _ = get_physics_data(b_path)
                p_m, energy = get_physics_data(m_path)
                with open(meta_path, "r") as f: w, g = map(float, f.read().split(','))
                X.append([p_m / (p_b + 1e-10), p_m, energy, w, g])
                y.append(grade_name)
    
    if len(X) < 2: return "Need more training data"
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KNeighborsClassifier(n_neighbors=1).fit(X_scaled, y)
    return model.predict(scaler.transform([new_features]))[0]

# --- UI Layer ---
st.title("🏏 Cricket Bat Performance Lab v4.2")
tab1, tab2 = st.tabs(["Analyze Bat", "⚙️ Owner Calibration"])

with tab2:
    st.subheader("Calibration")
    type_ref = st.selectbox("Select Baseline", ["English_G1", "Kashmir_Natural"])
    c_mallet = st.file_uploader("Ref Mallet Ping")
    if st.button("Set Baseline") and c_mallet:
        temp_cal = "temp_cal.m4a"
        with open(temp_cal, "wb") as f: f.write(c_mallet.getbuffer())
        _, energy = get_physics_data(temp_cal)
        refs = json.load(open(CALIB_FILE)) if os.path.exists(CALIB_FILE) else {}
        refs[type_ref] = float(energy)
        with open(CALIB_FILE, "w") as f: json.dump(refs, f)
        st.success(f"Baseline {type_ref} set.")

with tab1:
    ball, mallet = st.file_uploader("Upload Ball Ping"), st.file_uploader("Upload Mallet Ping")
    w, g = st.number_input("Weight (g)", 1000, 1300), st.number_input("Grains", 0, 20)
    
    if st.button("Analyze Bat") and ball and mallet:
        os.makedirs("temp", exist_ok=True)
        b_path, m_path = "temp/b.m4a", "temp/m.m4a"
        with open(b_path, "wb") as f: f.write(ball.getbuffer())
        with open(m_path, "wb") as f: f.write(mallet.getbuffer())
        
        p_b, _ = get_physics_data(b_path)
        p_m, energy = get_physics_data(m_path)
        
        # Prediction
        prediction = train_and_classify([p_m / (p_b + 1e-10), p_m, energy, w, g])
        st.metric("Performance Index", f"{get_score(energy)}/100")
        st.info(f"AI Prediction: **{prediction.replace('_', ' ').upper()}**")
        
        # Feedback Loop
        actual_grade = st.selectbox("Correct Grade (if wrong)", ["english_g1", "english_g2", "kashmir_natural"])
        if st.button("Save Feedback (Retrain AI)"):
            target_dir = os.path.join(TRAIN_DIR, actual_grade)
            os.makedirs(target_dir, exist_ok=True)
            bid = np.random.randint(1000, 9999)
            shutil.copy(b_path, os.path.join(target_dir, f"{bid}_ball.m4a"))
            shutil.copy(m_path, os.path.join(target_dir, f"{bid}_mallet.m4a"))
            with open(os.path.join(target_dir, f"{bid}_meta.txt"), "w") as f: f.write(f"{w},{g}")
            st.success("Data saved and model retrained.")
            st.rerun()

# --- Knowledge Base Stats ---
st.sidebar.write("### Knowledge Base")
stats = {os.path.basename(d): len(glob.glob(f"{d}/*.m4a")) for d in glob.glob(f"{TRAIN_DIR}/*")}
st.sidebar.json(stats)