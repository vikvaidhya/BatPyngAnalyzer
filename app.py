import streamlit as st
import librosa
import numpy as np
import os
import glob
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment

# --- Configuration & Helpers ---
CALIB_FILE = "calibration_ref.json"

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
    # Auto-convert to WAV for librosa compatibility
    temp_wav = "temp_proc.wav"
    audio = AudioSegment.from_file(wav_path)
    audio.export(temp_wav, format="wav")
    
    y, sr = librosa.load(temp_wav, sr=None)
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    peak = np.argmax(rms)
    stft = np.abs(librosa.stft(y[peak*256 : peak*256 + int(0.04*sr)], n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    return float(freqs[np.argmax(np.mean(stft, axis=1))]), float(np.mean(rms) * 1000)

def get_calibrated_score(energy):
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r") as f: ref = json.load(f).get("ref_energy", 50.0)
        return int(np.clip((energy / ref) * 100, 1, 100))
    return int(np.clip(energy * 2, 1, 100))

# --- Classification Engine ---
def train_and_classify(new_features, grade_choice):
    X, y = [], []
    for grade_path in glob.glob("training_data/english/*"):
        grade_name = os.path.basename(grade_path)
        for m in set([os.path.basename(f).replace("BallTest.m4a", "").replace("MalletTest.m4a", "") for f in glob.glob(f"{grade_path}/*.m4a")]):
            b, m_f = os.path.join(grade_path, f"{m}BallTest.m4a"), os.path.join(grade_path, f"{m}MalletTest.m4a")
            if os.path.exists(b) and os.path.exists(m_f):
                p_b, _ = get_physics_data(b)
                p_m, energy = get_physics_data(m_f)
                meta_file = os.path.join(grade_path, f"{m}_meta.txt")
                w, g = (map(float, open(meta_file, "r").read().split(',')) if os.path.exists(meta_file) else [1150.0, 8.0])
                X.append([p_m / (p_b + 1e-10), p_m, energy, w, g])
                y.append(grade_name.upper())
    
    if not X: return "Train Data Empty"
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KNeighborsClassifier(n_neighbors=1).fit(X_scaled, y)
    return model.predict(scaler.transform([new_features]))[0]

# --- UI Layer ---
st.title("🏏 Cricket Bat Performance Lab")
tab1, tab2 = st.tabs(["Analyze Bat", "⚙️ Owner Calibration"])

with tab2:
    st.subheader("Calibration Mode")
    cal_ball = st.file_uploader("Reference Ball Test")
    cal_mallet = st.file_uploader("Reference Mallet Test")
    if st.button("Set Gold Standard"):
        _, energy = get_physics_data(cal_mallet)
        with open(CALIB_FILE, "w") as f: 
            json.dump({"ref_energy": float(energy)}, f)
        st.success(f"Baseline updated: {int(energy)}")

with tab1:
    grade_choice = st.selectbox("Grade", ["GRADE1", "GRADE2", "GRADE3", "GRADE4", "GRADE5"])
    ball, mallet = st.file_uploader("Ball Test"), st.file_uploader("Mallet Test")
    weight = st.number_input("Weight (g)", 1000, 1300)
    grains = st.number_input("Grains", 5, 15)
    
    if st.button("Analyze & Save"):
        folder = os.path.join("training_data/english", grade_choice.lower())
        os.makedirs(folder, exist_ok=True)
        m_name = f"Bat_{np.random.randint(1000,9999)}"
        
        with open(os.path.join(folder, f"{m_name}BallTest.m4a"), "wb") as f: f.write(ball.getbuffer())
        with open(os.path.join(folder, f"{m_name}MalletTest.m4a"), "wb") as f: f.write(mallet.getbuffer())
        with open(os.path.join(folder, f"{m_name}_meta.txt"), "w") as f: f.write(f"{weight},{grains}")
        
        p_b, _ = get_physics_data(os.path.join(folder, f"{m_name}BallTest.m4a"))
        p_m, energy = get_physics_data(os.path.join(folder, f"{m_name}MalletTest.m4a"))
        
        score = get_calibrated_score(energy)
        grade = train_and_classify([p_m / (p_b + 1e-10), p_m, energy, weight, grains], grade_choice)
        
        st.metric("Overall Quality Index", f"{score}/100")
        st.success(f"Final Determination: **{grade}**")