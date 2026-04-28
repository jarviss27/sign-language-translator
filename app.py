import streamlit as st
import pickle
import mediapipe as mp
import numpy as np
from PIL import Image
import os
from datetime import datetime

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Dexora",
    page_icon="❤︎",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #C9BEFF !important;
    color: #1a1340;
    font-family: 'Syne', sans-serif;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
footer { display: none !important; }
#MainMenu { display: none; }

/* ── Noise overlay ── */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: 0.4;
}

/* ── Ambient glow blobs ── */
body::after {
    content: '';
    position: fixed;
    top: -20vh;
    left: -20vw;
    width: 70vw;
    height: 70vh;
    background: radial-gradient(ellipse at center, rgba(90,60,180,0.13) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: drift 18s ease-in-out infinite alternate;
}

@keyframes drift {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(8vw, 6vh) scale(1.12); }
}

/* ── Page enter animation ── */
[data-testid="stAppViewContainer"] > .main > div {
    animation: fadeUp 0.55s ease both;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.8rem;
    position: relative;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #3c1f99;
    border: 1px solid rgba(60,31,153,0.35);
    border-radius: 100px;
    padding: 0.35rem 1.1rem;
    margin-bottom: 1.4rem;
    background: rgba(60,31,153,0.08);
    animation: fadeUp 0.4s ease 0.1s both;
}

.hero-badge::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #3c1f99;
    animation: pulse 2s ease infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.75); }
}

.hero-title {
    font-size: clamp(5rem, 12vw, 9rem);
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #1a1340 0%, #3c1f99 45%, #7b3fbf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.6rem;
    animation: fadeUp 0.45s ease 0.15s both;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: rgba(26,19,64,0.45);
    letter-spacing: 0.1em;
    animation: fadeUp 0.5s ease 0.2s both;
}

/* ── Mode switcher pill ── */
.mode-wrapper {
    display: flex;
    justify-content: center;
    margin: 1.4rem 0 1.6rem;
    animation: fadeUp 0.5s ease 0.25s both;
}

.mode-pill {
    display: inline-flex;
    align-items: center;
    gap: 0;
    background: rgba(255,255,255,0.28);
    border: 1.5px solid rgba(60,31,153,0.20);
    border-radius: 16px;
    padding: 5px;
    backdrop-filter: blur(10px);
}

.mode-btn {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    padding: 0.55rem 1.5rem;
    border: none;
    border-radius: 11px;
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
    background: transparent;
    color: rgba(26,19,64,0.40);
    letter-spacing: 0.02em;
    position: relative;
}

.mode-btn.active {
    background: linear-gradient(135deg, #3c1f99, #5b2d8e);
    color: #fff;
    box-shadow: 0 3px 18px rgba(60,31,153,0.35);
    transform: scale(1.04);
}

.mode-sep {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: rgba(60,31,153,0.5);
    user-select: none;
    letter-spacing: 0.05em;
}

.mode-sep .arr {
    display: inline-block;
    transition: transform 0.3s ease;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.22) !important;
    border: 1.5px dashed rgba(60,31,153,0.30) !important;
    border-radius: 20px !important;
    padding: 1.4rem 1.6rem !important;
    transition: border-color 0.3s, background 0.3s !important;
    backdrop-filter: blur(8px) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(60,31,153,0.6) !important;
    background: rgba(255,255,255,0.32) !important;
}

[data-testid="stFileUploader"] label {
    color: rgba(26,19,64,0.60) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #3c1f99, #5b2d8e) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1.2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}

[data-testid="stFileUploader"] button:hover {
    opacity: 0.88 !important;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.28);
    border: 1px solid rgba(60,31,153,0.16);
    border-radius: 22px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(14px);
    margin-top: 1.2rem;
    position: relative;
    overflow: hidden;
    animation: cardIn 0.5s cubic-bezier(0.22,1,0.36,1) both;
}

@keyframes cardIn {
    from { opacity: 0; transform: translateY(20px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(60,31,153,0.4), transparent);
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: rgba(60,31,153,0.70);
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(60,31,153,0.15), transparent);
}

/* ── Image display ── */
[data-testid="stImage"] { border-radius: 14px; overflow: hidden; }
[data-testid="stImage"] img { border-radius: 14px !important; }

/* ── Result display ── */
.result-wrapper {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    margin: 0.8rem 0 1.2rem;
    flex-wrap: wrap;
}

.result-meta { flex: 1; min-width: 160px; }

.letter-badge {
    width: 100px;
    height: 100px;
    border-radius: 26px;
    background: linear-gradient(145deg, #2d1580 0%, #3c1f99 40%, #5b2d8e 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.4rem;
    font-weight: 800;
    color: #fff;
    box-shadow:
        0 0 0 1px rgba(60,31,153,0.3),
        0 0 40px rgba(91,45,142,0.35),
        0 8px 24px rgba(26,19,64,0.22);
    flex-shrink: 0;
    letter-spacing: -0.02em;
    animation: letterPop 0.55s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
    position: relative;
    overflow: hidden;
}

.letter-badge::after {
    content: '';
    position: absolute;
    top: 0; left: -60%;
    width: 40%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
    animation: shimmer 3s ease 0.6s infinite;
}

@keyframes shimmer {
    0%   { left: -60%; }
    60%, 100% { left: 130%; }
}

@keyframes letterPop {
    0%   { transform: scale(0.4) rotate(-8deg); opacity: 0; }
    70%  { transform: scale(1.08) rotate(2deg); opacity: 1; }
    100% { transform: scale(1) rotate(0deg); opacity: 1; }
}

.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(26,19,64,0.40);
    margin-bottom: 0.2rem;
}

.result-letter-name {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1a1340;
    margin-bottom: 0.6rem;
    letter-spacing: -0.03em;
    line-height: 1;
}

/* ── Confidence bar ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.2rem;
}

.conf-bar-bg {
    flex: 1;
    height: 7px;
    background: rgba(26,19,64,0.08);
    border-radius: 100px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #3c1f99, #7b3fbf);
    animation: growBar 1.1s cubic-bezier(0.22,1,0.36,1) forwards;
}

@keyframes growBar {
    from { width: 0%; }
}

.conf-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    color: #3c1f99;
    min-width: 46px;
    text-align: right;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(60,31,153,0.16), transparent);
    margin: 1.3rem 0;
}

/* ── Feedback ── */
.feedback-section {
    margin-top: 0.2rem;
}

.feedback-q {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(60,31,153,0.7);
    margin-bottom: 0.75rem;
}

/* Streamlit radio override into pill chips */
[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 0.6rem !important;
}

[data-testid="stRadio"] label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.45rem !important;
    padding: 0.48rem 1.15rem !important;
    border-radius: 100px !important;
    border: 1.5px solid rgba(60,31,153,0.22) !important;
    background: rgba(255,255,255,0.28) !important;
    cursor: pointer !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: rgba(26,19,64,0.60) !important;
    transition: all 0.22s ease !important;
}

[data-testid="stRadio"] label:hover {
    border-color: rgba(60,31,153,0.5) !important;
    background: rgba(255,255,255,0.40) !important;
    color: #1a1340 !important;
}

[data-testid="stRadio"] [aria-checked="true"] + label,
[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(60,31,153,0.12) !important;
    border-color: #3c1f99 !important;
    color: #3c1f99 !important;
}

/* Hide the actual radio circles */
[data-testid="stRadio"] label > div:first-child { display: none !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { font-family: 'Syne', sans-serif !important; font-size: 0.82rem !important; font-weight: 600 !important; }

/* ── Selectbox override ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.28) !important;
    border: 1.5px solid rgba(60,31,153,0.22) !important;
    border-radius: 12px !important;
    color: #1a1340 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    backdrop-filter: blur(8px) !important;
}

[data-testid="stSelectbox"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: rgba(60,31,153,0.70) !important;
}

/* ── Submit button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3c1f99, #5b2d8e) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.4rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 14px rgba(60,31,153,0.28) !important;
    letter-spacing: 0.01em !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(60,31,153,0.38) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Success / error ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    border: none !important;
}

/* ── ASL reference ── */
.ref-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(91,45,142,0.75);
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.ref-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(91,45,142,0.15), transparent);
}

/* ── Columns ── */
[data-testid="stColumns"] > div { gap: 1rem; }

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #3c1f99;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
}

/* ── No hand detected ── */
.no-hand-box {
    background: rgba(180,60,30,0.06);
    border: 1.5px solid rgba(180,60,30,0.20);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-top: 1.2rem;
    animation: fadeUp 0.4s ease both;
}

.no-hand-icon {
    font-size: 1.8rem;
    flex-shrink: 0;
    line-height: 1;
    margin-top: 1px;
}

.no-hand-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1a1340;
    margin-bottom: 0.25rem;
}

.no-hand-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: rgba(26,19,64,0.58);
    line-height: 1.6;
}

/* ── Footer strip ── */
.footer-strip {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    color: rgba(26,19,64,0.28);
    text-transform: uppercase;
}

/* ── Mode toggle custom widget ── */
.mode-display {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.2rem;
    padding: 0.9rem 1.4rem;
    background: rgba(255,255,255,0.28);
    border: 1.5px solid rgba(60,31,153,0.18);
    border-radius: 18px;
    backdrop-filter: blur(12px);
    margin-bottom: 1.6rem;
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
    animation: fadeUp 0.5s ease 0.2s both;
}

.mode-lang {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.05rem;
    color: #1a1340;
    letter-spacing: -0.01em;
    padding: 0.4rem 1.1rem;
    border-radius: 10px;
    transition: all 0.2s ease;
}

.mode-lang.active {
    background: linear-gradient(135deg, #3c1f99, #5b2d8e);
    color: #fff;
    box-shadow: 0 2px 14px rgba(60,31,153,0.30);
}

.mode-lang.inactive {
    color: rgba(26,19,64,0.40);
}

.mode-arrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: rgba(60,31,153,0.55);
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------
@st.cache_resource
def load_model(path):
    return pickle.load(open(path, "rb"))

@st.cache_resource
def load_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=True)

hands = load_hands()

# ------------------------------------------------
# HERO
# ------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">Computer Vision · Machine Learing</div>
    <h1 class="hero-title">Dexora</h1>
    <p class="hero-sub">Sign Language Translation System</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# MODE SELECT — styled as a pill display
# ------------------------------------------------
mode = st.selectbox(
    "Translation Mode",
    ["ISL → ASL", "ASL → ISL"],
    label_visibility="collapsed"
)

# Visual mode pill display
if mode == "ISL → ASL":
    from_lang, to_lang = "ISL", "ASL"
else:
    from_lang, to_lang = "ASL", "ISL"

st.markdown(f"""
<div class="mode-display">
    <span class="mode-lang active">{from_lang}</span>
    <span class="mode-arrow">──▶</span>
    <span class="mode-lang inactive">{to_lang}</span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# FILE UPLOADER
# ------------------------------------------------
uploaded = st.file_uploader(
    "Upload hand sign image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ------------------------------------------------
# MAIN
# ------------------------------------------------
if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">↑ Input Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    result = hands.process(img)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        row = []
        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])

        # --------------------------
        # BACKEND SWITCH
        # --------------------------
        if mode == "ISL → ASL":
            model = load_model("model_isl.pkl")
            output_folder = "assets/asl"
        else:
            model = load_model("model_asl.pkl")
            output_folder = "assets/isl"

        pred = model.predict([row])[0]
        probs = model.predict_proba([row])[0]
        conf = max(probs) * 100
        conf_int = int(conf)

        with col_b:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">⬡ Detection Result</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-wrapper">
                <div class="letter-badge">{pred}</div>
                <div class="result-meta">
                    <div class="result-label">Predicted Letter</div>
                    <div class="result-letter-name">{pred}</div>
                    <div class="result-label">Model Confidence</div>
                    <div class="conf-row">
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{conf_int}%"></div>
                        </div>
                        <div class="conf-pct">{conf:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # --------------------------
            # FEEDBACK — styled chips via CSS override
            # --------------------------
            st.markdown('<div class="feedback-q">Was this prediction correct?</div>', unsafe_allow_html=True)

            feedback = st.radio(
                "Feedback",
                ["✓  Yes, correct", "✕  No, fix it"],
                horizontal=True,
                label_visibility="collapsed"
            )

            if "fix it" in feedback:
                correct = st.selectbox(
                    "Correct Letter",
                    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                     "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                     "U", "V", "W", "X", "Y", "Z"],
                    label_visibility="visible"
                )

                if st.button("Submit Correction"):
                    folder = f"feedback/{correct}"
                    os.makedirs(folder, exist_ok=True)
                    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
                    image.save(os.path.join(folder, filename))
                    st.success("✓ Correction saved for retraining.")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            out_path = f"{output_folder}/{pred}.png"

            if os.path.exists(out_path):
                st.markdown('<div class="ref-label">◈ Translated Output</div>', unsafe_allow_html=True)
                st.image(Image.open(out_path), use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="no-hand-box">
            <div class="no-hand-icon">🖐</div>
            <div>
                <div class="no-hand-title">No hand detected</div>
                <div class="no-hand-text">
                    Make sure your hand is clearly visible against a plain background.
                    Good lighting and a centred frame improve accuracy significantly.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("""
<div class="footer-strip">
Dexora &nbsp;·&nbsp; Made with ❤︎ by Aahana and Niva &nbsp;·&nbsp; 
</div>
""", unsafe_allow_html=True)