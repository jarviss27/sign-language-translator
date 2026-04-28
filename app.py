import streamlit as st
import cv2
import pickle
import mediapipe as mp
import json
import numpy as np
from PIL import Image

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load mapping
with open("mapping.json") as f:
    mapping = json.load(f)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

st.title("AI Based Sign Language Translator")

uploaded = st.file_uploader("Upload Hand Sign Image", type=["jpg","png","jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image", width=300)

    img = np.array(image)

    result = hands.process(img)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        row = []

        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])

        pred = model.predict([row])[0]

        probs = model.predict_proba([row])[0]
        conf = max(probs) * 100

        st.success(f"Predicted Letter: {pred}")
        st.info(f"Confidence: {conf:.2f}%")

        asl_path = mapping[pred]
        asl_img = Image.open(asl_path)
        st.image(asl_img, width=250)

    else:
        st.error("No Hand Detected")