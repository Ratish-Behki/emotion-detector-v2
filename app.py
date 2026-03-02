import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

# ----------- Custom Styling ----------- #
st.markdown("""
    <style>
    body {
        background-color: #0f172a;
    }
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        background: linear-gradient(90deg, #facc15, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-text {
        text-align: center;
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- Load Model ----------- #
@st.cache_resource
def load_my_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "emotion_model.h5")
    return tf.keras.models.load_model(model_path, compile=False)

model = load_my_model()

# ----------- UI ----------- #
st.markdown('<div class="main-title">😊 Emotion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload an image and AI will predict Happy or Sad</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read image.")
        st.stop()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing emotion..."):
        resized = cv2.resize(img_rgb, (256, 256))
        resized = resized / 255.0
        input_img = np.expand_dims(resized, axis=0)

        yhat = model.predict(input_img, verbose=0)[0][0]
        confidence = float(yhat)

    st.markdown("### 🔍 Prediction Result")

    if confidence > 0.5:
        st.error("😢 Sad")
        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence*100:.2f}%**")
    else:
        st.success("😊 Happy")
        st.progress(int((1-confidence) * 100))
        st.write(f"Confidence: **{(1-confidence)*100:.2f}%**")