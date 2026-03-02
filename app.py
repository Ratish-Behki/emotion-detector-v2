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
.stApp {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 5px;
}

.sub-text {
    text-align: center;
    font-size: 18px;
    color: #475569;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
}

.happy {
    background-color: #dcfce7;
    color: #166534;
}

.sad {
    background-color: #fee2e2;
    color: #991b1b;
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

# ----------- Header ----------- #
st.markdown('<div class="main-title">😊 Emotion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload an image and let AI predict Happy or Sad</div>', unsafe_allow_html=True)

# ----------- Upload Section ----------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read image.")
        st.stop()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Analyzing emotion..."):
        resized = cv2.resize(img_rgb, (256, 256))
        resized = resized / 255.0
        input_img = np.expand_dims(resized, axis=0)

        yhat = model.predict(input_img, verbose=0)[0][0]
        confidence = float(yhat)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if confidence > 0.5:
        st.markdown('<div class="result-box sad">😢 Sad</div>', unsafe_allow_html=True)
        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence*100:.2f}%**")
    else:
        st.markdown('<div class="result-box happy">😊 Happy</div>', unsafe_allow_html=True)
        st.progress(int((1-confidence) * 100))
        st.write(f"Confidence: **{(1-confidence)*100:.2f}%**")

    st.markdown('</div>', unsafe_allow_html=True)