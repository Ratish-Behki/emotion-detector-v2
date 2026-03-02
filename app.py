import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

st.set_page_config(page_title="Emotion Classifier", page_icon="😊")

# ------------------ Load Model Safely ------------------ #
@st.cache_resource
def load_my_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "emotion_model.h5")  # ✅ changed to .h5

    if not os.path.exists(model_path):
        st.error("Model file not found!")
        st.stop()

    return tf.keras.models.load_model(model_path, compile=False)

model = load_my_model()

# ------------------ UI ------------------ #
st.title("😊 Emotion Detection System")
st.write("Upload an image and AI will predict Happy or Sad.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read the image.")
        st.stop()

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # Resize image
    resized = cv2.resize(img_rgb, (256, 256))
    resized = resized / 255.0
    input_img = np.expand_dims(resized, axis=0)

    # Prediction
    yhat = model.predict(input_img, verbose=0)[0][0]
    confidence = float(yhat)

    if confidence > 0.5:
        st.error(f"😢 Sad (Confidence: {confidence:.2f})")
    else:
        st.success(f"😊 Happy (Confidence: {1-confidence:.2f})")