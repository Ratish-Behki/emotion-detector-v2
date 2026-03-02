import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ----------------------------
# Download Model from Drive
# ----------------------------
MODEL_PATH = "emotion_model.keras"
FILE_ID = "1mpzEmGftosz1Bi_OpIP1e8-XhQjiM1hw"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait."):
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# ----------------------------
# Load Model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ----------------------------
# UI
# ----------------------------
st.title("😊 Emotion Detection System")
st.write("Upload an image and AI will predict emotion.")

uploaded_file = st.file_uploader(
    "Choose an image", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = image.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.success("😊 Happy")
    else:
        st.error("😢 Sad")