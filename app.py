import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Emotion Classifier", page_icon="😊")

@st.cache_resource
def load_my_model():
    return load_model("emotion_model.keras")

model = load_my_model()

st.title("😊 Emotion Detection System")
st.write("Upload an image and AI will predict Happy or Sad.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    resize = tf.image.resize(img, (256, 256))
    yhat = model.predict(np.expand_dims(resize/255, 0))[0][0]

    confidence = float(yhat)

    if confidence > 0.5:
        st.error(f"😢 Sad (Confidence: {confidence:.2f})")
    else:
        st.success(f"😊 Happy (Confidence: {1-confidence:.2f})")