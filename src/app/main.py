import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from config import MODEL_PATH, IMG_HEIGHT, IMG_WIDTH

model = load_model(MODEL_PATH)

st.title("Pneumonia Detection from Chest X-Ray")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type="jpg")

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error(f"Pneumonia Detected with Confidence: {prediction:.2f}")
    else:
        st.success(f"No Pneumonia Detected with Confidence: {1 - prediction:.2f}")
