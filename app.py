import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Disease Classifier", page_icon="🔬", layout="centered")

MODEL_PATH = Path("skin.keras")
CLASSES_PATH = Path("classes.json")
IMAGE_SIZE = (64, 64)


@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "skin.keras was not found. Train the model first using train_model.py."
        )
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            "classes.json was not found. Train the model first using train_model.py."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with CLASSES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    return model, class_names


def predict_image(image: Image.Image, model, class_names):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    if predicted_index >= len(class_names):
        raise ValueError("The model output does not match classes.json.")

    return class_names[predicted_index], float(probabilities[predicted_index]), probabilities


st.title("🔬 Skin Disease Image Classifier")
st.write(
    "Upload a skin image to see the class predicted by the CNN model "
    "trained from your dataset."
)

st.warning(
    "Educational project only: this model is not a medical diagnosis. "
    "It may make incorrect predictions. Consult a qualified dermatologist "
    "for symptoms, diagnosis, or treatment."
)

try:
    model, class_names = load_model_and_classes()
except Exception as error:
    st.error(str(error))
    st.info(
        "Keep skin.keras and classes.json in the same folder as app.py. "
        "If you have not trained the model yet, run train_model.py first."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a skin image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Classify image", type="primary"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, probabilities = predict_image(
                    image, model, class_names
                )

            st.subheader("Prediction")
            st.success(f"Predicted class: **{predicted_class}**")
            st.metric("Model confidence", f"{confidence * 100:.2f}%")

            st.subheader("Class probabilities")
            probability_data = {
                class_names[i]: float(probabilities[i])
                for i in range(min(len(class_names), len(probabilities)))
            }
            probability_data = dict(
                sorted(probability_data.items(), key=lambda item: item[1], reverse=True)
            )
            st.bar_chart(probability_data)
    except Exception as error:
        st.error(f"Could not process this image: {error}")
