import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (64, 64)

st.set_page_config(page_title="Skin Disease Classifier", page_icon="🩺")
st.title("Skin Disease Classifier")
st.caption("Upload an image and the CNN will predict the most likely class.")


@st.cache_resource
def load_model_and_classes(model_path="skin.keras", classes_path="classes.json"):
    model = tf.keras.models.load_model(model_path)
    with open(classes_path, "r") as f:
        class_names = json.load(f)
    return model, class_names


try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error(
        "Could not load `skin.keras` / `classes.json`. "
        "Run `python train.py` first and keep both files next to this app.\n\n"
        f"Details: {e}"
    )
    st.stop()


def predict(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    x = tf.keras.utils.img_to_array(img)  # same preprocessing as the notebook
    x = tf.expand_dims(x, 0)              # Rescaling(1/255) is inside the model
    return model.predict(x, verbose=0)[0]


uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Predicting..."):
        probs = predict(image)

    top = int(probs.argmax())
    st.success(f"Prediction: **{class_names[top]}** ({probs[top]:.1%})")

    st.subheader("Top 5 classes")
    top5 = probs.argsort()[::-1][:5]
    df = pd.DataFrame(
        {"Class": [class_names[i] for i in top5], "Probability": probs[top5]}
    ).set_index("Class")
    st.bar_chart(df)

    with st.expander("All probabilities"):
        st.dataframe(
            pd.DataFrame({"Class": class_names, "Probability": probs}).sort_values(
                "Probability", ascending=False
            ),
            hide_index=True,
        )

st.info("For learning/demo purposes only — not a medical diagnosis.")
