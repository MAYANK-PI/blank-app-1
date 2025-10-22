import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Title
st.title("🧠 Image Segmentation + Caption Generation App")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # Load Models (U-Net + CNN-LSTM)
    # -------------------------------
    @st.cache_resource
    def load_models():
        unet_model = load_model("models/unet_model.h5", compile=False)
        caption_model = load_model("models/caption_model.h5", compile=False)
        tokenizer = np.load("models/tokenizer.npy", allow_pickle=True).item()
        return unet_model, caption_model, tokenizer

    unet_model, caption_model, tokenizer = load_models()

    # -------------------------------
    # 1️⃣ Image Segmentation (U-Net)
    # -------------------------------
    img_array = np.array(img.resize((128, 128))) / 255.0
    mask_pred = unet_model.predict(np.expand_dims(img_array, axis=0))[0]
    st.image(mask_pred, caption="Predicted Segmentation Mask", use_column_width=True)

    # -------------------------------
    # 2️⃣ Image Captioning (CNN + LSTM)
    # -------------------------------
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
    encoder = InceptionV3(weights="imagenet")
    encoder = tf.keras.Model(encoder.input, encoder.layers[-2].output)

    x = kimage.img_to_array(img.resize((299, 299)))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    photo = encoder.predict(x)

    max_length = 30  # adjust to your tokenizer
    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = caption_model.predict([photo, seq], verbose=0)
        word = tokenizer.index_word.get(np.argmax(yhat))
        if word is None or word == "endseq":
            break
        in_text += " " + word

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    st.subheader("📝 Generated Caption:")
    st.write(caption)
