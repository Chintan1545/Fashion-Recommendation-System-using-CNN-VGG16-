import os
import glob
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine


# CONFIG

st.set_page_config(page_title="Fashion Recommendation", layout="wide")

IMAGE_DIR = r"E:\projects\Machine Learning Project\Supervised Learning Projects\women fashion"


# LOAD MODEL (CACHE)

@st.cache_resource
def load_model():
    base_model = VGG16(weights="imagenet", include_top=False)
    return Model(inputs=base_model.input, outputs=base_model.output)

model = load_model()


# IMAGE UTILITIES

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def extract_features(img):
    processed = preprocess_image(img)
    features = model.predict(processed, verbose=0)
    features = features.flatten()
    return features / np.linalg.norm(features)



# LOAD DATASET FEATURES (CACHE)

@st.cache_data
def load_dataset():
    image_paths = [
        img for img in glob.glob(os.path.join(IMAGE_DIR, "*"))
        if img.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))
    ]

    features = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        feat = extract_features(img)
        features.append(feat)

    return image_paths, np.array(features)

image_paths, dataset_features = load_dataset()


# STREAMLIT UI

st.title("👗 Fashion Recommendation System")
st.write("Upload a fashion image and get similar recommendations")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg", "webp"]
)

top_n = st.slider("Number of recommendations", 2, 6, 4)

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded Image", width=300)

    with st.spinner("Finding similar fashion items..."):
        input_features = extract_features(input_image)

        similarities = [
            1 - cosine(input_features, feat)
            for feat in dataset_features
        ]

        top_indices = np.argsort(similarities)[::-1][:top_n]

    st.subheader("✨ Recommended Items")

    cols = st.columns(top_n)
    for col, idx in zip(cols, top_indices):
        col.image(Image.open(image_paths[idx]), use_column_width=True)

