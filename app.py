import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Configure Streamlit page
st.set_page_config(page_title="Plane Classifier",
                   page_icon="✈️", layout="centered")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Make sure 'best.pt' is uploaded
    return model

model = load_model()

# Class names
CLASS_NAMES = model.names

# Function to classify image
def classify_image(image):
    results = model.predict(image)

    probs = results[0].probs  # classification probabilities

    if probs is None:
        return "No plane detected", 0.0, image

    # Get the class with the highest probability
    top1_id = int(probs.data.argmax())
    confidence = float(probs.data[top1_id])
    label = CLASS_NAMES[top1_id]

    return label, confidence, image

# Streamlit UI
st.title("✈️ Plane Classifier")
st.write("Upload an image of a plane and see what model it is!")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner('Classifying...'):
        label, confidence, output_image = classify_image(image)

    st.image(output_image, caption="Uploaded Image", use_container_width=True)

    if label == "No plane detected":
        st.warning("⚠️ No plane detected in the uploaded image.")
    else:
        st.markdown(f"### ✈️ Prediction: **{label}**")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")


