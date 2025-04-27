import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os

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

# Function to classify video frames
def classify_video(video_path, frame_skip=5):
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    processed_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            # Only classify and draw every `frame_skip` frames
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            label, confidence, _ = classify_image(image)

            # Draw label on frame
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            processed_count += 1

        # Write EVERY frame (with or without label)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return output_path, processed_count

# Streamlit UI
st.title("✈️ Plane Classifier")
st.write("Upload an image or a video of a plane!")

# Frame skip selector (optional in sidebar)
frame_skip = st.sidebar.slider("Frame Skip (Video)", 1, 30, 5)

uploaded_file = st.file_uploader(
    "Choose a file...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file:
    file_type = uploaded_file.type

    if file_type.startswith('image'):
        image = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner('Classifying image...'):
            label, confidence, output_image = classify_image(image)

        st.image(output_image, caption="Uploaded Image", use_container_width=True)

        if label == "No plane detected":
            st.warning("⚠️ No plane detected in the uploaded image.")
        else:
            st.markdown(f"### ✈️ Prediction: **{label}**")
            st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

    elif file_type.startswith('video'):
        # Save the uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.video(tfile.name)

        with st.spinner('Processing video...'):
            processed_video_path, total_processed = classify_video(tfile.name, frame_skip=frame_skip)

        st.success(f"✅ Processed {total_processed} labeled frames out of total video frames!")

        st.video(processed_video_path)
