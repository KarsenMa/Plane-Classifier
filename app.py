import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

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

    top1_id = int(probs.data.argmax())
    confidence = float(probs.data[top1_id])
    label = CLASS_NAMES[top1_id]

    return label, confidence, image

# Function to classify video frames and collect them
def classify_video_frames(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open input video.")
        return []

    frame_count = 0
    labeled_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            label, confidence, _ = classify_image(image)

            # Draw label on frame
            cv2.putText(frame_rgb, f"{label} ({confidence*100:.1f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert back to PIL image for display
            frame_pil = Image.fromarray(frame_rgb)
            labeled_frames.append((frame_pil, label, confidence))

        frame_count += 1

    cap.release()
    return labeled_frames

# Streamlit UI
st.title("✈️ Plane Classifier")
st.write("Upload an image or a video of a plane!")

# Frame skip control in sidebar
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
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        with st.spinner('Processing video frames...'):
            labeled_frames = classify_video_frames(tfile.name, frame_skip=frame_skip)

        if labeled_frames:
            st.success(f"✅ Processed {len(labeled_frames)} frames!")

            frame_index = st.slider('Slide through frames:', 0, len(labeled_frames)-1, 0)
            frame_image, frame_label, frame_confidence = labeled_frames[frame_index]

            st.image(frame_image, caption=f"{frame_label} ({frame_confidence*100:.1f}%)", use_container_width=True)
        else:
            st.error("No frames processed.")
