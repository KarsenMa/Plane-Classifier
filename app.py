import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw

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

# Function to classify image and draw bounding box
def classify_image(image):
    results = model.predict(image)
    
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        # Make sure you return three things: label, confidence, image
        return "No plane detected", 0.0, image

    cls_id = int(boxes.cls[0].item())
    confidence = float(boxes.conf[0].item())
    label = CLASS_NAMES[cls_id]

    # Draw bounding box
    box = boxes.xyxy[0].tolist()  # (x1, y1, x2, y2)
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red", width=5)
    draw.text((box[0], box[1] - 10), f"{label} {confidence*100:.1f}%", fill="red")

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

    st.image(output_image, caption="Detected Plane", use_container_width=True)

    if label == "No plane detected":
        st.warning("⚠️ No plane detected in the uploaded image.")
    else:
        st.markdown(f"### ✈️ Prediction: **{label}**")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

