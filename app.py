import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🛣 Road Pothole Detection System")

st.write("Upload a road image to detect potholes.")

# Load pothole detection model
model = YOLO("pothole.pt")

uploaded_file = st.file_uploader("Upload Road Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Road Image", use_column_width=True)

    st.write("Analyzing road surface...")

    results = model(img, conf=0.4)

    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detected Potholes", channels="BGR")

    potholes = len(results[0].boxes)

    st.subheader(f"Potholes Detected: {potholes}")

    if potholes == 0:
        st.success("Road Condition: GOOD")
    elif potholes <= 3:
        st.warning("Road Condition: MODERATE DAMAGE")
    else:
        st.error("Road Condition: SEVERE DAMAGE")
