import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.vision_engine import VisionEngine

# Config
st.set_page_config(page_title="Vision Intelligence Pro", layout="wide", page_icon="👁️")
st.title("👁️ Vision Intelligence Pro Dashboard")
st.write("Advanced Computer Vision Analysis Suite")

# Initialize Engine
@st.cache_resource
def load_engine():
    return VisionEngine()

engine = load_engine()

# Sidebar
st.sidebar.header("🔧 Settings")
mode = st.sidebar.selectbox("Analysis Mode", ["Object Detection", "Pose Estimation", "Facial Intelligence"])
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4)

# Media Source
source = st.sidebar.radio("Input Source", ["Static Image", "Webcam/Video Stream"])

if source == "Static Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        
        # Process
        st.write("🔍 Processing...")
        if mode == "Object Detection":
            result = engine.detect_objects(image, conf=conf_thresh)
        elif mode == "Pose Estimation":
            result = engine.process_pose(image.copy())
        else:
            result = engine.process_faces(image.copy())
            
        st.image(result, caption=f"Result: {mode}", use_container_width=True)

else:
    st.info("💡 Run locally to use real-time webcam analysis.")
    st.write("Current analysis profile: ", mode)
    # Note: Full real-time webcam support via Streamlit requires additional setup (WebRTC).
    # This serves as a conceptual UI for professional portfolios.
