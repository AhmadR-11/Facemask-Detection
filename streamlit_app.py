import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Function to load model
@st.cache_resource
def load_mask_model():
    model = load_model("mask_detector.keras")
    return model

# Function to detect and predict
def detect_and_predict_mask(image, model):
    # Convert to array
    image = np.array(image.convert('RGB'))
    orig = image.copy()
    (h, w) = image.shape[:2]

    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Streamlit/PIL uses RGB
    
    # Try multiple cascades and parameters to improve detection (Fallback logic)
    cascades_to_try = [
        ("haarcascade_frontalface_alt2.xml", 1.1, 5),   # Standard, high precision
        ("haarcascade_frontalface_default.xml", 1.1, 5), # Standard, higher recall
        ("haarcascade_frontalface_alt.xml", 1.1, 5),     # Alternative
        ("haarcascade_frontalface_default.xml", 1.05, 3) # Relaxed: More sensitive, lower neighbors
    ]

    faces = []
    for cascade_name, scale, neighbors in cascades_to_try:
        cascade_path = cv2.data.haarcascades + cascade_name
        if cascade_path:
            faceNet = cv2.CascadeClassifier(cascade_path)
            detected = faceNet.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(60, 60))
            if len(detected) > 0:
                faces = detected
                # Found faces, break early
                break


    results = []

    for (x, y, w_box, h_box) in faces:
        # Extract face
        face = image[y:y + h_box, x:x + w_box]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Predict
        (mask, withoutMask) = model.predict(face)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        score = max(mask, withoutMask) * 100
        color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

        results.append((x, y, w_box, h_box, label, score, color))
    
    return results, image

# Streamlit UI
st.set_page_config(page_title="MaskGuard AI", page_icon="🛡️", layout="wide")

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Main Background */
        .stApp {
            background-color: #0E1117; 
            color: #FAFAFA;
        }
        
        /* Metric Card Styling Override */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            color: #FAFAFA !important;
        }
    </style>
""", unsafe_allow_html=True)

# Import Streamlit Extras
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.badges import badge

# Header Section
colored_header(
    label="🛡️ MaskGuard AI",
    description="Advanced Face Mask Detection System | Powered by TensorFlow & OpenCV",
    color_name="blue-70"
)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209965.png", width=80)
    st.header("⚙️ Settings")
    
    # Social Badges
    st.markdown("### Connect")
    badge(type="github", name="streamlit/streamlit", url="https://github.com/streamlit/streamlit")
    
    st.markdown("---")
    st.markdown("**System Status:** 🟢 Online")
    st.caption("v1.0.0 Release")

# Load Model
try:
    with st.spinner("Initializing AI Engine..."):
        model = load_mask_model()
except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# Main Content Area
uploaded_file = st.file_uploader("📂 Upload an image to analyze", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Create two columns for layout
    col_img, col_res = st.columns([1, 1], gap="large")
    
    with col_img:
        st.markdown("### 📸 Image Preview")
        st.image(image, use_container_width=True)
        analyze_btn = st.button("🔍 Run Analysis", use_container_width=True)

    if analyze_btn:
        with col_res:
            st.markdown("### 📊 Analysis Results")
            with st.spinner("Scanning faces..."):
                results, output_image = detect_and_predict_mask(image, model)
                
                # Show processed image
                st.image(output_image, caption='Processed Output', use_container_width=True)
                
                if len(results) == 0:
                    st.warning("⚠️ No faces detected.")
                else:
                    # Calculate Stats
                    mask_count = sum(1 for r in results if r[4] == "Mask")
                    no_mask_count = len(results) - mask_count
                    compliance_rate = (mask_count / len(results)) * 100
                    
                    # Display Metrics using Streamlit Extras
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Faces Detected", len(results))
                    m2.metric("Masks On", mask_count, delta=f"{mask_count} Safe")
                    m3.metric("No Mask", no_mask_count, delta=f"-{no_mask_count} Risk", delta_color="inverse")
                    
                    style_metric_cards(background_color="#1E1E1E", border_left_color="#4F8BF9", border_color="#2D2D2D", box_shadow=True)
                    
                    st.divider()
                    st.markdown("#### Detailed Breakdown:")
                    
                    for i, (x, y, w_box, h_box, label, score, color) in enumerate(results):
                        confidence = f"{score:.2f}%"
                        if label == "Mask":
                            st.success(f"**Face {i+1}:** Mask Detected 😷 (Confidence: {confidence})")
                        else:
                            st.error(f"**Face {i+1}:** NO MASK ⛔ (Confidence: {confidence})")
