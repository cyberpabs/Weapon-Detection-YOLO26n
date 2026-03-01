import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av # VIDEO FRAME LIBRARY ??

st.set_page_config(page_title="Weapon Detection AI", layout="wide")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. model selection
    option = st.selectbox(
        "Choose a model:",
        ("best", "best_tuned"),
        help="Choose 'best' for the base model or 'best_tuned' for the Optuna optimized version."
    )
    
    # 2. Slider coonfidence
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.05)
    
    st.info(f"Currently using: YOLO26n ({option})")

# --- MODEL LOAD ---
@st.cache_resource
def load_model(model_name):
    return YOLO(f"models/{model_name}.pt")

model = load_model(option)

st.title("Weapon Detection System")

# Tabs for the webpage
tab1, tab2, tab3 = st.tabs(["Images", "Video", "Webcam"])

# --- TAB 1: IMAGES (MULTIPLE) ---
with tab1:
    uploaded_files = st.file_uploader("Upload images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if uploaded_files:
        num_cols = 3
        cols = st.columns(num_cols)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % num_cols]:
                img = Image.open(file)
                results = model(img, conf=conf_threshold)
                res_plotted = results[0].plot()
                # BGR to RGB for Streamlit
                res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.image(res_plotted, caption=file.name, use_container_width=True)

# --- TAB 2: VIDEO ---
with tab2:
    uploaded_video = st.file_uploader("Upload video...", type=['mp4', 'avi', 'mov', 'mpeg'])
    
    if uploaded_video:
        # Streamlit needs to store the vid temporally in order to read it with OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty() # Placeholder updates frames
        
        stop_btn = st.button("Stop processing")
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Actual frame inference
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            
            # Draw results
            res_plotted = results[0].plot()
            res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Update web frame
            st_frame.image(res_plotted, channels="RGB", use_container_width=True)
            
        cap.release()
        st.success("Video processing stopped.")

# --- TAB 3: WEBCAM IN REAL TIME ---
with tab3:
    st.header("Webcam Live Detection")
    
    class VideoProcessor:
        def __init__(self, model, conf):
            self.model = model
            self.conf = conf

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Utilitza el model i la conf que hem passat a la classe
            results = self.model.predict(img, conf=self.conf, verbose=False)
            annotated_frame = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="weapon-detection",
        # Passem el model i la confiança actuals de la sidebar a la classe
        video_processor_factory=lambda: VideoProcessor(model, conf_threshold),
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )