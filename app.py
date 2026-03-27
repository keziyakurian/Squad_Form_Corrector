import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import tempfile
import traceback
from src.pose_service import PoseEstimator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Intelligent Biometric Tracker",
    layout="wide"
)

# Professional Branding
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-right: 5px solid #00ff00;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Biomechanical Analytics Dashboard")
st.subheader("Automated Multi-Point Pose Extraction")

# --- INITIALIZATION ---
# Renamed to force cache refresh on Streamlit Cloud
@st.cache_resource
def get_pose_engine() -> PoseEstimator:
    return PoseEstimator()

# --- SYSTEM SETTINGS ---
with st.sidebar:
    st.header("Control Panel")
    input_source = st.radio("Select Input Source", ["Webcam (Real-Time)", "Video Upload (Fallback)"])
    st.divider()
    ai_enabled = st.checkbox("Enable AI Tracking", value=True)
    smoothing_factor = st.slider("Coordinate Smoothing", 0.1, 1.0, 0.4)
    st.divider()
    st.caption("AI Biometrics | Build 1.5.1-Production")

# --- WEBRTC PROCESSING SERVICE ---
class PoseProcessingService(VideoProcessorBase):
    def __init__(self):
        self.estimator = get_pose_engine()
        self.ai_enabled = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if not self.ai_enabled:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            results = self.estimator.process_frame(img)
            # Safe unpacking with error feedback
            if len(results) != 5:
                raise ValueError(f"Engine out of sync: Expected 5 values, got {len(results)}")
            processed_img, reps, state, warning, angle = results
            
            # UI Overlay
            cv2.rectangle(processed_img, (10, 10), (280, 110), (0, 0, 0), -1)
            cv2.putText(processed_img, f"REPETITIONS: {reps}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if angle:
                cv2.putText(processed_img, f"ANGLE: {int(angle)} DEG", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            # Fallback for display if processing fails
            cv2.putText(img, f"PROCESSING ERROR: {str(e)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- MAIN DASHBOARD INTERFACE ---
col_stream, col_metrics = st.columns([2, 1])

with col_stream:
    if input_source == "Webcam (Real-Time)":
        st.markdown("### Real-Time Inference Feed")
        webrtc_ctx = webrtc_streamer(
            key="fitness-tracker-v1.5.1",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=PoseProcessingService,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": {"width": 640}, "audio": False},
            async_processing=True,
        )
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.ai_enabled = ai_enabled
            webrtc_ctx.video_processor.estimator.alpha = smoothing_factor

    else:
        st.markdown("### Video File Analysis")
        uploaded_file = st.file_uploader("Upload a video of squats (MP4/MOV)", type=["mp4", "mov", "avi"])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            engine = get_pose_engine()
            engine.rep_count = 0 
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                frame = cv2.resize(frame, (640, 480))
                try:
                    res = engine.process_frame(frame)
                    processed_img, reps, state, warning, angle = res
                    st_frame.image(processed_img, channels="BGR", use_container_width=True)
                except Exception as e:
                    st.error(f"Frame Processing Error: {str(e)}")
                    st.text(traceback.format_exc())
                    break
            
            vf.release()
            st.success("Analysis Complete.")

with col_metrics:
    st.markdown("### Biometric Telemetry")
    if input_source == "Webcam (Real-Time)" and webrtc_ctx.video_processor:
        st.metric("Total Repetitions", webrtc_ctx.video_processor.estimator.rep_count)
        st.write(f"Inference State: **{webrtc_ctx.video_processor.estimator.state}**")
    else:
        st.info("Biometric data will appear during active stream analysis.")
    
    st.divider()
    st.markdown("""
    **System Architecture:**
    - AI: Mediapipe BlazePose
    - Latency: Optimized for 30FPS
    - Resilience: EMA Smoothing + FSM
    """)
