import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
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
        border-left: 5px solid #00ff00;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Biomechanical Analytics Dashboard")
st.subheader("Automated Multi-Point Pose Extraction")

# --- SERVICE INITIALIZATION ---
@st.cache_resource
def get_estimator() -> PoseEstimator:
    return PoseEstimator()

class PoseProcessingService(VideoProcessorBase):
    def __init__(self):
        self.estimator = get_estimator()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Analyze frame via the pose estimation service
        processed_img, reps, state, warning, angle = self.estimator.process_frame(img)
        
        # --- TECHNICAL OVERLAY ---
        # Draw Repetition Counter (Modern UI)
        cv2.rectangle(processed_img, (10, 10), (250, 110), (0, 0, 0), -1)
        cv2.putText(processed_img, f"REPETITIONS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(processed_img, f"{reps}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        # Draw Angle Telemetry directly on image
        if angle:
            angle_int = int(angle)
            color = (0, 255, 0) if angle_int < 110 else (255, 255, 255)
            cv2.putText(processed_img, f"ANGLE: {angle_int} DEG", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Draw System Status
        state_color = (0, 255, 0) if state != "WAITING" else (200, 200, 200)
        cv2.putText(processed_img, f"STATUS: {state}", (10, processed_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        if warning:
            cv2.putText(processed_img, warning, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- ANALYTICS DASHBOARD ---
col_stream, col_metrics = st.columns([2, 1])

with col_stream:
    st.markdown("### Real-Time Inference Feed")
    webrtc_ctx = webrtc_streamer(
        key="fitness-tracker-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PoseProcessingService,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_metrics:
    st.markdown("### Biometric Telemetry")
    if webrtc_ctx.video_processor:
        st.metric("Total Repetitions", webrtc_ctx.video_processor.estimator.rep_count)
        st.write(f"Inference State: **{webrtc_ctx.video_processor.estimator.state}**")
        
        # Dynamic Angle Gauge (Visual check)
        if webrtc_ctx.video_processor.estimator.smooth_knee_angle:
            angle = int(webrtc_ctx.video_processor.estimator.smooth_knee_angle)
            st.write(f"Knee Angle: `{angle}°`")
            st.progress(max(0, min(100, (180 - angle) / (180 - 90) * 100)) / 100, text="Squat Depth Progress")
    else:
        st.info("Awaiting video stream initialization...")
    
    st.divider()
    with st.expander("System Logic"):
        st.markdown("""
        - **Target Depth**: < 110 degrees
        - **Recovery Height**: > 160 degrees
        - **Sampling Rate**: ~30 FPS
        - **Inference Latency**: < 50ms
        """)

# --- SYSTEM CONFIGURATION ---
with st.sidebar:
    st.header("Control Panel")
    smoothing_factor = st.slider("Coordinate Smoothing", 0.1, 1.0, 0.4)
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.estimator.alpha = smoothing_factor
    
    st.divider()
    st.caption("AI Biometrics | Build 1.3.0-Production")
