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

# --- SYSTEM SETTINGS ---
with st.sidebar:
    st.header("Control Panel")
    ai_enabled = st.checkbox("Enable AI Tracking", value=True, help="Toggle this off if the video is stuck.")
    smoothing_factor = st.slider("Coordinate Smoothing", 0.1, 1.0, 0.4)
    st.divider()
    st.caption("AI Biometrics | Build 1.3.1-Production")

# --- SERVICE INITIALIZATION ---
@st.cache_resource
def get_estimator() -> PoseEstimator:
    return PoseEstimator()

class PoseProcessingService(VideoProcessorBase):
    def __init__(self):
        self.estimator = get_estimator()
        self.ai_enabled = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Bypass AI if disabled (to test raw connectivity)
        if not self.ai_enabled:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Analyze frame via the pose estimation service
        processed_img, reps, state, warning, angle = self.estimator.process_frame(img)
        
        # Technical Overlay
        cv2.rectangle(processed_img, (10, 10), (250, 110), (0, 0, 0), -1)
        cv2.putText(processed_img, f"REPETITIONS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(processed_img, f"{reps}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        if angle:
            cv2.putText(processed_img, f"ANGLE: {int(angle)} DEG", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        cv2.putText(processed_img, f"STATUS: {state}", (10, processed_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if warning:
            cv2.putText(processed_img, warning, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- DASHBOARD LAYOUT ---
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
        # Optimization: Lower resolution (480p) to ensure connectivity on slow networks
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, 
            "audio": False
        },
        async_processing=True,
    )
    
    # Sync AI toggle to the processor
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.ai_enabled = ai_enabled
        webrtc_ctx.video_processor.estimator.alpha = smoothing_factor

with col_metrics:
    st.markdown("### Biometric Telemetry")
    if webrtc_ctx.video_processor:
        st.metric("Total Repetitions", webrtc_ctx.video_processor.estimator.rep_count)
        st.write(f"Inference State: **{webrtc_ctx.video_processor.estimator.state}**")
        
        if webrtc_ctx.video_processor.estimator.smooth_knee_angle:
            angle = int(webrtc_ctx.video_processor.estimator.smooth_knee_angle)
            st.write(f"Knee Angle: `{angle}°`")
            st.progress(max(0, min(100, (180 - angle) / (180 - 90) * 100)) / 100)
    else:
        st.info("Awaiting video stream initialization...")
    
    st.divider()
    with st.expander("Connection Troubleshooter"):
        st.warning("Stuck at 'Awaiting' message?")
        st.markdown("""
        1. **Refresh** the page and click **START**.
        2. Ensure **Camera Permission** is granted (check browser URL bar).
        3. Toggle **Enable AI Tracking** OFF in the sidebar to test raw connection.
        4. If still failing, your network (firewall) may be blocking the WebRTC stream.
        """)
