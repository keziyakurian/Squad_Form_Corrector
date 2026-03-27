import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import time
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
    .st-status {
        font-size: 0.8rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Biomechanical Analytics Dashboard")
st.subheader("Automated Multi-Point Pose Extraction")

# --- INITIALIZATION ---
@st.cache_resource
def get_estimator() -> PoseEstimator:
    return PoseEstimator()

# --- SYSTEM SETTINGS ---
with st.sidebar:
    st.header("Control Panel")
    ai_enabled = st.checkbox("Enable AI Tracking", value=True)
    st.divider()
    
    with st.expander("Advanced Connectivity", expanded=False):
        st.info("If the video is stuck, try refreshing the page or switching networks (e.g., to a mobile hotspot).")
        st.write("Current ICE Strategy: **STUN (Multipath)**")
    
    st.divider()
    st.caption("AI Biometrics | Build 1.4.0-Production")

class PoseProcessingService(VideoProcessorBase):
    def __init__(self):
        self.estimator = get_estimator()
        self.ai_enabled = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        if not self.ai_enabled:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Analyze frame
        processed_img, reps, state, warning, angle = self.estimator.process_frame(img)
        
        # Technical Overlay
        cv2.rectangle(processed_img, (10, 10), (280, 110), (0, 0, 0), -1)
        cv2.putText(processed_img, f"REPETITIONS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(processed_img, f"{reps}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        if angle:
            cv2.putText(processed_img, f"ANGLE: {int(angle)} DEG", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        cv2.putText(processed_img, f"STATUS: {state}", (10, processed_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- DASHBOARD LAYOUT ---
col_stream, col_metrics = st.columns([2, 1])

with col_stream:
    st.markdown("### Real-Time Inference Feed")
    
    # MAX COMPATIBILITY WebRTC Configuration
    webrtc_ctx = webrtc_streamer(
        key="fitness-tracker-stream-v1",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PoseProcessingService,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ],
            "iceTransportPolicy": "all",
        },
        # Low-latency constraints
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 15}, 
            "audio": False
        },
        async_processing=True,
    )
    
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.ai_enabled = ai_enabled

with col_metrics:
    st.markdown("### Biometric Telemetry")
    if webrtc_ctx.state.playing:
        st.metric("Total Repetitions", webrtc_ctx.video_processor.estimator.rep_count if webrtc_ctx.video_processor else 0)
        st.write(f"System State: **{webrtc_ctx.video_processor.estimator.state if webrtc_ctx.video_processor else 'INITIALIZING'}**")
    else:
        st.info("System Ready. Please click **START** above and allow camera access.")
    
    st.divider()
    with st.expander("Connection Troubleshooter", expanded=True):
        if not webrtc_ctx.state.playing:
            st.warning("⚠️ If the feed is stuck at 'Awaiting' or showing a white screen:")
            st.markdown("""
            1. **Verify Permissions**: Look for the camera icon in your browser URL bar.
            2. **Network Block**: Your current network (e.g., Office/Public Wifi) may be blocking WebRTC. Try switching to a **Mobile Hotspot**.
            3. **Browser**: Ensure you are using a modern browser (Chrome, Edge, or Safari).
            4. **Hard Refresh**: Press `Cmd + Shift + R` to clear the cache.
            """)
        else:
            st.success("✅ Connection Established. Tracking Active.")
