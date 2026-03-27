import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
from src.pose_service import PoseEstimator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Computer Vision Fitness Analytics",
    layout="wide"
)

# Custom CSS for Professional Branding
st.markdown("""
    <style>
    .main {
        background-color: #fcfcfc;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Intelligent Fitness Tracking System")
st.subheader("Real-Time Biomechanical Analysis")

st.markdown("""
This system utilizes **MediaPipe Pose Estimation** and **Streamlit WebRTC** to provide real-time exercise feedback. 
The architecture is optimized for cloud deployment, ensuring zero local dependency overhead while maintaining sub-100ms inference latency.
""")

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
        processed_img, reps, state, warning = self.estimator.process_frame(img)
        
        # Consistent UI Branding
        brand_color = (0, 0, 0) 
        cv2.rectangle(processed_img, (0, 0), (320, 90), brand_color, -1)
        cv2.putText(processed_img, f"REPETITIONS: {reps}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(processed_img, f"SYSTEM STATE: {state}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if warning:
            cv2.putText(processed_img, warning, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- ANALYTICS DASHBOARD ---
col_stream, col_metrics = st.columns([2, 1])

with col_stream:
    st.markdown("### Primary Video Feed")
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
    st.markdown("### Telemetry & Analytics")
    if webrtc_ctx.video_processor:
        st.metric("Total Repetition Count", webrtc_ctx.video_processor.estimator.rep_count)
        st.write(f"Inference Engine State: **{webrtc_ctx.video_processor.estimator.state}**")
        
        # System Feedback
        if webrtc_ctx.video_processor.estimator.state == "DESCENDING":
            st.info("Descent phase detected. Target depth: < 110 degrees.")
        elif webrtc_ctx.video_processor.estimator.state == "ASCENDING":
            st.success("Target depth achieved. Begin ascent.")
    else:
        st.info("System initializing. Please activate webcam stream.")
    
    st.divider()
    with st.expander("Operational Instructions", expanded=True):
        st.markdown("""
        1. Enable **Camera Permissions** in the secure browser prompt.
        2. Position the subject such that **shoulders, hips, and ankles** are fully visible.
        3. Execute the movement ensuring the knee angle crosses the **110 degree** threshold.
        4. Return to a fully vertical position (**> 160 degrees**) to register a successful rep.
        """)

# --- SYSTEM CONFIGURATION ---
with st.sidebar:
    st.title("System Settings")
    st.write("Inference Parameter Tuning")
    smoothing_factor = st.slider("Signal Smoothing (EMA Alpha)", 0.1, 1.0, 0.4, 
                                 help="Lower alpha indicates higher signal stability at the cost of processing lag.")
    
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.estimator.alpha = smoothing_factor
    
    st.divider()
    st.caption("AI Fitness Tracker Analytics | Build 1.2.2-Production")
