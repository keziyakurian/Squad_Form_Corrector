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
    .report-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        border-top: 5px solid #00ff00;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a1a1a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Biomechanical Analytics Dashboard")
st.subheader("Automated Multi-Point Pose Extraction")

# --- INITIALIZATION ---
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
    st.caption("AI Biometrics | Build 1.6.0-Production")

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
            processed_img, reps, state, warning, angle = results
            
            # UI Overlay
            cv2.rectangle(processed_img, (10, 10), (280, 110), (0, 0, 0), -1)
            cv2.putText(processed_img, f"REPETITIONS: {reps}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if angle:
                cv2.putText(processed_img, f"ANGLE: {int(angle)} DEG", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- MAIN DASHBOARD INTERFACE ---
col_stream, col_metrics = st.columns([2, 1])

with col_stream:
    if input_source == "Webcam (Real-Time)":
        st.markdown("### Real-Time Inference Feed")
        webrtc_ctx = webrtc_streamer(
            key="fitness-tracker-v1.6",
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
            
            # Analytics Accumulators
            max_depth = 180.0
            total_reps = 0
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                frame = cv2.resize(frame, (640, 480))
                try:
                    res = engine.process_frame(frame)
                    processed_img, total_reps, state, warning, angle = res
                    if angle and angle < max_depth:
                        max_depth = angle
                    st_frame.image(processed_img, channels="BGR", use_container_width=True)
                except Exception:
                    break
            
            vf.release()
            
            # --- POST-PROCESSING REPORT ---
            st.markdown(f"""
            <div class="report-card">
                <h3 style="margin-top:0;">Final Performance Report</h3>
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <div class="metric-label">Total Repetitions</div>
                        <div class="metric-value">{total_reps}</div>
                    </div>
                    <div>
                        <div class="metric-label">Max Depth Achieved</div>
                        <div class="metric-value">{int(max_depth)}&deg;</div>
                    </div>
                    <div>
                        <div class="metric-label">Form Accuracy</div>
                        <div class="metric-value">{'High' if max_depth < 110 else 'Low'}</div>
                    </div>
                </div>
                <p style="margin-top: 20px; color: #555;">
                    <b>Assessment:</b> {'Excellent depth achieved. Maintaining consistent form throughout the kinetic chain.' if max_depth < 110 else 'Partial depth detected. Increase descent for optimal muscle recruitment.'}
                </p>
            </div>
            """, unsafe_allow_html=True)

with col_metrics:
    st.markdown("### Biometric Telemetry")
    if input_source == "Webcam (Real-Time)" and webrtc_ctx.video_processor:
        st.metric("Total Repetitions", webrtc_ctx.video_processor.estimator.rep_count)
        st.write(f"Inference State: **{webrtc_ctx.video_processor.estimator.state}**")
    elif input_source == "Video Upload (Fallback)" and 'total_reps' in locals():
        st.metric("Video Rep Count", total_reps)
        st.write(f"Deepest Squat: **{int(max_depth)}°**")
    else:
        st.info("System Ready. Connect webcam or upload video to begin.")
    
    st.divider()
    st.markdown("""
    **Core Technologies:**
    - Computer Vision: BlazePose
    - Kinematic Analysis: Joint Angles
    - Signal Logic: EMA Smoothing
    """)
