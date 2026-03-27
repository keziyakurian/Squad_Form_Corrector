import cv2
import numpy as np
import time
import sys
import traceback
from typing import Tuple, Optional, List, Deque
from collections import deque

# HYPER-VERBOSE DEFENSIVE IMPORT FOR MEDIAPIPE
mp = None
mp_pose = None
import_error_details = ""

try:
    import mediapipe as mp
    try:
        from mediapipe.python.solutions import pose as mp_pose
    except Exception as e:
        import_error_details += f"Sub-import (pose) failed: {str(e)}. "
        try:
            import mediapipe.solutions.pose as mp_pose
        except Exception as e2:
            import_error_details += f"Alternative sub-import failed: {str(e2)}. "
except Exception as e:
    import_error_details += f"Primary import failed: {str(e)}. Traceback: {traceback.format_exc()}"
    mp = None
    mp_pose = None

class PoseEstimator:
    """
    High-performance pose estimation service for automated exercise tracking.
    """
    
    def __init__(self, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.7):
        if mp is None or mp_pose is None:
            # We raise a very loud error with the full detail to catch it in the Streamlit logs
            raise ImportError(f"CRITICAL: MediaPipe could not be initialized.\nDetails: {import_error_details}")
            
        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # State Machine Configuration
        self.state: str = "WAITING"
        self.rep_count: int = 0
        self.smooth_knee_angle: Optional[float] = None
        self.alpha: float = 0.4
        self.last_state_change_time: float = time.time()
        self.timeout_threshold_seconds: float = 5.0
        self.coordinate_buffer: Deque[List[float]] = deque(maxlen=30)
        self.form_voting_buffer: Deque[str] = deque(maxlen=5)

    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        return float(angle)

    def _apply_timeout_logic(self):
        if self.state != "WAITING" and time.time() - self.last_state_change_time > self.timeout_threshold_seconds:
            self.state = "WAITING"
            self.last_state_change_time = time.time()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, str, Optional[str]]:
        self._apply_timeout_logic()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        warning_msg: Optional[str] = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            ankle_landmark = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            if ankle_landmark.visibility < 0.7:
                warning_msg = "VISIBILITY ALERT: ENSURE FULL BODY IS IN FRAME"
            else:
                hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                raw_knee_angle = self.calculate_angle(hip, knee, ankle)
                if self.smooth_knee_angle is None: self.smooth_knee_angle = raw_knee_angle
                else: self.smooth_knee_angle = (raw_knee_angle * self.alpha) + (self.smooth_knee_angle * (1 - self.alpha))
                
                if self.smooth_knee_angle > 160:
                    if self.state == "ASCENDING": self.rep_count += 1
                    self.state = "WAITING"
                elif self.smooth_knee_angle < 150 and self.state == "WAITING": self.state = "DESCENDING"
                elif self.smooth_knee_angle < 110 and self.state == "DESCENDING": self.state = "ASCENDING"

            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
        return image, self.rep_count, self.state, warning_msg
