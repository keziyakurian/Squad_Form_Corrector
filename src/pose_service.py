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
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if mp is None or mp_pose is None:
            raise ImportError(f"CRITICAL: MediaPipe could not be initialized.\nDetails: {import_error_details}")
            
        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, 
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Premium Drawing Specifications
        self.landmark_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        
        # State Machine Configuration
        self.state: str = "WAITING"
        self.rep_count: int = 0
        self.smooth_knee_angle: Optional[float] = None
        self.alpha: float = 0.4
        self.last_state_change_time: float = time.time()
        self.timeout_threshold_seconds: float = 5.0
        self.coordinate_buffer: Deque[List[float]] = deque(maxlen=30)

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

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, str, Optional[str], Optional[float]]:
        self._apply_timeout_logic()
        
        # Optimization: Process at lower resolution if needed? No, let's keep it standard first
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        warning_msg: Optional[str] = None
        current_angle: Optional[float] = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract main tracking point (Left Hip for visibility check)
            hip_l = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            knee_l = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle_l = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            if hip_l.visibility < 0.5 or knee_l.visibility < 0.5 or ankle_l.visibility < 0.5:
                warning_msg = "PARTIAL OCCLUSION: ENSURE KNEES AND HIPS ARE VISIBLE"
            else:
                # Calculation Logic
                a = [hip_l.x, hip_l.y]
                b = [knee_l.x, knee_l.y]
                c = [ankle_l.x, ankle_l.y]
                
                current_angle = self.calculate_angle(a, b, c)
                if self.smooth_knee_angle is None: self.smooth_knee_angle = current_angle
                else: self.smooth_knee_angle = (current_angle * self.alpha) + (self.smooth_knee_angle * (1 - self.alpha))
                
                # FSM Transitions
                if self.smooth_knee_angle > 160:
                    if self.state == "ASCENDING": 
                        self.rep_count += 1
                        self.last_state_change_time = time.time() # Reset on successful completion
                    self.state = "WAITING"
                elif self.smooth_knee_angle < 150 and self.state == "WAITING":
                    self.state = "DESCENDING"
                    self.last_state_change_time = time.time()
                elif self.smooth_knee_angle < 110 and self.state == "DESCENDING":
                    self.state = "ASCENDING"
                    self.last_state_change_time = time.time()

            # DRAW PREMIUM SKELETON
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.landmark_spec,
                connection_drawing_spec=self.connection_spec
            )
            
        return image, self.rep_count, self.state, warning_msg, self.smooth_knee_angle
