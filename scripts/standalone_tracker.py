import cv2
import sys
import os

# Ensure the parent directory is in the path for modular imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pose_service import PoseEstimator

def main():
    """
    Standalone Desktop Version of the AI Fitness Tracker.
    Uses OpenCV for webcam capture and real-time visualization.
    """
    estimator = PoseEstimator()
    cap = cv2.VideoCapture(0)
    
    print("--- AI Fitness Tracker (Desktop Interface) ---")
    print("Initializing camera stream. Please stand in full view.")
    print("Press 'q' to terminate session.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Analyze frame via the centralized pose service
        processed_img, reps, state, warning = estimator.process_frame(frame)
        
        # UI Overlay Logic
        overlay_color = (245, 117, 16)
        cv2.rectangle(processed_img, (0, 0), (350, 100), overlay_color, -1)
        cv2.putText(processed_img, 'REPETITIONS', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
        cv2.putText(processed_img, str(reps), (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        cv2.putText(processed_img, 'SYSTEM STATE', (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
        cv2.putText(processed_img, state, (130, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        if warning:
            cv2.putText(processed_img, warning, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Render processed frame
        cv2.imshow('AI Fitness Tracker (Standard Analytics)', processed_img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
