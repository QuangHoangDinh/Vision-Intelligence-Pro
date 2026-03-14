import cv2
import mediapipe as mp
from ultralytics import YOLO

class VisionEngine:
    def __init__(self):
        # Load YOLOv8 model (Nano for speed)
        self.model = YOLO('yolov8n.pt')
        
        # Setup MediaPipe Pose and Face Detection
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def detect_objects(self, frame, conf=0.4):
        results = self.model(frame, conf=conf)
        return results[0].plot()

    def process_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb_frame)
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def process_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                self.mp_draw.draw_detection(frame, detection)
        return frame
