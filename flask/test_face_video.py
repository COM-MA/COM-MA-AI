import cv2
import mediapipe as mp

# MediaPipe 얼굴 탐지기 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 카메라 또는 비디오 파일 초기화
cap = cv2.VideoCapture('test.mp4')

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # 원본 프레임으로 다시 변환하여 얼굴 탐지 결과 그리기
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # 결과 표시
        cv2.imshow('MediaPipe Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc 키로 종료
            break

cap.release()
cv2.destroyAllWindows()
