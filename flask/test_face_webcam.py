import cv2
import mediapipe as mp

# MediaPipe 얼굴 탐지기 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 웹캠 초기화 (0은 일반적으로 기본 웹캠을 가리킴)
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 이미지를 읽는데 실패했습니다. 스트림을 종료합니다.")
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

