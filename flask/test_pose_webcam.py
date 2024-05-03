import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Holistic 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 웹캠 초기화
cap = cv2.VideoCapture(0)

# Holistic 모델 설정
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 이미지를 읽는데 실패했습니다. 스트림을 종료합니다.")
            break

        # 프레임을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # 원본 프레임으로 다시 변환하여 얼굴 탐지 결과 그리기
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 얼굴, 손, 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # 결과 표시
        cv2.imshow('MediaPipe Holistic', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc 키로 종료
            break

cap.release()
cv2.destroyAllWindows()

