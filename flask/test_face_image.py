import cv2
import mediapipe as mp

# MediaPipe 얼굴 탐지기 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# 이미지 로드
image = cv2.imread('test.png') 
# 이미지 처리
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_detection.process(image)

# 얼굴 탐지 결과 출력
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
if results.detections:
    for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

# 이미지 표시
cv2.imshow('MediaPipe Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 자원 해제
face_detection.close()