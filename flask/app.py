from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import tempfile
import uuid
import mediapipe as mp
from tensorflow.keras.models import load_model

app = Flask(__name__)

# MediaPipe initialization
mp_holistic = mp.solutions.holistic

# load the model
model = load_model('model_4.h5')


def extract_keypoints(results):
    if not results.pose_landmarks or not results.left_hand_landmarks or not results.right_hand_landmarks:
        return np.zeros([258])  # Adjust the size according to your model's input size if necessary
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

def process_video(video):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
        video.save(temp_video.name)
        cap = cv2.VideoCapture(temp_video.name)
        if not cap.isOpened():
            raise IOError("Cannot open video")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = np.linspace(0, frame_count - 1, num=30, dtype=int)  # Adjusted to 30 frames
        keypoints_list = []

        try:
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for i in frames_to_process:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False
                    results = holistic.process(frame)
                    frame.flags.writeable = True
                    keypoints = extract_keypoints(results)
                    keypoints_list.append(keypoints)

            keypoints_array = np.array(keypoints_list)
            if keypoints_array.shape[0] < 30:
                padding = np.zeros((30 - keypoints_array.shape[0], 258))
                keypoints_array = np.concatenate((keypoints_array, padding), axis=0)
            elif keypoints_array.shape[0] > 30:
                keypoints_array = keypoints_array[:30]

            return np.expand_dims(keypoints_array, axis=0)
        finally:
            cap.release()

    
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        keypoints = process_video(file)
        prediction = model.predict(keypoints)
        if prediction.shape == (1, 10):
            predicted_class = np.argmax(prediction, axis=1)
            actions = ['곰', '놀이터', '다리', '바다', '벌', '병원', '선생님', '엄마', '유치원', '학교']
            predicted_label = actions[predicted_class[0]]
            # 선생님일 때 선생으로 출력 변경
            if predicted_label == '선생님':
                predicted_label = '선생'
            if predicted_label == '다리':
                predicted_label = '(시설물)다리'
            return jsonify({"prediction": predicted_label}), 200
        else:
            return jsonify({"error": "Model prediction output shape mismatch"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0')
