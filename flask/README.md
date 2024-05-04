1. **Set Up Flask**: Create a new Flask application.
2. **Receive Video Data**: Set up an endpoint to receive video uploads from the Android application.
3. **Process Video**: Extract frames and keypoints from the received video.
4. **Load Model**: Load your trained model.
5. **Predict**: Use the model to predict the video's content based on the processed data.
6. **Return Result**: Send the prediction result back to the Android app.

<br>
<br>

```bash
python3 -m venv env
python3 -m pip install --upgrade pip
pip install flask 
```

```bash
source env/bin/activate
deactivate
```

```bash 
python app.py
```
