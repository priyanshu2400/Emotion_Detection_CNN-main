import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import tempfile
import time

# Function to load the model
def load_emotion_model():
    try:
        model = load_model('./model/model.h5')
        st.write("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the pre-trained models
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_emotion_model()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Streamlit app
st.title("Emotion Detector")

st.write("Upload a video file or use the webcam to detect emotions in real-time.")

# Video file upload
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Use webcam
use_webcam = st.checkbox("Use Webcam")

# Initialize video capture
cap = None
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
elif use_webcam:
    cap = cv2.VideoCapture(0)

# Initialize stop state
if 'stop' not in st.session_state:
    st.session_state.stop = False

# Create stop button
if st.button("Stop"):
    st.session_state.stop = True
    time.sleep(1)  # Add a delay to ensure the stop action is processed
    st.experimental_rerun()  # Refresh the page

# Display the video frame in Streamlit
frame_window = st.image([])

# Process video
if cap is not None:
    while cap.isOpened() and not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y - 10)  # Position label above the face rectangle
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Faces", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with detected faces and emotions
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

cv2.destroyAllWindows()
