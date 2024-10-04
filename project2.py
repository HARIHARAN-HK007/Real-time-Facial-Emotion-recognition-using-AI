import os
import cv2
import numpy as np
from fer import FER
import tensorflow as tf
import mss
import mss.tools

# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Initialize the FER detector with MTCNN
detector = FER(mtcnn=True)

# Function to provide feedback based on the detected emotion
def provide_feedback(emotion):
    feedback = {
        'happy': 'Great job! Keep it up!',
        'sad': 'Is everything okay? Need help?',
        'angry': 'Please calm down and focus.',
        'neutral': 'Let’s try to make the class more engaging!',
        'surprise': 'Wow! That was unexpected!',
        'fear': 'Is something bothering you?',
        'disgust': 'Is there an issue with the content?'
    }
    return feedback.get(emotion, "Let's keep going!")

# Function to capture screen from a specific region
def capture_screen(region=None):
    with mss.mss() as sct:
        # Capture the screen
        screenshot = sct.grab(region)
        img = np.array(screenshot)

        # Convert RGB to BGR (OpenCV format)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

# Function to capture video, detect emotion, and provide feedback
def capture_video_with_feedback(region=None):
    while True:
        frame = capture_screen(region)

        # Detect emotion
        emotion, score = detector.top_emotion(frame)

        # Default values if no emotion is detected
        if emotion is None or score is None:
            emotion = 'Unknown'
            score = 0

        feedback = provide_feedback(emotion)

        # Display emotion and feedback
        cv2.putText(frame, f"{emotion} ({score * 100:.2f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection with Feedback', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Define the region for Zoom window capture (left, top, width, height)
zoom_window_region = {'top': 100, 'left': 100, 'width': 800, 'height': 600}

# Start the video capture and feedback loop
capture_video_with_feedback(region=zoom_window_region)
