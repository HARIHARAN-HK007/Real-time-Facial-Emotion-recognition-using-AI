import os

# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import cv2
import numpy as np
from fer import FER
import tensorflow as tf

# Initialize the FER detector with MTCNN
detector = FER(mtcnn=True)


# Function to provide feedback based on the detected emotion
def provide_feedback(emotion):
    feedback = {
        'happy': 'Great job! Keep it up!',
        'sad': 'Is everything okay? Need help?',
        'angry': 'Please calm down and focus.',
        'neutral': 'Let’s try to make the session more engaging!',
        'surprise': 'Wow! That was unexpected!',
        'fear': 'Is something bothering you?',
        'disgust': 'Is there an issue with the content?'
    }
    return feedback.get(emotion, "Let's keep going!")


# Function to capture video, detect emotion, and provide feedback
def capture_video_with_feedback():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        emotion, score = detector.top_emotion(frame)

        if emotion is None or score is None:
            emotion = 'neutral'
            score = 0

        feedback = provide_feedback(emotion)

        # Display the detected emotion and feedback
        cv2.putText(frame, f"{emotion} ({score * 100:.2f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection with Feedback', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Run the video capture with feedback
capture_video_with_feedback()
