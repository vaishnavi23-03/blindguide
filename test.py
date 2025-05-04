from flask import Flask
import cv2
from utils import process_frame
import threading
import pyttsx3  # For text-to-speech

app = Flask(__name__)
video_capture = cv2.VideoCapture(0)  # Use webcam

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

def text_to_audio(text):
    # Speak the given text
    engine.say(text)
    engine.runAndWait()

def capture_and_process():
    while True:
        print("Working on frame")
        ret, frame = video_capture.read()
        if not ret:
            continue
        processed, scene_pred = process_frame(frame)
        
        # Speak the scene prediction
        text_to_audio(f"The scene is {scene_pred}")
        
        # Save the processed frame
        cv2.imwrite("static/output.jpg", cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        print("Frame processed and saved")

# Start the capture and process thread
threading.Thread(target=capture_and_process, daemon=True).start()

@app.route('/')
def index():
    return "Video processing started..."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
