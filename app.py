# # from flask import Flask, Response, render_template
# # import cv2
# # from utils import process_frame
# # import threading
# # import pyttsx3  # For text-to-speech
# # import queue
# # app = Flask(__name__)
# # video_capture = cv2.VideoCapture(0)  # Use webcam

# # # Initialize the text-to-speech engine
# # engine = pyttsx3.init()
# # engine.setProperty('rate', 150)  # Speed of speech
# # engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# # def text_to_audio(text):
# #     # Speak the given text
# #     engine.say(text)
# #     engine.runAndWait()

# # def capture_and_process():
# #     while True:
# #         ret, frame = video_capture.read()
# #         if not ret:
# #             continue
# #         processed, scene_pred = process_frame(frame)

# #         # Speak the scene prediction
# #         text_to_audio(f"The scene is {scene_pred}")

# #         # Save the processed frame to a buffer
# #         _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
# #         frame_jpeg = jpeg.tobytes()

# #         # Send the frame to the frontend (as part of streaming)
# #         frame_queue.put(frame_jpeg)

# # # Start the capture and process thread
# # frame_queue = queue.Queue()
# # threading.Thread(target=capture_and_process, daemon=True).start()

# # @app.route('/')
# # def index():
# #     return render_template("index.html")  # Your HTML template for the frontend

# # def generate_frames():
# #     while True:
# #         # Get the processed frame from the queue and yield it as part of the streaming response
# #         if not frame_queue.empty():
# #             frame = frame_queue.get()
# #             yield (b'--frame\r\n'
# #                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# # @app.route('/video_feed')
# # def video_feed():
# #     # Generate frames and stream them
# #     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5001)
# from flask import Flask, jsonify, render_template, send_from_directory
# from flask_cors import CORS
# import cv2
# import os
# import threading
# import time
# import json
# from utils import process_frame, text_to_audio

# app = Flask(__name__, static_folder='static', template_folder='templates')
# CORS(app)  # Enable CORS for all routes

# # Ensure the static directory exists
# os.makedirs('static', exist_ok=True)

# # Global variables to store the latest detection results
# latest_scene = "Processing environment..."
# latest_objects = []
# processing_active = True
# video_capture = None

# def initialize_camera():
#     global video_capture
#     video_capture = cv2.VideoCapture(0)
#     if not video_capture.isOpened():
#         print("Error: Could not open webcam")
#         return False
#     return True

# def release_camera():
#     global video_capture
#     if video_capture is not None:
#         video_capture.release()

# def capture_and_process():
#     global latest_scene, latest_objects, processing_active, video_capture
    
#     if not initialize_camera():
#         processing_active = False
#         return
        
#     while processing_active:
#         try:
#             ret, frame = video_capture.read()
#             if not ret:
#                 print("Failed to grab frame")
#                 time.sleep(1)
#                 continue
                
#             # Process the frame using your existing function
#             processed_frame, scene_pred = process_frame(frame)
            
#             # Update the global variables with detection results
#             latest_scene = scene_pred
            
#             # Extract object information from the processed frame
#             # (This would depend on how your process_frame function works)
#             # For demonstration, we'll assume objects are available in the processed_frame
#             # In a real implementation, modify this to match your process_frame output
            
#             # Save the processed frame for the frontend to access
#             cv2.imwrite("static/output.jpg", cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
            
#             # Don't process too frequently to avoid overloading
#             time.sleep(0.5)
            
#         except Exception as e:
#             print(f"Error in capture_and_process: {e}")
#             time.sleep(1)
    
#     release_camera()

# # Routes for serving the React app and API endpoints
# @app.route('/')
# def index():
#     """Serve the React frontend"""
#     return render_template('index.html')

# @app.route('/api/detection-results')
# def get_detection_results():
#     """Return the latest detection results as JSON"""
#     global latest_scene, latest_objects
    
#     detection_text = f"The scene is {latest_scene}"
#     if latest_objects:
#         detection_text += f". Detected {', '.join(latest_objects)}."
    
#     return jsonify({
#         'scene': latest_scene,
#         'objects': latest_objects,
#         'detectionText': detection_text
#     })

# @app.route('/api/speak/<text>')
# def speak(text):
#     """Endpoint to trigger text-to-speech on the server (optional)"""
#     try:
#         text_to_audio(text)
#         return jsonify({'success': True})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     """Serve static files including the processed image"""
#     return send_from_directory(app.static_folder, filename)

# @app.route('/api/health')
# def health_check():
#     """Simple health check endpoint"""
#     return jsonify({'status': 'running'})

# # Create the templates directory and index.html file
# def create_template_files():
#     os.makedirs('templates', exist_ok=True)
    
#     # Basic HTML file that will load the React app
#     index_html = """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>Guide for the Blind</title>
#         <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
#         <style>
#             body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; }
#             #root { min-height: 100vh; }
#         </style>
#     </head>
#     <body>
#         <div id="root"></div>
        
#         <!-- Load React from CDN for development -->
#         <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
#         <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
#         <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        
#         <!-- Load our compiled React app -->
#         <script src="/static/app.js"></script>
#     </body>
#     </html>
#     """
    
#     with open('templates/index.html', 'w') as f:
#         f.write(index_html)

# def shutdown_server():
#     global processing_active
#     processing_active = False

# if __name__ == '__main__':
#     try:
#         # Create necessary files
#         create_template_files()
        
#         # Start the frame capture and processing thread
#         processing_thread = threading.Thread(target=capture_and_process, daemon=True)
#         processing_thread.start()
        
#         # Run the Flask app
#         app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
#     finally:
#         # Ensure resources are released when the app exits
#         shutdown_server()




import streamlit as st
import cv2
import torch
import threading
import queue
import numpy as np
from utils import process_frame  # Your process_frame logic
import pyttsx3
from PIL import Image

# Set Streamlit page config
st.set_page_config(page_title="Live Object Detection", layout="wide")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

frame_queue = queue.Queue()

def text_to_audio(text):
    engine.say(text)
    engine.runAndWait()

def capture_and_process():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        # Process frame and get prediction
        processed, scene_pred = process_frame(frame)

        # Speak the scene description
        text_to_audio(f"The scene is {scene_pred}")

        # Store original and processed frames
        frame_queue.put((frame, processed))

# Start background thread
threading.Thread(target=capture_and_process, daemon=True).start()

# Streamlit UI
st.title("🔍 Real-Time Object Detection & Scene Analysis")

# Placeholders for input and output
input_placeholder, output_placeholder = st.columns(2)

input_image = input_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Live Input", channels="BGR")
output_image = output_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Processed Output", channels="RGB")

while True:
    if not frame_queue.empty():
        raw, processed = frame_queue.get()

        # Display images
        input_image.image(raw, channels="BGR", caption="Live Input")
        output_image.image(processed, channels="RGB", caption="Processed Output")
