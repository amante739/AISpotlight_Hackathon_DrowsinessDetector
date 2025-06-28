Drowsiness Detection System
Overview
This project implements a real-time drowsiness detection system using computer vision techniques with OpenCV and audio alerts using Pygame. It detects faces and eyes to monitor whether a person's eyes are open or closed, calculating blink frequency and triggering alerts for potential drowsiness.
Features

Detects frontal and profile faces using Haar Cascade classifiers
Tracks eye openness to detect blinks and prolonged eye closure
Logs events (blinks, drowsiness alerts) to a file
Displays real-time status, FPS, and blink frequency on the video feed
Plays an audio alert when drowsiness is detected

Prerequisites

Python 3.x
OpenCV (cv2) for computer vision tasks
NumPy for numerical operations
Pygame for audio alerts
A webcam for video input
An audio file named audio.mp3 in the project directory for alerts

Installation

Install Python: Ensure Python 3.x is installed on your system.
Install Dependencies: Install the required Python libraries using pip:pip install opencv-python numpy pygame


Prepare Audio File: Place an audio.mp3 file in the project directory for drowsiness alerts.
Download Haar Cascades: The script uses OpenCV's pre-trained Haar Cascade classifiers, which are included with the OpenCV installation. Ensure they are accessible in cv2.data.haarcascades.

Usage

Run the Script: Execute the Python script in your terminal or IDE:python drowsiness_detection.py


Monitor Output: The webcam feed will display with:
Blue rectangles around detected faces
Green rectangles around detected eyes
Text overlays for FPS, status (Awake, Stay Alert!, Drowsiness Detected!), and blink frequency


Exit the Program: Press q to quit the application.
View Logs: Check the log.txt file for recorded events, including timestamps for blinks and drowsiness alerts.

Configuration
The script includes configurable parameters in the FACE_DETECTION_PARAMS dictionary:

FRONTAL_SCALE: Scale factor for frontal face detection (default: 1.1)
FRONTAL_NEIGHBORS: Minimum neighbors for frontal face detection (default: 5)
PROFILE_SCALE: Scale factor for profile face detection (default: 1.1)
PROFILE_NEIGHBORS: Minimum neighbors for profile face detection (default: 5)
EYE_SCALE: Scale factor for eye detection (default: 1.1)
EYE_NEIGHBORS: Minimum neighbors for eye detection (default: 8)
MIN_FACE_SIZE: Minimum face size for detection (default: (30, 30))

Other parameters:

EYE_CLOSED_THRESHOLD: Duration (seconds) to consider an eye closed (default: 0.2)
DROWSY_THRESHOLD: Duration (seconds) for drowsiness alert (default: 2.0)
FRAME_REDUCTION: Frame size reduction factor for performance (default: 0.5)

How It Works

Face and Eye Detection:
Uses Haar Cascade classifiers to detect frontal and profile faces, as well as left and right eyes.
Processes eye regions to determine if eyes are open or closed based on contour area and aspect ratio.


Drowsiness Detection:
Tracks the duration of eye closure.
If eyes are closed for longer than EYE_CLOSED_THRESHOLD, logs a potential drowsiness event.
If eyes remain closed beyond DROWSY_THRESHOLD, plays an audio alert.


Blink Frequency:
Counts blinks per minute and logs the frequency every 60 seconds.


Logging:
Events (blinks, drowsiness levels) are written to log.txt with timestamps.


Display:
Shows real-time video with face/eye rectangles and status information.



Notes

Ensure good lighting and clear webcam visibility for accurate detection.
The system may require tuning of parameters (e.g., scale factors, thresholds) based on lighting conditions or hardware.
The audio file (audio.mp3) must be present in the project directory, or the script will fail to load the sound.

Troubleshooting

Webcam Issues: Ensure the webcam is accessible and not used by another application.
Cascade Classifier Errors: Verify that OpenCV is installed correctly and Haar Cascade files are available.
Performance Issues: Adjust FRAME_REDUCTION to a lower value (e.g., 0.3) to reduce processing load on slower systems.

License
This project is for educational purposes and provided as-is. Ensure compliance with OpenCV and Pygame licensing terms for commercial use.