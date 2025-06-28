import cv2
import numpy as np
import time
from datetime import datetime
import pygame

pygame.mixer.init()
alert_sound = pygame.mixer.Sound('audio.mp3')

FACE_DETECTION_PARAMS = {
    'FRONTAL_SCALE': 1.1,
    'FRONTAL_NEIGHBORS': 5,
    'PROFILE_SCALE': 1.1,
    'PROFILE_NEIGHBORS': 5,
    'EYE_SCALE': 1.1,
    'EYE_NEIGHBORS': 8,
    'MIN_FACE_SIZE': (30, 30)
}

EYE_CLOSED_THRESHOLD = 0.2
DROWSY_THRESHOLD = 2.0
FRAME_REDUCTION = 0.5


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')


if (face_cascade.empty() or eye_cascade.empty() or profile_face_cascade.empty() or
        left_eye_cascade.empty() or right_eye_cascade.empty()):
    raise Exception("Error loading cascade classifiers")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Error opening webcam")


eyes_closed_start = None
drowsiness_level = 0
blink_count = 0
last_blink_time = time.time()
blink_frequency = 0
frame_count = 0
start_time = time.time()
log_file = open('log.txt', 'w')


def calculate_fps():
    global frame_count, start_time
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        return fps
    return None


def log_event(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file.write(f"{timestamp}: {message}\n")
    log_file.flush()


def process_eye(eye_roi):
    if eye_roi.size == 0:
        return False


    if len(eye_roi.shape) > 2:
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = eye_roi


    gray = cv2.equalizeHist(gray)


    gray = cv2.bilateralFilter(gray, 5, 75, 75)


    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)
    center = (width // 2, height // 2)
    cv2.ellipse(mask, center, (width // 2, height // 2), 0, 0, 360, 255, -1)


    gray = cv2.bitwise_and(gray, gray, mask=mask)


    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)


    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)


    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False


    max_area = max([cv2.contourArea(c) for c in contours])
    total_area = eye_roi.shape[0] * eye_roi.shape[1]
    area_ratio = max_area / total_area


    aspect_ratio = 0
    if max_area > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0


    is_eye_open = (
            area_ratio > 0.1 and
            max_area > 50 and
            aspect_ratio > 0.2
    )

    return is_eye_open


def detect_frontal_eyes(roi_gray, roi_color):
    height, width = roi_gray.shape
    midpoint = width // 2

    # Left eye region
    left_roi = roi_gray[height // 4:3 * height // 4, 0:midpoint]
    left_roi_color = roi_color[height // 4:3 * height // 4, 0:midpoint]
    left_eyes = left_eye_cascade.detectMultiScale(
        left_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # Right eye region
    right_roi = roi_gray[height // 4:3 * height // 4, midpoint:width]
    right_roi_color = roi_color[height // 4:3 * height // 4, midpoint:width]
    right_eyes = right_eye_cascade.detectMultiScale(
        right_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    left_eye_open = False
    right_eye_open = False


    for (ex, ey, ew, eh) in left_eyes:
        cv2.rectangle(left_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eye_roi = left_roi_color[ey:ey + eh, ex:ex + ew]
        if process_eye(eye_roi):
            left_eye_open = True
            break


    for (ex, ey, ew, eh) in right_eyes:

        ex_adjusted = ex + midpoint
        cv2.rectangle(roi_color, (ex_adjusted, ey), (ex_adjusted + ew, ey + eh), (0, 255, 0), 2)
        eye_roi = right_roi_color[ey:ey + eh, ex:ex + ew]
        if process_eye(eye_roi):
            right_eye_open = True
            break


    return left_eye_open or right_eye_open


def detect_profile_eyes(roi_gray, roi_color, is_flipped):
    height = roi_gray.shape[0]
    width = roi_gray.shape[1]


    eye_region_x = width // 2 if is_flipped else 0
    eye_region_width = width // 2
    eye_region_y = height // 4
    eye_region_height = height // 2

    eye_roi_gray = roi_gray[eye_region_y:eye_region_y + eye_region_height,
                   eye_region_x:eye_region_x + eye_region_width]


    eyes = eye_cascade.detectMultiScale(
        eye_roi_gray,
        scaleFactor=FACE_DETECTION_PARAMS['EYE_SCALE'],
        minNeighbors=FACE_DETECTION_PARAMS['EYE_NEIGHBORS']
    )

    eye_detected = False
    for (ex, ey, ew, eh) in eyes:
        ex_adjusted = ex + eye_region_x
        ey_adjusted = ey + eye_region_y
        cv2.rectangle(roi_color, (ex_adjusted, ey_adjusted),
                      (ex_adjusted + ew, ey_adjusted + eh), (0, 255, 0), 2)
        eye_roi = roi_color[ey_adjusted:ey_adjusted + eh, ex_adjusted:ex_adjusted + ew]
        if process_eye(eye_roi):
            eye_detected = True
            break

    return eye_detected


def detect_faces_and_eyes(frame, gray):
    eyes_open = False
    faces_detected = []

    # Detect frontal faces
    frontal_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTION_PARAMS['FRONTAL_SCALE'],
        minNeighbors=FACE_DETECTION_PARAMS['FRONTAL_NEIGHBORS'],
        minSize=FACE_DETECTION_PARAMS['MIN_FACE_SIZE']
    )
    faces_detected.extend([(x, y, w, h, 'frontal') for (x, y, w, h) in frontal_faces])


    profile_faces = profile_face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTION_PARAMS['PROFILE_SCALE'],
        minNeighbors=FACE_DETECTION_PARAMS['PROFILE_NEIGHBORS'],
        minSize=FACE_DETECTION_PARAMS['MIN_FACE_SIZE']
    )
    faces_detected.extend([(x, y, w, h, 'profile') for (x, y, w, h) in profile_faces])


    flipped = cv2.flip(gray, 1)
    profile_faces_flipped = profile_face_cascade.detectMultiScale(
        flipped,
        scaleFactor=FACE_DETECTION_PARAMS['PROFILE_SCALE'],
        minNeighbors=FACE_DETECTION_PARAMS['PROFILE_NEIGHBORS'],
        minSize=FACE_DETECTION_PARAMS['MIN_FACE_SIZE']
    )

    for (x, y, w, h) in profile_faces_flipped:
        x = frame.shape[1] - x - w  # Adjust x coordinate for flip
        faces_detected.append((x, y, w, h, 'profile_flipped'))

    for (x, y, w, h, face_type) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        if face_type == 'frontal':
            eyes_open = detect_frontal_eyes(roi_gray, roi_color)
        else:
            eyes_open = detect_profile_eyes(roi_gray, roi_color, face_type == 'profile_flipped')

        if eyes_open:
            break

    return eyes_open



try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break


        frame = cv2.resize(frame, (0, 0), fx=FRAME_REDUCTION, fy=FRAME_REDUCTION)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        current_time = time.time()
        eyes_open = detect_faces_and_eyes(frame, gray)
        alert_message = ""

        if eyes_open:
            if eyes_closed_start is not None:
                blink_duration = current_time - eyes_closed_start
                if blink_duration < EYE_CLOSED_THRESHOLD:
                    blink_count += 1
                    log_event(f"Blink detected, duration: {blink_duration:.2f} sec")
                eyes_closed_start = None
            drowsiness_level = 0
            alert_message = "Awake"
        else:
            if eyes_closed_start is None:
                eyes_closed_start = current_time
            else:
                closed_duration = current_time - eyes_closed_start
                if closed_duration >= DROWSY_THRESHOLD:
                    drowsiness_level = 2
                    alert_message = "Drowsiness Detected!"
                    alert_sound.play()
                    log_event("Drowsiness Level 2 detected")
                elif closed_duration >= EYE_CLOSED_THRESHOLD:
                    drowsiness_level = 1
                    alert_message = "Stay Alert!"
                    log_event("Drowsiness Level 1 detected")
                else:
                    drowsiness_level = 0
                    alert_message = "Awake"


        if current_time - last_blink_time >= 60:
            blink_frequency = blink_count
            blink_count = 0
            last_blink_time = current_time
            log_event(f"Blink frequency: {blink_frequency} blinks/min")


        fps = calculate_fps()
        if fps:
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {alert_message}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Blink Freq: {blink_frequency} blinks/min", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print("Resources released")