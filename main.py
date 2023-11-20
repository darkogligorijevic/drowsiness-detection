import cv2
import dlib
from scipy.spatial import distance as dist
import time
import winsound
import pygame

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

# Function to check if eyes are closed based on EAR
def are_eyes_closed(landmarks):
    landmarks_list = [(p.x, p.y) for p in landmarks]
    left_eye_indices = list(range(36, 42))
    right_eye_indices = list(range(42, 48))
    left_eye = [landmarks_list[i] for i in left_eye_indices]
    right_eye = [landmarks_list[i] for i in right_eye_indices]

    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)

    ear_avg = (ear_left + ear_right) / 2.0

    ear_threshold = 0.2

    return ear_avg < ear_threshold

# Function to play a sound using pygame
def play_sound(sound):
    pygame.mixer.Sound.play(sound)

# Function to play a beep (this should be default since .mp3 is ignored)
def play_beep():
    winsound.Beep(1000, 1000)

# Path to the facial landmarks predictor model
predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Open a video capture object for the webcam
video = cv2.VideoCapture(0)

# Initialize pygame mixer and load the sound file
pygame.mixer.init()
sound_path = "wakeywakey.mp3"
sound = pygame.mixer.Sound(sound_path)

# Initialize start_time variable
start_time = None

# Initialize alert_duration variable (in seconds)
alert_duration = 0.5

while True:
    # Read a frame from the webcam
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use dlib to detect faces in the frame
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    # Initialize variable for eyes_closed state
    eyes_closed = False

    # Loop over detected faces
    for face in faces:
        landmarks = predictor(gray, face)

        # Draw landmarks on the frame (green dots)
        for i in range(36, 48): # Range(36, 48) landmarks are drawing an eye
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Check if eyes are closed
        if are_eyes_closed(landmarks.parts()):
            if start_time is None:
                start_time = time.time()
                play_sound(sound)  # Play the sound when eyes are closed
                play_beep() # Play the beep when eyes are closed
            elif time.time() - start_time > alert_duration:
                eyes_closed = True
        else:
            start_time = None

    # Display text if eyes are closed
    if eyes_closed:
        cv2.putText(frame, "Eyes are closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame in a window
    cv2.imshow("Drowsiness Detection", frame)

    # Click 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
