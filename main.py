import cv2
import dlib
from scipy.spatial import distance as dist
import time
import winsound
import pygame

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

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

def play_sound(sound):
    pygame.mixer.Sound.play(sound)

predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

video = cv2.VideoCapture(0)
start_time = None

pygame.mixer.init()
sound_path = "wakeywakey.mp3"
sound = pygame.mixer.Sound(sound_path)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    eyes_closed = False

    for face in faces:
        landmarks = predictor(gray, face)

        for i in range(36, 48):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if are_eyes_closed(landmarks.parts()):
            if start_time is None:
                start_time = time.time()
                play_sound(sound)
            elif time.time() - start_time:
                eyes_closed = True
        else:
            start_time = None

    if eyes_closed:
        cv2.putText(frame, "Eyes are closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
