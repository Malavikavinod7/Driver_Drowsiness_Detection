
import cv2
import mediapipe as mp
import pygame
from detector import compute_metrics
from utils import (LEFT_EYE, RIGHT_EYE, MOUTH,
                   EAR_THRESH, MAR_THRESH, CONSEC_FRAMES,
                   draw_landmarks, draw_status)

pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound("assets/alert.wav")
except:
    alert_sound = None
    print("No alert.wav found — audio disabled")

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
frame_counter = 0
alert_playing = False

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        drowsy = False
        yawning = False

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ear, mar = compute_metrics(lm, w, h)

            drowsy = ear < EAR_THRESH
            yawning = mar > MAR_THRESH

            if drowsy:
                frame_counter += 1
            else:
                frame_counter = 0
                if alert_playing:
                    if alert_sound: alert_sound.stop()
                    alert_playing = False

            sustained_drowsy = frame_counter >= CONSEC_FRAMES

            if (sustained_drowsy or yawning) and not alert_playing:
                if alert_sound: alert_sound.play(-1)
                alert_playing = True

            if sustained_drowsy or yawning:
                cv2.putText(frame, "DROWSINESS ALERT!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            draw_landmarks(frame, lm, LEFT_EYE, w, h, (0, 255, 255))
            draw_landmarks(frame, lm, RIGHT_EYE, w, h, (0, 255, 255))
            draw_landmarks(frame, lm, MOUTH, w, h, (255, 0, 255))
            draw_status(frame, ear, mar, drowsy, yawning)

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()