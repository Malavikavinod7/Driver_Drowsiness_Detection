# utils.py
import cv2

# MediaPipe landmark indices
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291, 39,  181, 0,   17, 269, 405]

EAR_THRESH    = 0.25
MAR_THRESH    = 0.60
CONSEC_FRAMES = 48

def draw_landmarks(frame, landmarks, indices, img_w, img_h, color=(0, 255, 0)):
    for i in indices:
        x = int(landmarks[i].x * img_w)
        y = int(landmarks[i].y * img_h)
        cv2.circle(frame, (x, y), 2, color, -1)

def draw_status(frame, ear, mar, drowsy, yawning):
    h, w = frame.shape[:2]
    color = (0, 0, 255) if (drowsy or yawning) else (0, 255, 0)
    cv2.putText(frame, f"EAR: {ear:.2f}", (30, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (30, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if drowsy:
        cv2.putText(frame, "EYES CLOSED", (w - 220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if yawning:
        cv2.putText(frame, "YAWNING", (w - 200, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)