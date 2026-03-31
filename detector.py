# detector.py
from scipy.spatial import distance as dist
from utils import LEFT_EYE, RIGHT_EYE, MOUTH

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
           for i in eye_indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, mouth_indices, img_w, img_h):
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
           for i in mouth_indices]
    A = dist.euclidean(pts[2], pts[6])
    B = dist.euclidean(pts[3], pts[7])
    C = dist.euclidean(pts[0], pts[4])
    return (A + B) / (2.0 * C)

def compute_metrics(landmarks, img_w, img_h):
    ear = (eye_aspect_ratio(landmarks, LEFT_EYE, img_w, img_h) +
           eye_aspect_ratio(landmarks, RIGHT_EYE, img_w, img_h)) / 2.0
    mar = mouth_aspect_ratio(landmarks, MOUTH, img_w, img_h)
    return ear, mar