import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def get_score(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = np.sum(dark_mask > 0) / (64 * 64)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / (64 * 64)
    center = gray[20:44, 20:44]
    center_dark = np.sum(center < 60) / (24 * 24)
    contrast = gray.std()
    return (dark_ratio * 0.4) + (center_dark * 0.4) + (edge_density * 0.1) + (contrast / 255 * 0.1)

records = []
for label_name, label_val in [("awake", 0), ("drowsy", 1)]:
    folder = f"dataset/{label_name}"
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
    for fname in tqdm(files, desc=label_name):
        score = get_score(os.path.join(folder, fname))
        if score is not None:
            records.append({"score": score, "actual": label_val})

df = pd.DataFrame(records)
best_acc, best_thresh = 0, 0
for thresh in np.arange(0.05, 0.50, 0.01):
    preds = (df["score"] > thresh).astype(int)
    acc = (preds == df["actual"]).mean()
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"\nBest threshold: {best_thresh:.2f}")
print(f"Best accuracy:  {best_acc:.2%}")