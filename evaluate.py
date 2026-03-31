import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import os

def predict_from_image(img_path, true_label):
    """
    Since images are tiny mouth/eye crops, we use pixel intensity
    and edge density instead of MediaPipe landmarks.
    - Yawn images: mouth wide open = large dark region in center
    - No-yawn images: mouth closed = mostly skin tone, less dark area
    """
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
    yawn_score = (dark_ratio * 0.4) + (center_dark * 0.4) + (edge_density * 0.1) + (contrast / 255 * 0.1)
    predicted = 1 if yawn_score > 0.11 else 0

    return {
        "dark_ratio": dark_ratio,
        "edge_density": edge_density,
        "center_dark": center_dark,
        "contrast": contrast,
        "yawn_score": yawn_score,
        "predicted": predicted,
        "actual": true_label
    }

def evaluate_dataset(dataset_dir="dataset"):
    all_records = []
    os.makedirs("demo", exist_ok=True)

    for label_name, label_val in [("awake", 0), ("drowsy", 1)]:
        folder = os.path.join(dataset_dir, label_name)
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"\nProcessing {len(files)} {label_name} images...")

        for fname in tqdm(files):
            path = os.path.join(folder, fname)
            result = predict_from_image(path, label_val)
            if result:
                all_records.append(result)

    if not all_records:
        print("No records processed.")
        return

    df = pd.DataFrame(all_records)
    print(f"\nTotal images processed: {len(df)}")
    print(f"Awake: {len(df[df['actual']==0])}  |  Drowsy: {len(df[df['actual']==1])}")

    cm = confusion_matrix(df["actual"], df["predicted"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Awake", "Drowsy"],
                yticklabels=["Awake", "Drowsy"])
    plt.title("Confusion Matrix — Drowsiness Detection")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("demo/confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: demo/confusion_matrix.png")

    
    plt.figure(figsize=(8, 4))
    df[df["actual"] == 0]["yawn_score"].hist(bins=40, alpha=0.6,
                                              label="Awake", color="green")
    df[df["actual"] == 1]["yawn_score"].hist(bins=40, alpha=0.6,
                                              label="Drowsy", color="red")
    plt.axvline(x=0.11, color="black", linestyle="--", label="Threshold=0.11")
    plt.title("Yawn Score Distribution: Awake vs Drowsy")
    plt.xlabel("Yawn Score")
    plt.ylabel("Image Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("demo/score_distribution.png", dpi=150)
    plt.show()
    print("Saved: demo/score_distribution.png")

    plt.figure(figsize=(8, 4))
    df[df["actual"] == 0]["dark_ratio"].hist(bins=40, alpha=0.6,
                                              label="Awake", color="green")
    df[df["actual"] == 1]["dark_ratio"].hist(bins=40, alpha=0.6,
                                              label="Drowsy", color="red")
    plt.title("Dark Pixel Ratio: Awake vs Drowsy")
    plt.xlabel("Dark Pixel Ratio")
    plt.ylabel("Image Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("demo/dark_ratio_distribution.png", dpi=150)
    plt.show()
    print("Saved: demo/dark_ratio_distribution.png")

   
    print("\n--- Classification Report ---")
    print(classification_report(df["actual"], df["predicted"],
                                target_names=["Awake", "Drowsy"],
                                zero_division=0))

    df.to_csv("demo/results.csv", index=False)
    print("Saved: demo/results.csv")

if __name__ == "__main__":
    evaluate_dataset()