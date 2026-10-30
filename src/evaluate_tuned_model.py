import numpy as np
import pandas as pd
import tensorflow as tf
import json

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fabricguard_cnn.keras"
TEST_CSV = ROOT / "data" / "processed" / "splits" / "test.csv"
IMAGE_DIR = ROOT / "data" / "processed" / "patches" / "images"
METRIC_DIR = ROOT / "outputs" / "metrics"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

with open(METRIC_DIR / "selected_threshold.txt") as f:
    threshold = float(f.read().strip())

df = pd.read_csv(TEST_CSV)

paths = [
    str(IMAGE_DIR / f"{patch_id}.png")
    for patch_id in df["patch_id"]
]
labels = df["label"].astype(int).values

missing = [p for p in paths if not Path(p).exists()]
if missing:
    raise FileNotFoundError(f"Missing images: {len(missing)}")

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.load_model(MODEL_PATH)
probabilities = model.predict(dataset, verbose=1).ravel()
predictions = (probabilities >= threshold).astype(int)

results = {
    "threshold": threshold,
    "accuracy": float(accuracy_score(labels, predictions)),
    "precision": float(precision_score(labels, predictions, zero_division=0)),
    "recall": float(recall_score(labels, predictions, zero_division=0)),
    "f1_score": float(f1_score(labels, predictions, zero_division=0)),
    "pr_auc": float(average_precision_score(labels, probabilities))
}

print("\nFINAL TEST RESULTS")
for key, value in results.items():
    print(f"{key}: {value:.4f}")

cm = confusion_matrix(labels, predictions, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
report = classification_report(
    labels, predictions,
    target_names=["Normal", "Defect"],
    zero_division=0
)
print(report)

METRIC_DIR.mkdir(parents=True, exist_ok=True)

with open(METRIC_DIR / "tuned_test_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

pd.DataFrame(
    cm,
    index=["Actual_Normal", "Actual_Defect"],
    columns=["Predicted_Normal", "Predicted_Defect"]
).to_csv(METRIC_DIR / "tuned_confusion_matrix.csv")

with open(METRIC_DIR / "tuned_classification_report.txt", "w") as f:
    f.write(report)

print("\nFINAL TEST EVALUATION COMPLETED")
print("Results saved in:", METRIC_DIR)
