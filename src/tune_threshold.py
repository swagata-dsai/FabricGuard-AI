import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    average_precision_score
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fabricguard_cnn.keras"
VAL_CSV = ROOT / "data" / "processed" / "splits" / "validation.csv"
IMAGE_DIR = ROOT / "data" / "processed" / "patches" / "images"
OUTPUT_DIR = ROOT / "outputs" / "metrics"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

df = pd.read_csv(VAL_CSV)

paths = [
    str(IMAGE_DIR / f"{patch_id}.png")
    for patch_id in df["patch_id"]
]
labels = df["label"].astype(int).values

missing = [p for p in paths if not Path(p).exists()]
if missing:
    raise FileNotFoundError(f"Missing image files: {len(missing)}")

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

precision, recall, thresholds = precision_recall_curve(
    labels, probabilities
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-8)
)

best_index = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_index])

predictions = (probabilities >= best_threshold).astype(int)

print("\nVALIDATION THRESHOLD ANALYSIS")
print("Best threshold:", round(best_threshold, 4))
print("Validation precision:", round(float(precision[best_index]), 4))
print("Validation recall:", round(float(recall[best_index]), 4))
print("Validation F1:", round(float(f1_scores[best_index]), 4))
print("Validation PR-AUC (Average Precision):",
      round(float(average_precision_score(labels, probabilities)), 4))

print("\nValidation Confusion Matrix:")
print(confusion_matrix(labels, predictions))

print("\nValidation Classification Report:")
print(classification_report(
    labels, predictions,
    target_names=["Normal", "Defect"],
    zero_division=0
))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "selected_threshold.txt", "w") as f:
    f.write(str(best_threshold))

print("\nSelected threshold saved successfully.")
