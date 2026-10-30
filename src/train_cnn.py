from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Reproducibility
tf.keras.utils.set_random_seed(42)

# Paths
PATCH_DIR = Path("data/processed/patches/images")
SPLIT_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models")
METRIC_DIR = Path("outputs/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25


# Load split CSV files
train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df = pd.read_csv(SPLIT_DIR / "validation.csv")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")


# Create image paths
def prepare_paths(df):
    paths = [
        str(PATCH_DIR / f"{patch_id}.png")
        for patch_id in df["patch_id"]
    ]

    labels = df["label"].astype("float32").values

    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    return paths, labels


train_paths, y_train = prepare_paths(train_df)
val_paths, y_val = prepare_paths(val_df)
test_paths, y_test = prepare_paths(test_df)


# TensorFlow image pipeline
def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(
    train_paths,
    y_train,
    training=True
)

val_ds = create_dataset(
    val_paths,
    y_val
)

test_ds = create_dataset(
    test_paths,
    y_test
)


# Calculate class weights
classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.astype(int)
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("Class weights:", class_weights)


# Build CNN architecture
model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(128, 128, 3)
    ),

    tf.keras.layers.Conv2D(
        32, 3, activation="relu", padding="same"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        64, activation="relu"
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        1, activation="sigmoid"
    )
])


# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(
            curve="PR",
            name="pr_auc"
        )
    ]
)


model.summary()


# Training callbacks
callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "fabricguard_cnn.keras"),
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True
    )
]


# Train
print("\nSTARTING CNN TRAINING")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# Load best saved model
best_model = tf.keras.models.load_model(
    MODEL_DIR / "fabricguard_cnn.keras"
)


# Test evaluation
print("\nEVALUATING TEST DATA")

probabilities = best_model.predict(test_ds).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)

y_true = y_test.astype(int)


# Metrics
metrics = {
    "accuracy": float(
        accuracy_score(y_true, predictions)
    ),

    "precision": float(
        precision_score(
            y_true, predictions, zero_division=0
        )
    ),

    "recall": float(
        recall_score(
            y_true, predictions, zero_division=0
        )
    ),

    "f1_score": float(
        f1_score(
            y_true, predictions, zero_division=0
        )
    ),

    "pr_auc": float(
        average_precision_score(
            y_true, probabilities
        )
    )
}


# Save classification report
report = classification_report(
    y_true,
    predictions,
    target_names=["Normal", "Defect"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

print("\nTEST METRICS")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


# Save outputs
with open(
    METRIC_DIR / "test_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)


with open(
    METRIC_DIR / "classification_report.txt",
    "w"
) as file:
    file.write(report)


cm = confusion_matrix(y_true, predictions)

pd.DataFrame(
    cm,
    index=["Actual_Normal", "Actual_Defect"],
    columns=["Predicted_Normal", "Predicted_Defect"]
).to_csv(
    METRIC_DIR / "confusion_matrix.csv"
)


print("\nTRAINING COMPLETED")
print("Model:", MODEL_DIR / "fabricguard_cnn.keras")
print("Metrics saved:", METRIC_DIR)

