from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import average_precision_score, f1_score

tf.keras.utils.set_random_seed(42)

ROOT = Path(__file__).resolve().parent.parent
PATCH_DIR = ROOT / "data/processed/patches/images"
SPLIT_DIR = ROOT / "data/processed/splits"
MODEL_DIR = ROOT / "models"
METRIC_DIR = ROOT / "outputs/metrics"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df = pd.read_csv(SPLIT_DIR / "validation.csv")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")

def prepare_data(df):
    paths = [
        str(PATCH_DIR / f"{pid}.png")
        for pid in df["patch_id"]
    ]
    labels = df["label"].astype("float32").values

    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    return paths, labels

train_paths, y_train = prepare_data(train_df)
val_paths, y_val = prepare_data(val_df)
test_paths, y_test = prepare_data(test_df)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(len(paths), seed=42)

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = create_dataset(train_paths, y_train, True)
val_ds = create_dataset(val_paths, y_val)
test_ds = create_dataset(test_paths, y_test)

classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.astype(int)
)
class_weights = dict(zip(classes, weights))

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomContrast(0.1)
], name="augmentation")

def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    return x

inputs = tf.keras.Input(shape=(128, 128, 3))
x = augmentation(inputs)

x = conv_block(x, 32)
x = conv_block(x, 64)
x = conv_block(x, 128)
x = conv_block(x, 256)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs, name="FabricGuard_CNN_V2")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(curve="PR", name="pr_auc")
    ]
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "fabricguard_cnn_v2.keras"),
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=7,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
]

print("\nSTARTING FABRICGUARD CNN V2 TRAINING")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

best_model = tf.keras.models.load_model(
    MODEL_DIR / "fabricguard_cnn_v2.keras"
)

probabilities = best_model.predict(test_ds).ravel()
predictions = (probabilities >= 0.5).astype(int)

report = classification_report(
    y_test.astype(int),
    predictions,
    target_names=["Normal", "Defect"],
    zero_division=0
)

metrics = {
    "accuracy": float(np.mean(predictions == y_test)),
    "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    "pr_auc": float(average_precision_score(y_test, probabilities))
}

with open(METRIC_DIR / "cnn_v2_test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(METRIC_DIR / "cnn_v2_classification_report.txt", "w") as f:
    f.write(report)

pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index=["Actual_Normal", "Actual_Defect"],
    columns=["Predicted_Normal", "Predicted_Defect"]
).to_csv(METRIC_DIR / "cnn_v2_confusion_matrix.csv")

pd.DataFrame(history.history).to_csv(
    METRIC_DIR / "cnn_v2_training_history.csv",
    index=False
)

print("\nCNN V2 TRAINING COMPLETED")
print("Test metrics:", metrics)
print("Model:", MODEL_DIR / "fabricguard_cnn_v2.keras")


