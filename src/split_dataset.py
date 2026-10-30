import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load patch metadata
df = pd.read_csv("data/processed/patches/patch_labels.csv")

# Create one record per original image
sources = df[["source_image", "category"]].drop_duplicates()

# First split: 70% train, 30% temporary
train_sources, temp_sources = train_test_split(
    sources,
    test_size=0.30,
    random_state=42,
    stratify=sources["category"]
)

# Second split: 15% validation, 15% test
val_sources, test_sources = train_test_split(
    temp_sources,
    test_size=0.50,
    random_state=42,
    stratify=temp_sources["category"]
)

# Map source images to their respective splits
train_df = df[df["source_image"].isin(train_sources["source_image"])]
val_df = df[df["source_image"].isin(val_sources["source_image"])]
test_df = df[df["source_image"].isin(test_sources["source_image"])]

# Save split files
output = Path("data/processed/splits")
output.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output / "train.csv", index=False)
val_df.to_csv(output / "validation.csv", index=False)
test_df.to_csv(output / "test.csv", index=False)

# Verify no source image leakage
train_ids = set(train_df["source_image"])
val_ids = set(val_df["source_image"])
test_ids = set(test_df["source_image"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("GROUP-BASED SPLIT COMPLETED")

for name, part in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:
    print(f"\n{name}")
    print("Source images:", part["source_image"].nunique())
    print("Patches:", len(part))
    print("Patch labels:")
    print(part["label"].value_counts().sort_index())

print("\nData leakage check: PASSED")
print("Split files saved in:", output)
