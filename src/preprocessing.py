from pathlib import Path
import shutil
import re

import numpy as np
import pandas as pd
from PIL import Image


DATASET = Path(r"A:\AITEX_Dataset")

DEFECT_DIR = DATASET / "Defect_images"
NORMAL_DIR = DATASET / "NODefect_images"
MASK_DIR = DATASET / "Mask_images"

OUTPUT_DIR = Path("data/processed/patches")
IMAGE_OUTPUT = OUTPUT_DIR / "images"
MASK_OUTPUT = OUTPUT_DIR / "masks"

PATCH_SIZE = 256


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def process_image(image_path, category, mask_paths=None):

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    height, width = image_array.shape[:2]

    if height != PATCH_SIZE:
        raise ValueError(
            f"Unexpected image height: {image_path}, {height}"
        )

    combined_mask = np.zeros((height, width), dtype=np.uint8)

    if category == "DEFECT":

        if not mask_paths:
            return []

        for mask_path in mask_paths:

            mask_image = Image.open(mask_path).convert("L")

            if mask_image.size != image.size:
                raise ValueError(
                    f"Image-mask size mismatch: {image_path.name}"
                )

            mask_array = np.array(mask_image)

            combined_mask = np.maximum(
                combined_mask,
                mask_array
            )

    combined_mask = np.where(
        combined_mask > 0, 255, 0
    ).astype(np.uint8)

    if category == "NORMAL":
        relative_name = image_path.relative_to(NORMAL_DIR)
    else:
        relative_name = image_path.relative_to(DEFECT_DIR)

    source_id = safe_name(
        category + "_" +
        relative_name.with_suffix("").as_posix().replace("/", "__")
    )

    records = []

    for index, x in enumerate(range(0, width, PATCH_SIZE)):

        patch = image_array[:, x:x + PATCH_SIZE, :]
        patch_mask = combined_mask[:, x:x + PATCH_SIZE]

        actual_width = patch.shape[1]

        if actual_width < PATCH_SIZE:

            padded_image = np.zeros(
                (PATCH_SIZE, PATCH_SIZE, 3),
                dtype=np.uint8
            )

            padded_mask = np.zeros(
                (PATCH_SIZE, PATCH_SIZE),
                dtype=np.uint8
            )

            padded_image[:, :actual_width, :] = patch
            padded_mask[:, :actual_width] = patch_mask

            patch = padded_image
            patch_mask = padded_mask

        label = int(np.any(patch_mask > 0))
        mask_pixels = int(np.count_nonzero(patch_mask))

        patch_id = f"{source_id}_patch_{index:02d}"

        Image.fromarray(patch).save(
            IMAGE_OUTPUT / f"{patch_id}.png"
        )

        Image.fromarray(patch_mask).save(
            MASK_OUTPUT / f"{patch_id}_mask.png"
        )

        records.append({
            "patch_id": patch_id,
            "source_image": source_id,
            "category": category,
            "patch_index": index,
            "label": label,
            "mask_pixels": mask_pixels
        })

    return records


def main():

    for folder in [DEFECT_DIR, NORMAL_DIR, MASK_DIR]:
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder missing: {folder}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
    MASK_OUTPUT.mkdir(parents=True, exist_ok=True)

    all_records = []
    skipped = []

    defect_images = sorted(DEFECT_DIR.glob("*.png"))
    normal_images = sorted(NORMAL_DIR.rglob("*.png"))

    print("Defect images found:", len(defect_images))
    print("Normal images found:", len(normal_images))

    for image_path in defect_images:

        mask_paths = sorted(
            MASK_DIR.glob(image_path.stem + "_mask*.png")
        )

        if not mask_paths:
            skipped.append(image_path.name)
            continue

        records = process_image(
            image_path,
            "DEFECT",
            mask_paths
        )

        all_records.extend(records)

    for image_path in normal_images:

        records = process_image(
            image_path,
            "NORMAL"
        )

        all_records.extend(records)

    if not all_records:
        raise RuntimeError("No patches were generated.")

    df = pd.DataFrame(all_records)

    csv_path = OUTPUT_DIR / "patch_labels.csv"
    df.to_csv(csv_path, index=False)

    print("\nPREPROCESSING COMPLETED")
    print("Total patches:", len(df))

    print("\nPatch labels:")
    print(df["label"].value_counts().sort_index())

    print("\nImages processed by category:")
    print(df.groupby("category")["source_image"].nunique())

    print("\nPatches by category:")
    print(df.groupby(["category", "label"]).size())

    print("\nSkipped defect images:", skipped)
    print("Unique source images:", df["source_image"].nunique())
    print("CSV saved:", csv_path)


if __name__ == "__main__":
    main()
