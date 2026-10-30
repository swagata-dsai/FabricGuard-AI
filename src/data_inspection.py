
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd

# ==========================================
# FABRICGUARD AI - AITEX DATASET INSPECTION
# ==========================================

print("\n========================================")
print(" FABRICGUARD AI - DATASET INSPECTION")
print("========================================\n")

# Enter your extracted AITEX dataset folder path here
DATASET_PATH = input("Enter your AITEX dataset folder path: ").strip().strip('"')

root = Path(DATASET_PATH)

if not root.exists() or not root.is_dir():
    print("\nERROR: Dataset folder not found.")
    print("Please check the folder path.")
    raise SystemExit

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

records = []
folder_summary = {}

for current_folder, subfolders, files in os.walk(root):
    current_path = Path(current_folder)

    image_files = [
        file for file in files
        if Path(file).suffix.lower() in image_extensions
    ]

    if image_files:
        folder_summary[str(current_path)] = len(image_files)

    for filename in image_files:
        file_path = current_path / filename

        record = {
            "filename": filename,
            "folder": str(current_path),
            "full_path": str(file_path),
            "format": "",
            "width": None,
            "height": None,
            "status": "Unknown"
        }

        try:
            with Image.open(file_path) as img:
                img.verify()

            with Image.open(file_path) as img:
                record["format"] = img.format
                record["width"] = img.width
                record["height"] = img.height

            record["status"] = "Valid"

        except (UnidentifiedImageError, OSError, ValueError) as error:
            record["status"] = f"Invalid: {error}"

        records.append(record)

if not records:
    print("\nNo image files found in this folder.")
    raise SystemExit

df = pd.DataFrame(records)

# Save audit report
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

report_path = output_folder / "dataset_audit_report.csv"
df.to_csv(report_path, index=False)

print("\n========== INSPECTION SUMMARY ==========")
print(f"Dataset location: {root}")
print(f"Total image files: {len(df)}")
print(f"Valid images: {(df['status'] == 'Valid').sum()}")
print(f"Invalid images: {(df['status'] != 'Valid').sum()}")

print("\n========== IMAGE FOLDERS ==========")

for folder, count in folder_summary.items():
    print(f"{folder} : {count} images")

print("\n========== FILE DETAILS ==========")
print(df[["filename", "format", "width", "height", "status"]].head(10))

print("\n========================================")
print("DATASET INSPECTION COMPLETED")
print(f"Report saved at: {report_path}")
print("========================================")

