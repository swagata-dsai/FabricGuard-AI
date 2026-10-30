from pathlib import Path
from PIL import Image
import pandas as pd

ROOT = Path(r"A:\AITEX_Dataset")
FOLDERS = ["Defect_images", "NODefect_images", "Mask_images"]

records = []

for folder in FOLDERS:
    folder_path = ROOT / folder

    for path in folder_path.rglob("*"):
        if path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
        }:
            continue

        try:
            with Image.open(path) as img:
                records.append({
                    "folder": folder,
                    "filename": path.name,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode
                })
        except Exception as error:
            print(f"Could not read {path.name}: {error}")

df = pd.DataFrame(records)

Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/dataset_eda.csv", index=False)

print("\nIMAGE DISTRIBUTION")
print(df.groupby("folder").size())

print("\nIMAGE DIMENSIONS")
print(df.groupby(["folder", "width", "height"]).size())

print("\nEDA report saved: outputs/dataset_eda.csv")