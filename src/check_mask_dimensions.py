from PIL import Image
from pathlib import Path

folders = [
    Path(r"A:\AITEX_Dataset\Defect_images"),
    Path(r"A:\AITEX_Dataset\Mask_images")
]

for folder in folders:
    print("\nFOLDER:", folder.name)

    for pattern in ["0044*", "0097*", "0100*"]:
        for file in sorted(folder.glob(pattern)):
            try:
                with Image.open(file) as img:
                    print(file.name, "->", img.size, img.mode)
            except Exception as e:
                print(file.name, "ERROR:", e)