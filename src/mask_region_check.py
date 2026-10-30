from PIL import Image
from pathlib import Path
import numpy as np

folder = Path(r"A:\AITEX_Dataset\Mask_images")

files = [
    "0044_019_04_mask1.png",
    "0044_019_04_mask2.png",
    "0097_030_03_mask1.png",
    "0097_030_03_mask2.png"
]

for name in files:
    with Image.open(folder / name) as img:
        gray = np.array(img.convert("L"))

    print("\nFILE:", name)

    for threshold in [1, 100, 150, 200, 240]:
        binary = gray >= threshold
        count = int(binary.sum())

        print(
            f"Threshold {threshold}:",
            count,
            "pixels"
        )

    ys, xs = np.where(gray > 0)

    if len(xs):
        print(
            "Bounding box:",
            (int(xs.min()), int(ys.min()),
             int(xs.max()), int(ys.max()))
        )
    else:
        print("No visible region found")